"""Adaptive-drift SGLD probe: MSGLD and ASGLD vs literal SGLD.

Compares three drift modes on the CR -> CR method (same as gamma sweep):
  - literal: plain SGLD (our Option B baseline from the production run)
  - msgld:   Momentum SGLD (Kim et al. 2020, Algorithm 1)
  - asgld:   Adam SGLD    (Kim et al. 2020, Algorithm 2)

The adaptive drifts add momentum-based bias that helps chains navigate
narrow ravines and escape local modes faster. If the trace plot from the
literal-mode run shows chains frozen in separate basins, MSGLD/ASGLD may
improve mixing (lower R-hat, tighter CIF width) while maintaining coverage.

Run from the repository root:

    PYTHONPATH=src python examples/sgld_adaptive_probe.py
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    MultiChainSampler,
    competing_risk_c_index,
    effective_sample_size,
    fit_time_grid,
    gelman_rubin_rhat,
    predictive_deviance,
    simulate_competing_risks,
    transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import _iter_samples

SEED = 7
N_SUBJECTS = 5000
NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 256

TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 30))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 2000))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 20))
SGLD_STEP_SIZE = float(os.environ.get("SGLD_STEP_SIZE", 1e-3))
SGLD_FINAL = float(os.environ.get("SGLD_FINAL", 1e-5))
SGLD_GAMMA = float(os.environ.get("SGLD_GAMMA", 1.0))

# Adaptive-drift hyperparameters (paper defaults from Kim et al. 2020).
BIAS_FACTOR_MSGLD = float(os.environ.get("BIAS_FACTOR_MSGLD", 1.0))
BIAS_FACTOR_ASGLD = float(os.environ.get("BIAS_FACTOR_ASGLD", 0.1))
MOMENTUM_BETA = float(os.environ.get("MOMENTUM_BETA", 0.9))
ADAM_BETA2 = float(os.environ.get("ADAM_BETA2", 0.999))

# Modes to probe — extend from env if needed.
MODES_STR = os.environ.get("PROBE_MODES", "literal,msgld,asgld")
MODES = [m.strip() for m in MODES_STR.split(",")]

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))
OUT_TABLE = OUT_DIR / "sgld_adaptive_probe.txt"
OUT_FIGURE = OUT_DIR / "sgld_adaptive_probe.pdf"
OUT_FIGURE_PNG = OUT_FIGURE.with_suffix(".png")


def _sgld_kwargs(mode):
    kw = dict(
        optimizer="sgld",
        sgld_step_size=SGLD_STEP_SIZE,
        sgld_final_step_size=SGLD_FINAL,
        sgld_gamma=SGLD_GAMMA,
        sgld_burnin_epochs=SGLD_EPOCHS // 2,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
        sgld_drift_mode=mode,
        sgld_momentum_beta=MOMENTUM_BETA,
        sgld_adam_beta2=ADAM_BETA2,
    )
    if mode == "msgld":
        kw["sgld_bias_factor"] = BIAS_FACTOR_MSGLD
    elif mode == "asgld":
        kw["sgld_bias_factor"] = BIAS_FACTOR_ASGLD
    return kw


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    data = simulate_competing_risks(n=N_SUBJECTS, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()

    time_grid = fit_time_grid(train["duration"].values, NUM_DURATIONS)
    test_idx = transform_durations(test_durations, time_grid)
    true_cif = true_cif_at_grid(test, time_grid, num_risks=NUM_RISKS)

    print(f"Setup: N={N_SUBJECTS}, train={len(train)}, test={len(test)}")
    print(f"  step_size {SGLD_STEP_SIZE:.0e} -> {SGLD_FINAL:.0e}, gamma={SGLD_GAMMA}")
    print(f"  epochs={SGLD_EPOCHS}, chains={N_CHAINS}, samples/chain={SAMPLES_PER_CHAIN}")
    print(f"  Modes: {MODES}")
    print(f"  MSGLD bias_factor={BIAS_FACTOR_MSGLD}, beta1={MOMENTUM_BETA}")
    print(f"  ASGLD bias_factor={BIAS_FACTOR_ASGLD}, beta1={MOMENTUM_BETA}, beta2={ADAM_BETA2}")

    print("\n--- training CR teacher ---", flush=True)
    t0 = time.time()
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=48,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid,
    ).fit(train, feature_cols=all_features)
    print(f"  teacher NLL = {teacher.history.losses[-1]:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n--- CR -> CR SGLD with adaptive-drift probe ({len(MODES)} modes) ---\n")
    hdr = (f"{'mode':>8s} | {'Ctd1 median [95% CI]':28s} | {'Ctd2 median [95% CI]':28s} | "
           f"{'R-hat (C1/C2/Dev)':22s} | {'CIF cov':8s} | {'CIF wid':8s} | time")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    probe_results = []
    for mode in MODES:
        t0 = time.time()
        base = DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
            epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=time_grid,
            **_sgld_kwargs(mode),
        )
        sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
        sampler.fit(train, feature_cols=student_features)
        elapsed = time.time() - t0

        lead = sampler.lead_chain
        ctd_by_chain = np.zeros((N_CHAINS, SAMPLES_PER_CHAIN, NUM_RISKS))
        dev_by_chain = np.zeros((N_CHAINS, SAMPLES_PER_CHAIN))
        cifs = []
        for c_idx, chain in enumerate(sampler.chains):
            for s_idx, (_, m) in enumerate(_iter_samples(lead, chain.posterior_samples)):
                ctd_by_chain[c_idx, s_idx] = competing_risk_c_index(
                    m.predict_cif(test), test_durations, test_events)
                dev_by_chain[c_idx, s_idx] = predictive_deviance(
                    m.predict_interval_probs(test).numpy(), test_idx, test_events)
                cifs.append(np.asarray(m.predict_cif(test)))
        cif_arr = np.stack(cifs, axis=0)
        cif_q = np.quantile(cif_arr, [0.025, 0.5, 0.975], axis=0)
        cov = coverage_from_quantiles(cif_q, true_cif)

        rhat_c1 = gelman_rubin_rhat(ctd_by_chain[:, :, 0])
        rhat_c2 = gelman_rubin_rhat(ctd_by_chain[:, :, 1])
        rhat_dev = gelman_rubin_rhat(dev_by_chain)
        ctd1_flat = ctd_by_chain[:, :, 0].reshape(-1)
        ctd2_flat = ctd_by_chain[:, :, 1].reshape(-1)
        q1 = np.quantile(ctd1_flat, [0.025, 0.5, 0.975])
        q2 = np.quantile(ctd2_flat, [0.025, 0.5, 0.975])

        loss_traces = [np.asarray(ch.history.losses) for ch in sampler.chains]

        rhat_str = f"{rhat_c1:.2f}/{rhat_c2:.2f}/{rhat_dev:.2f}"
        ci1 = f"{q1[1]:.3f} [{q1[0]:.3f}, {q1[2]:.3f}]"
        ci2 = f"{q2[1]:.3f} [{q2[0]:.3f}, {q2[2]:.3f}]"
        print(f"{mode:>8s} | {ci1:28s} | {ci2:28s} | {rhat_str:22s} | "
              f"{cov['overall']:.4f}  | {cov['mean_interval_width']:.4f}  | {elapsed:5.0f}s",
              flush=True)

        probe_results.append({
            "mode": mode, "ctd1_q": q1, "ctd2_q": q2,
            "rhat": (rhat_c1, rhat_c2, rhat_dev),
            "cif_coverage": cov["overall"],
            "cif_width": cov["mean_interval_width"],
            "ctd1_by_chain": ctd_by_chain[:, :, 0],
            "loss_traces": loss_traces,
            "elapsed": elapsed,
        })

    print("=" * len(hdr))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w") as f:
        f.write(f"# Adaptive SGLD probe: CR->CR, eps {SGLD_STEP_SIZE:.0e}->{SGLD_FINAL:.0e}, "
                f"gamma={SGLD_GAMMA}, {SGLD_EPOCHS} epochs, {N_CHAINS}x{SAMPLES_PER_CHAIN}\n")
        for r in probe_results:
            f.write(f"{r['mode']}\t{r['ctd1_q'][1]:.4f}\t"
                    f"{r['rhat'][0]:.3f}\t{r['rhat'][1]:.3f}\t{r['rhat'][2]:.3f}\t"
                    f"{r['cif_coverage']:.4f}\t{r['cif_width']:.4f}\n")
    print(f"\nProbe table saved to {OUT_TABLE}")

    make_probe_figure(probe_results, OUT_FIGURE)
    make_probe_figure(probe_results, OUT_FIGURE_PNG)
    print(f"Probe figure saved to {OUT_FIGURE} (and .png)")


def make_probe_figure(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.0 * n), squeeze=False)
    burnin = SGLD_EPOCHS // 2
    for row, r in enumerate(results):
        mode = r["mode"]
        # Col 0: per-chain training loss.
        ax = axes[row, 0]
        for ci, trace in enumerate(r["loss_traces"]):
            ax.plot(trace, lw=0.7, alpha=0.8, label=f"chain {ci}")
        if len(r["loss_traces"][0]) > burnin:
            ax.axvspan(burnin, len(r["loss_traces"][0]) - 1,
                       color="#a1d99b", alpha=0.25, lw=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title(f"{mode}: per-chain loss")
        ax.grid(ls=":", alpha=0.5)
        if row == 0:
            ax.legend(fontsize=6, ncol=2)

        # Col 1: per-chain Ctd1 trace.
        ax = axes[row, 1]
        arr = r["ctd1_by_chain"]
        for ci in range(arr.shape[0]):
            ax.plot(arr[ci], "o-", markersize=2, lw=0.8, alpha=0.85)
        ax.set_xlabel("draw (within chain)")
        ax.set_ylabel("Ctd cause 1")
        ax.set_title(f"{mode}: Ctd1 trace  (R̂={r['rhat'][0]:.2f})")
        ax.grid(ls=":", alpha=0.5)

        # Col 2: summary bar chart.
        ax = axes[row, 2]
        ax.bar(["CIF cov", "CIF width"],
               [r["cif_coverage"], r["cif_width"]],
               color=["#2ca02c", "#ff7f0e"])
        ax.axhline(0.95, color="gray", ls="--", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{mode}: coverage={r['cif_coverage']:.3f}, width={r['cif_width']:.3f}")

    fig.suptitle(
        f"Adaptive-drift SGLD probe — CR→CR, "
        f"ε₀={SGLD_STEP_SIZE:.0e}, γ={SGLD_GAMMA}, "
        f"{SGLD_EPOCHS} epochs, {N_CHAINS}×{SAMPLES_PER_CHAIN}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
