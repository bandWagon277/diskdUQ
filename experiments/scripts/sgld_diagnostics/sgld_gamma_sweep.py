"""Gamma-parameter sweep for the SGLD polynomial step-size schedule.

Tests how the decay exponent gamma in ε_t = a(b+t)^{-γ} affects chain
mixing, CIF coverage, and interval width. Runs a single method (CR → CR,
which had the best coverage at 0.950 with γ=1.0) across a grid of gamma
values in (0.5, 1.0]. For each gamma, produces the same diagnostics as the
full Table 1 job: R-hat, CIF coverage vs ground truth, CIF width, and Ctd.

The Welling-Teh convergence condition requires γ ∈ (0.5, 1]:
  - γ close to 1.0: fast decay → chain freezes early, narrow within-chain
    variability, high R-hat but potentially good coverage via multi-mode
    envelope.
  - γ close to 0.55: slow decay → chain explores longer, potentially better
    mixing at the cost of higher per-draw variance. The key question is
    whether the improved mixing yields tighter CIs with maintained coverage.

Run from the repository root:

    PYTHONPATH=src python examples/sgld_gamma_sweep.py
    PYTHONPATH=src SGLD_MODE=literal python examples/sgld_gamma_sweep.py
"""
from __future__ import annotations

import copy
import os
import time

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
HORIZON_INDEX = 7

TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 30))
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 2000))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 20))
SGLD_MODE = os.environ.get("SGLD_MODE", "literal")

_DEFAULT_STEPS = {"welling_teh": (3e-7, 3e-9), "literal": (1e-3, 1e-5)}
if SGLD_MODE not in _DEFAULT_STEPS:
    raise ValueError(f"SGLD_MODE must be one of {sorted(_DEFAULT_STEPS)}, got {SGLD_MODE!r}")
_default_init, _default_final = _DEFAULT_STEPS[SGLD_MODE]
SGLD_STEP_SIZE = float(os.environ.get("SGLD_STEP_SIZE", _default_init))
SGLD_FINAL = float(os.environ.get("SGLD_FINAL", _default_final))

# Gamma grid — sweep from the Welling-Teh theoretical minimum (0.55) to 1.0.
GAMMA_GRID_STR = os.environ.get("GAMMA_GRID", "0.55,0.65,0.75,0.85,0.95,1.0")
GAMMA_GRID = [float(g) for g in GAMMA_GRID_STR.split(",")]

from pathlib import Path
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))
OUT_TABLE = OUT_DIR / "sgld_gamma_sweep.txt"
OUT_FIGURE = OUT_DIR / "sgld_gamma_sweep.pdf"
OUT_FIGURE_PNG = OUT_FIGURE.with_suffix(".png")


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
    print(f"SGLD mode: {SGLD_MODE}  step_size {SGLD_STEP_SIZE:.0e} -> {SGLD_FINAL:.0e}")
    print(f"epochs={SGLD_EPOCHS}, chains={N_CHAINS}, samples/chain={SAMPLES_PER_CHAIN}")
    print(f"Gamma grid: {GAMMA_GRID}")

    # Shared teacher (deterministic AdamW).
    print("\n--- training CR teacher (deterministic AdamW) ---", flush=True)
    t0 = time.time()
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=48,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        time_grid=time_grid,
    ).fit(train, feature_cols=all_features)
    print(f"  teacher NLL = {teacher.history.losses[-1]:.4f}  ({time.time()-t0:.1f}s)")

    # --- gamma sweep -------------------------------------------------------
    print(f"\n--- CR -> CR SGLD with gamma sweep ({len(GAMMA_GRID)} values) ---\n")
    hdr = (f"{'gamma':>6s} | {'Ctd1 median [95% CI]':28s} | {'Ctd2 median [95% CI]':28s} | "
           f"{'R-hat (C1/C2/Dev)':22s} | {'CIF cov':8s} | {'CIF wid':8s} | "
           f"{'ε(t=0)':>10s} | {'ε(t=T)':>10s} | time")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    sweep_results = []
    for gamma in GAMMA_GRID:
        t0 = time.time()
        base = DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
            epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=time_grid,
            optimizer="sgld",
            sgld_step_size=SGLD_STEP_SIZE,
            sgld_final_step_size=SGLD_FINAL,
            sgld_gamma=gamma,
            sgld_burnin_epochs=SGLD_EPOCHS // 2,
            sgld_samples_per_chain=SAMPLES_PER_CHAIN,
            sgld_drift_mode=SGLD_MODE,
        )
        sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
        sampler.fit(train, feature_cols=student_features)
        elapsed = time.time() - t0

        # Evaluate.
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

        # Compute actual ε at t=0 and t=T-1 for display.
        from diskd.samplers import SGLD as _SGLD
        steps_per_epoch = max(1, (len(train) + BATCH_SIZE - 1) // BATCH_SIZE)
        total_steps = steps_per_epoch * SGLD_EPOCHS
        _tmp = _SGLD([torch.nn.Parameter(torch.zeros(1))],
                     step_size=SGLD_STEP_SIZE, n_train=len(train),
                     final_step_size=SGLD_FINAL, total_steps=total_steps,
                     gamma=gamma, drift_mode=SGLD_MODE)
        eps_0 = _tmp._current_step_size()
        _tmp._step_count = total_steps - 1
        eps_T = _tmp._current_step_size()

        # Per-chain loss traces.
        loss_traces = [np.asarray(ch.history.losses) for ch in sampler.chains]

        rhat_str = f"{rhat_c1:.2f}/{rhat_c2:.2f}/{rhat_dev:.2f}"
        ci1 = f"{q1[1]:.3f} [{q1[0]:.3f}, {q1[2]:.3f}]"
        ci2 = f"{q2[1]:.3f} [{q2[0]:.3f}, {q2[2]:.3f}]"
        print(f"{gamma:6.2f} | {ci1:28s} | {ci2:28s} | {rhat_str:22s} | "
              f"{cov['overall']:.4f}  | {cov['mean_interval_width']:.4f}  | "
              f"{eps_0:10.2e} | {eps_T:10.2e} | {elapsed:5.0f}s", flush=True)

        sweep_results.append({
            "gamma": gamma,
            "ctd1_q": q1, "ctd2_q": q2,
            "rhat": (rhat_c1, rhat_c2, rhat_dev),
            "cif_coverage": cov["overall"],
            "cif_width": cov["mean_interval_width"],
            "cif_per_cause_cov": cov["per_cause"],
            "eps_0": eps_0, "eps_T": eps_T,
            "ctd1_by_chain": ctd_by_chain[:, :, 0],
            "ctd2_by_chain": ctd_by_chain[:, :, 1],
            "loss_traces": loss_traces,
            "elapsed": elapsed,
        })

    print("=" * len(hdr))

    # --- Save text table ---------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w") as f:
        f.write(f"# SGLD gamma sweep: CR->CR, {SGLD_MODE} mode, "
                f"eps {SGLD_STEP_SIZE:.0e}->{SGLD_FINAL:.0e}, "
                f"{SGLD_EPOCHS} epochs, {N_CHAINS} chains x {SAMPLES_PER_CHAIN} samples\n")
        f.write(f"{'gamma':>6s}\t{'Ctd1_med':>8s}\t{'Ctd1_lo':>8s}\t{'Ctd1_hi':>8s}\t"
                f"{'Ctd2_med':>8s}\t{'Rhat_C1':>8s}\t{'Rhat_C2':>8s}\t{'Rhat_Dev':>8s}\t"
                f"{'CIF_cov':>8s}\t{'CIF_wid':>8s}\t{'eps_0':>10s}\t{'eps_T':>10s}\n")
        for r in sweep_results:
            f.write(f"{r['gamma']:6.2f}\t{r['ctd1_q'][1]:8.4f}\t{r['ctd1_q'][0]:8.4f}\t"
                    f"{r['ctd1_q'][2]:8.4f}\t{r['ctd2_q'][1]:8.4f}\t"
                    f"{r['rhat'][0]:8.3f}\t{r['rhat'][1]:8.3f}\t{r['rhat'][2]:8.3f}\t"
                    f"{r['cif_coverage']:8.4f}\t{r['cif_width']:8.4f}\t"
                    f"{r['eps_0']:10.2e}\t{r['eps_T']:10.2e}\n")
    print(f"\nSweep table saved to {OUT_TABLE}")

    # --- Sweep figure: 3 columns x 2 rows ---------------------------------
    make_sweep_figure(sweep_results, OUT_FIGURE)
    make_sweep_figure(sweep_results, OUT_FIGURE_PNG)
    print(f"Sweep figure saved to {OUT_FIGURE} (and .png)")


def make_sweep_figure(results, out_path):
    gammas = [r["gamma"] for r in results]
    n_gamma = len(gammas)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 0, col 0: R-hat vs gamma.
    ax = axes[0, 0]
    ax.plot(gammas, [r["rhat"][0] for r in results], "o-", label="Ctd1")
    ax.plot(gammas, [r["rhat"][1] for r in results], "s-", label="Ctd2")
    ax.plot(gammas, [r["rhat"][2] for r in results], "^-", label="Deviance")
    ax.axhline(1.1, color="gray", ls="--", lw=0.8, label="R̂ = 1.1")
    ax.set_xlabel("γ")
    ax.set_ylabel("R̂")
    ax.set_title("R-hat vs γ")
    ax.legend(fontsize=7)
    ax.grid(ls=":", alpha=0.5)

    # Row 0, col 1: CIF coverage vs gamma.
    ax = axes[0, 1]
    ax.plot(gammas, [r["cif_coverage"] for r in results], "o-", color="tab:green")
    ax.axhline(0.95, color="gray", ls="--", lw=0.8, label="nominal 0.95")
    ax.set_xlabel("γ")
    ax.set_ylabel("CIF coverage")
    ax.set_title("CIF coverage vs γ")
    ax.legend(fontsize=7)
    ax.grid(ls=":", alpha=0.5)

    # Row 0, col 2: CIF width vs gamma.
    ax = axes[0, 2]
    ax.plot(gammas, [r["cif_width"] for r in results], "o-", color="tab:orange")
    ax.set_xlabel("γ")
    ax.set_ylabel("mean CIF interval width")
    ax.set_title("CIF width vs γ")
    ax.grid(ls=":", alpha=0.5)

    # Row 1, col 0: Ctd1 median + 95% CI band vs gamma.
    ax = axes[1, 0]
    meds = [r["ctd1_q"][1] for r in results]
    los = [r["ctd1_q"][0] for r in results]
    his = [r["ctd1_q"][2] for r in results]
    ax.fill_between(gammas, los, his, alpha=0.3)
    ax.plot(gammas, meds, "o-")
    ax.set_xlabel("γ")
    ax.set_ylabel("Ctd cause 1")
    ax.set_title("Ctd1 median [95% CI] vs γ")
    ax.grid(ls=":", alpha=0.5)

    # Row 1, col 1: per-chain Ctd1 boxplot per gamma.
    ax = axes[1, 1]
    data = [r["ctd1_by_chain"].reshape(-1) for r in results]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    ax.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax.set_xlabel("γ")
    ax.set_ylabel("Ctd cause 1")
    ax.set_title("Ctd1 distribution across draws")
    ax.grid(axis="y", ls=":", alpha=0.5)

    # Row 1, col 2: per-chain loss trace overlay for each gamma.
    ax = axes[1, 2]
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_gamma))
    for idx, r in enumerate(results):
        for ci, trace in enumerate(r["loss_traces"]):
            label = f"γ={r['gamma']:.2f}" if ci == 0 else None
            ax.plot(trace, color=colors[idx], lw=0.5, alpha=0.6, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Per-chain loss traces (all γ overlaid)")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(ls=":", alpha=0.5)

    fig.suptitle(
        f"SGLD gamma sweep — CR→CR, {SGLD_MODE} mode, "
        f"ε₀={SGLD_STEP_SIZE:.0e}→{SGLD_FINAL:.0e}, "
        f"{SGLD_EPOCHS} epochs, {N_CHAINS}×{SAMPLES_PER_CHAIN} draws",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
