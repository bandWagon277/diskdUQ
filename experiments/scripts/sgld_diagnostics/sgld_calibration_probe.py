"""Warm-start SGLD calibration probe.

Goal: use AdamW for the point estimate, then run SGLD from that solution
purely for uncertainty calibration. Sweeps step-size and gamma to find the
combination that gives good CIF coverage with tight intervals, without
degrading Ctd or deviance too much from AdamW.

Simulation design (matches original/DiscreteSurvKD-main separate-cohort):
  - Teacher: large independent cohort, reduced features x1..x8
    (misses shared signal block x9..x12)
  - Student: small independent cohort, full features x1..x12
  - Test: 25% split from student cohort

    PYTHONPATH=src python examples/sgld_calibration_probe.py
"""
from __future__ import annotations

import copy, os, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    DiSKDStudent, DiscreteSurvivalModel, WarmStartMultiChainSampler,
    competing_risk_c_index, fit_time_grid, predictive_deviance,
    simulate_competing_risks, transform_durations,
)
from diskd._ground_truth import coverage_from_quantiles, true_cif_at_grid
from diskd.uncertainty import _iter_samples

SEED = 42
TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
STUDENT_N = int(os.environ.get("STUDENT_N", 500))
NUM_RISKS = 2
NUM_DURATIONS = 12
TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 32
BATCH_SIZE = 64
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 100))
ADAMW_EPOCHS = int(os.environ.get("ADAMW_EPOCHS", 50))
N_ADAM_RESTARTS = int(os.environ.get("N_ADAM_RESTARTS", 20))
METHODS = [m.strip() for m in os.environ.get("METHODS", "internal,competing").split(",")]
SGLD_EPOCHS = int(os.environ.get("SGLD_EPOCHS", 500))
BURNIN_EPOCHS = int(os.environ.get("BURNIN_EPOCHS", 100))
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))

CONFIGS_STR = os.environ.get("CONFIGS", "")

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "responses"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(_DEFAULT_DIR)))


def parse_configs():
    if CONFIGS_STR:
        configs = []
        for tok in CONFIGS_STR.split(";"):
            parts = tok.strip().split(",")
            eps0 = float(parts[0])
            eps_f = float(parts[1])
            gamma = float(parts[2])
            label = parts[3] if len(parts) > 3 else f"e{eps0:.0e}_g{gamma}"
            configs.append((eps0, eps_f, gamma, label))
        return configs
    return [
        (1e-3, 1e-5, 0.55, "e3_g55"),
        (5e-4, 5e-6, 0.55, "e3.5_g55"),
        (1e-4, 1e-6, 0.55, "e4_g55"),
        (5e-4, 5e-4, 0.55, "e3.5_const"),
        (1e-4, 1e-4, 0.55, "e4_const"),
    ]


def evaluate_chains(sampler, test, true_cif, test_dur, test_ev, test_idx):
    lead = sampler.lead_chain
    chain_results = []
    for c_idx, chain in enumerate(sampler.chains):
        cifs, ctds, devs = [], [], []
        for _, m in _iter_samples(lead, chain.posterior_samples):
            cif = np.asarray(m.predict_cif(test))
            cifs.append(cif)
            ctds.append(competing_risk_c_index(cif, test_dur, test_ev))
            devs.append(predictive_deviance(
                m.predict_interval_probs(test).numpy(), test_idx, test_ev))
        if not cifs:
            chain_results.append(None)
            continue
        arr = np.stack(cifs, axis=0)
        q = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
        cov = coverage_from_quantiles(q, true_cif)
        ctd_arr = np.array(ctds)
        dev_arr = np.array(devs)
        chain_results.append({
            "n": len(cifs),
            "cif_cov": cov["overall"],
            "cif_wid": cov["mean_interval_width"],
            "ctd1": np.quantile(ctd_arr[:, 0], [0.025, 0.5, 0.975]),
            "ctd2": np.quantile(ctd_arr[:, 1], [0.025, 0.5, 0.975]),
            "dev": np.quantile(dev_arr, [0.025, 0.5, 0.975]),
            "loss": chain.history.losses[-1],
        })
    return chain_results


def best_chain(results):
    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    if not valid:
        return None, None
    idx, r = min(valid, key=lambda x: x[1]["dev"][1])
    return idx, r


def fmt(q):
    return f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    configs = parse_configs()
    print(f"Configs: {len(configs)}, chains={N_CHAINS}, draws/chain={SAMPLES_PER_CHAIN}, "
          f"SGLD epochs={SGLD_EPOCHS}, burnin={BURNIN_EPOCHS}")

    # --- Separate-cohort simulation ---
    # Teacher: large cohort, reduced features x1..x8 (misses shared signal x9..x12)
    # Student: small independent cohort, full features x1..x12
    teacher_data = simulate_competing_risks(n=TEACHER_N, seed=SEED, censor_max=0.05)
    student_data = simulate_competing_risks(n=STUDENT_N, seed=SEED + 1, censor_max=0.05)
    train, test = train_test_split(student_data, test_size=0.25, random_state=SEED)

    all_features = [c for c in teacher_data.columns if c.startswith("x")]
    teacher_features = all_features[:8]
    student_features = all_features  # full x1..x12

    # Time grid from combined durations
    combined_dur = np.concatenate([teacher_data["duration"].values, train["duration"].values])
    time_grid = fit_time_grid(combined_dur, NUM_DURATIONS)

    test_dur = test["duration"].to_numpy()
    test_ev = test["event"].to_numpy()
    test_idx = transform_durations(test_dur, time_grid)
    true_cif = true_cif_at_grid(test, time_grid, num_risks=NUM_RISKS)

    print(f"Teacher: N={len(teacher_data)}, features={len(teacher_features)} ({teacher_features[0]}..{teacher_features[-1]})")
    print(f"Student: train={len(train)}, test={len(test)}, features={len(student_features)} ({student_features[0]}..{student_features[-1]})")

    # Teacher (trained on its own large cohort with reduced features)
    t0 = time.time()
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=TEACHER_HIDDEN,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, time_grid=time_grid,
    ).fit(teacher_data, feature_cols=teacher_features)
    print(f"Teacher: loss={teacher.history.losses[-1]:.4f} ({time.time()-t0:.0f}s)")

    # AdamW multi-restart: gives Adam CI from initialization variability and
    # also provides the warm-start state for SGLD (first restart, seed=SEED).
    # We also keep the per-restart predicted CIFs to compute Adam-style CIF
    # coverage / interval width using the same quantile-based definition as for
    # SGLD posterior draws (different "replicate" source: init randomness).
    t0 = time.time()
    adam_ctd1, adam_ctd2, adam_dev = [], [], []
    adam_cifs = []
    pretrained = None
    cif_a_first = None
    for r in range(N_ADAM_RESTARTS):
        torch.manual_seed(SEED + r)
        np.random.seed(SEED + r)
        a = DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
            epochs=ADAMW_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=time_grid, optimizer="adamw",
        ).fit(train, feature_cols=student_features)
        cif_r = np.asarray(a.predict_cif(test))
        ctd_r = competing_risk_c_index(cif_r, test_dur, test_ev)
        dev_r = predictive_deviance(a.predict_interval_probs(test).numpy(), test_idx, test_ev)
        adam_ctd1.append(float(ctd_r[0]))
        adam_ctd2.append(float(ctd_r[1]))
        adam_dev.append(float(dev_r))
        adam_cifs.append(cif_r)
        if r == 0:
            pretrained = {k: v.clone() for k, v in a.net.state_dict().items()}
            cif_a_first = cif_r

    adam_ctd1_q = np.quantile(adam_ctd1, [0.025, 0.5, 0.975])
    adam_ctd2_q = np.quantile(adam_ctd2, [0.025, 0.5, 0.975])
    adam_dev_q = np.quantile(adam_dev, [0.025, 0.5, 0.975])
    ctd_a = (adam_ctd1_q[1], adam_ctd2_q[1])
    dev_a = adam_dev_q[1]
    # Adam CIF coverage and width from the multi-restart spread
    adam_cif_arr = np.stack(adam_cifs, axis=0)  # [N_restarts, N, J, K]
    adam_cif_q = np.quantile(adam_cif_arr, [0.025, 0.5, 0.975], axis=0)
    adam_cif_cov = coverage_from_quantiles(adam_cif_q, true_cif)
    print(f"AdamW ({N_ADAM_RESTARTS} restarts, {time.time()-t0:.0f}s):")
    print(f"  Ctd1    = {fmt(adam_ctd1_q)}")
    print(f"  Ctd2    = {fmt(adam_ctd2_q)}")
    print(f"  Dev     = {fmt(adam_dev_q)}")
    print(f"  CIF cov = {adam_cif_cov['overall']:.4f}")
    print(f"  CIF wid = {adam_cif_cov['mean_interval_width']:.4f}")
    print(f"  warm-start = seed {SEED}, Ctd1={adam_ctd1[0]:.4f} Ctd2={adam_ctd2[0]:.4f} dev={adam_dev[0]:.3f}")

    # Header
    print(f"\n{'config':>12s} | {'best':>4s} | {'Ctd1':>28s} | {'Ctd2':>28s} | "
          f"{'Deviance':>28s} | {'CIF cov':>7s} | {'CIF wid':>7s} | {'loss':>7s} | {'time':>5s}")
    print(f"  {'-'*150}")

    all_results = {}
    for eps0, eps_f, gamma, label in configs:
        t0 = time.time()
        base = DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=STUDENT_HIDDEN,
            epochs=SGLD_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=time_grid, optimizer="sgld",
            sgld_step_size=eps0, sgld_final_step_size=eps_f if eps_f != eps0 else None,
            sgld_gamma=gamma, sgld_drift_mode="literal",
            sgld_burnin_epochs=BURNIN_EPOCHS,
            sgld_samples_per_chain=SAMPLES_PER_CHAIN,
        )
        sampler = WarmStartMultiChainSampler(
            base, pretrained, n_chains=N_CHAINS,
            seeds=list(range(100, 100 + N_CHAINS)))
        sampler.fit(train, feature_cols=student_features)
        elapsed = time.time() - t0

        chains = evaluate_chains(sampler, test, true_cif, test_dur, test_ev, test_idx)
        bc_idx, bc = best_chain(chains)
        all_results[label] = {"chains": chains, "best_idx": bc_idx, "best": bc,
                              "sampler": sampler, "config": (eps0, eps_f, gamma)}

        if bc is not None:
            print(f"  {label:>12s} | {bc_idx:>4d} | {fmt(bc['ctd1']):>28s} | {fmt(bc['ctd2']):>28s} | "
                  f"{fmt(bc['dev']):>28s} | {bc['cif_cov']:>7.3f} | {bc['cif_wid']:>7.3f} | "
                  f"{bc['loss']:>7.2f} | {elapsed:>4.0f}s", flush=True)
            for ci, cr in enumerate(chains):
                if cr is None:
                    continue
                tag = " <--best" if ci == bc_idx else ""
                print(f"    chain {ci}: Ctd1={cr['ctd1'][1]:.3f} Ctd2={cr['ctd2'][1]:.3f} "
                      f"dev={cr['dev'][1]:.2f} cov={cr['cif_cov']:.3f} wid={cr['cif_wid']:.3f} "
                      f"loss={cr['loss']:.2f}{tag}", flush=True)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = OUT_DIR / "sgld_calibration_probe.txt"
    with out_txt.open("w") as f:
        f.write(f"# AdamW ({N_ADAM_RESTARTS} restarts):\n")
        f.write(f"#   Ctd1={fmt(adam_ctd1_q)}\n")
        f.write(f"#   Ctd2={fmt(adam_ctd2_q)}\n")
        f.write(f"#   Dev ={fmt(adam_dev_q)}\n")
        f.write(f"#   CIF cov={adam_cif_cov['overall']:.4f}\n")
        f.write(f"#   CIF wid={adam_cif_cov['mean_interval_width']:.4f}\n")
        for label, mr in all_results.items():
            bc = mr["best"]
            cfg = mr["config"]
            if bc:
                f.write(f"{label}\teps={cfg[0]:.0e}->{cfg[1]:.0e}\tgamma={cfg[2]}\t"
                        f"best={mr['best_idx']}\tctd1={bc['ctd1'][1]:.4f}\tctd2={bc['ctd2'][1]:.4f}\t"
                        f"dev={bc['dev'][1]:.3f}\tcov={bc['cif_cov']:.4f}\twid={bc['cif_wid']:.4f}\n")
                for ci, cr in enumerate(mr["chains"]):
                    if cr is None:
                        continue
                    f.write(f"  chain{ci}\tctd1={cr['ctd1'][1]:.4f}\tctd2={cr['ctd2'][1]:.4f}\t"
                            f"dev={cr['dev'][1]:.3f}\tcov={cr['cif_cov']:.4f}\twid={cr['cif_wid']:.4f}\n")
    print(f"\nSaved to {out_txt}")

    # Figure: loss traces + coverage/width per config
    make_figure(all_results, ctd_a, dev_a)


def make_figure(all_results, ctd_adamw, dev_adamw):
    n_cfg = len(all_results)
    fig, axes = plt.subplots(n_cfg, 3, figsize=(15, 3.5 * n_cfg), squeeze=False)

    for row, (label, mr) in enumerate(all_results.items()):
        sampler = mr["sampler"]
        chains_res = mr["chains"]
        cfg = mr["config"]

        # Col 0: loss traces
        ax = axes[row, 0]
        for ci, chain in enumerate(sampler.chains):
            ax.plot(chain.history.losses, lw=0.5, alpha=0.7, label=f"c{ci}")
        ax.axvline(BURNIN_EPOCHS, color="red", ls="--", lw=0.8, label="burnin end")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{label}: eps={cfg[0]:.0e}->{cfg[1]:.0e}, g={cfg[2]}")
        ax.legend(fontsize=5, ncol=3)
        ax.grid(ls=":", alpha=0.4)

        # Col 1: per-chain coverage
        ax = axes[row, 1]
        covs = [cr["cif_cov"] if cr else 0 for cr in chains_res]
        colors = ["#2ca02c" if c > 0.5 else "#d62728" for c in covs]
        ax.bar(range(len(covs)), covs, color=colors)
        ax.axhline(0.95, color="gray", ls="--", lw=0.8, label="target 0.95")
        ax.set_xlabel("chain")
        ax.set_ylabel("CIF coverage")
        ax.set_title(f"Per-chain coverage")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

        # Col 2: per-chain width
        ax = axes[row, 2]
        wids = [cr["cif_wid"] if cr else 0 for cr in chains_res]
        ax.bar(range(len(wids)), wids, color="#ff7f0e")
        ax.set_xlabel("chain")
        ax.set_ylabel("CIF width")
        ax.set_title(f"Per-chain CI width")

    fig.suptitle(f"Warm-start SGLD Calibration — {SGLD_EPOCHS}ep, burnin={BURNIN_EPOCHS}, "
                 f"{N_CHAINS}x{SAMPLES_PER_CHAIN} draws\n"
                 f"AdamW: Ctd1={ctd_adamw[0]:.3f}, Ctd2={ctd_adamw[1]:.3f}, dev={dev_adamw:.2f}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "sgld_calibration_probe.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
