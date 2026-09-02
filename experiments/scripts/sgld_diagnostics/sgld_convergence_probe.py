"""SGLD convergence / chain-length study for warm-start DiSKD.

Addresses the #1 open question on the de-leaked pooled run: are the warm-start
chains converged, or are the pooled intervals finite-time transient jitter?

For ONE method (CR->CR, the headline) plus the Internal-CR baseline, we run the
warm-start chains at increasing SGLD-epoch budgets and watch whether the
Gelman-Rubin R-hat, ESS, pooled CIF coverage, width, and the pooled-median's
distance from the AdamW MAP STABILIZE.

  - If R-hat -> ~1.1 and coverage/width plateau with longer chains:
    the intervals are genuine posterior draws -> "credible intervals".
  - If they keep drifting: the method is a finite-time warm-start ensemble;
    relabel accordingly (still valid, still shows the distillation lift).

Fixed eps = 2e-4 (no decay), matching the de-leaked production config. The only
swept axis is the SGLD epoch budget. Run from the repo root:

    PYTHONPATH=src python examples/sgld_convergence_probe.py

Env knobs: BUDGETS (csv epochs), SEEDS, METHODS, N_CHAINS, SAMPLES_PER_CHAIN,
EPS0, TEACHER_N, STUDENT_N, TEST_N, TEACHER_FEATURE_QUALITY, OUT_DIR, DEVICE.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from diskd import (
    WarmStartMultiChainSampler,
    competing_risk_c_index,
    fit_time_grid,
    predictive_deviance,
    simulate_competing_risk_cohorts,
    transform_durations,
)
from diskd._ground_truth import true_cif_at_grid
from diskd.uncertainty import (
    _iter_samples,
    effective_sample_size,
    gelman_rubin_rhat,
)

# Reuse the validated builders from the calibration probe.
import uq_calibration_probe as P

# ---------- Config ----------
BUDGETS = [int(b) for b in os.environ.get("BUDGETS", "500,2000,5000").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
METHODS = [m.strip() for m in os.environ.get("METHODS", "internal,cr_to_cr").split(",")]
N_CHAINS = int(os.environ.get("N_CHAINS", 5))
SAMPLES_PER_CHAIN = int(os.environ.get("SAMPLES_PER_CHAIN", 500))
EPS0 = float(os.environ.get("EPS0", 2e-4))
EPS_T = float(os.environ.get("EPS_T", 2e-4))  # fixed step (no decay)
GAMMA = float(os.environ.get("GAMMA", 0.55))
NUM_RISKS = P.NUM_RISKS
NUM_DURATIONS = P.NUM_DURATIONS
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent.parent / "responses_convergence")))

# Push shared config into the reused probe module so its builders match.
P.TEACHER_N = int(os.environ.get("TEACHER_N", 5000))
P.STUDENT_N = int(os.environ.get("STUDENT_N", 500))
P.TEST_N = int(os.environ.get("TEST_N", 500))
P.TEACHER_FEATURE_QUALITY = os.environ.get("TEACHER_FEATURE_QUALITY", "full")
P.N_CHAINS = N_CHAINS
P.SAMPLES_PER_CHAIN = SAMPLES_PER_CHAIN
P.EPS0 = EPS0
P.EPS_T = EPS_T
P.GAMMA = GAMMA
P.BURNIN_EPOCHS = 0


def map_distance(pooled_median_cif, adam_map_cif):
    """Mean abs distance between pooled posterior-median CIF and the MAP CIF."""
    return float(np.mean(np.abs(pooled_median_cif - adam_map_cif)))


def run_one(method, budget, teachers, cohorts, time_grid, pretrained, seed,
            true_cif, adam_map_cif):
    P.SGLD_EPOCHS = budget
    chains = P.run_sgld(method, teachers, cohorts, time_grid, pretrained, seed)
    pooled = np.concatenate([c["cifs"] for c in chains], axis=0)
    pe = P.per_entry(pooled, true_cif)
    rhat, ess = P.chain_diagnostics(chains)
    test = cohorts.test
    ctd = competing_risk_c_index(pe["median"], test["duration"].to_numpy(),
                                 test["event"].to_numpy())
    dev = float(np.mean([np.mean(c["dev"]) for c in chains]))
    return {
        "cov": float(pe["inside"].mean()),
        "wid": float(pe["width"].mean()),
        "ctd1": float(ctd[0]), "ctd2": float(ctd[1]), "dev": dev,
        "rhat": rhat, "ess": ess,
        "map_dist": map_distance(pe["median"], adam_map_cif),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print("=== SGLD convergence / chain-length study ===")
    print(f"Budgets: {BUDGETS}  Seeds: {SEEDS}  Methods: {METHODS}")
    print(f"Fixed eps={EPS0:.0e} (no decay), {N_CHAINS} chains x {SAMPLES_PER_CHAIN} draws")
    print(f"Cell: TEACHER_N={P.TEACHER_N} STUDENT_N={P.STUDENT_N} quality={P.TEACHER_FEATURE_QUALITY}")

    # results[(method, budget)] -> list over seeds of metric dicts
    results = {}
    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n=== Seed {seed} ===", flush=True)
        cohorts, time_grid = P.setup_cohorts(seed)
        teachers = P.train_teachers(cohorts, time_grid, METHODS)
        true_cif = true_cif_at_grid(cohorts.test, time_grid, num_risks=NUM_RISKS)
        for method in METHODS:
            adam = P.run_adamw(method, teachers, cohorts, time_grid, seed)
            adam_map_cif = adam["cifs"][0]  # first restart = the warm-start MAP
            for budget in BUDGETS:
                t0 = time.time()
                r = run_one(method, budget, teachers, cohorts, time_grid,
                            adam["pretrained"], seed, true_cif, adam_map_cif)
                results.setdefault((method, budget), []).append(r)
                print(f"  {method:12s} ep={budget:5d}  Rhat={r['rhat']:.2f} ESS={r['ess']:.0f} "
                      f"cov={r['cov']:.3f} wid={r['wid']:.3f} Ctd1={r['ctd1']:.3f} "
                      f"dev={r['dev']:.2f} mapdist={r['map_dist']:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
        print(f"  seed {seed} done in {time.time()-t_seed:.0f}s", flush=True)

    # ---- Aggregate (median across seeds) ----
    def med(method, budget, key):
        return float(np.nanmedian([r[key] for r in results[(method, budget)]]))

    # ---- Save npz ----
    save = {"budgets": np.array(BUDGETS), "seeds": np.array(SEEDS)}
    for (method, budget), lst in results.items():
        for key in ["cov", "wid", "ctd1", "ctd2", "dev", "rhat", "ess", "map_dist"]:
            save[f"{method}_ep{budget}_{key}"] = np.array([r[key] for r in lst])
    np.savez(OUT_DIR / "sgld_convergence_data.npz", **save)

    # ---- Figure: diagnostics vs budget ----
    panels = [("rhat", "Gelman-Rubin R-hat", 1.1),
              ("cov", "Pooled CIF coverage", 0.95),
              ("wid", "Pooled CIF width", None),
              ("map_dist", "Pooled median distance from MAP", None),
              ("ctd1", "SGLD Ctd cause 1", None),
              ("dev", "SGLD deviance", None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, ref) in zip(axes.ravel(), panels):
        for method in METHODS:
            ys = [med(method, b, key) for b in BUDGETS]
            ax.plot(BUDGETS, ys, marker="o", lw=2, label=method)
        if ref is not None:
            ax.axhline(ref, color="red", ls="--", alpha=0.5,
                       label=f"target {ref}")
        ax.set_title(title)
        ax.set_xlabel("SGLD epochs (budget)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Warm-start SGLD convergence vs chain-length budget "
                 f"(fixed eps={EPS0:.0e}, {N_CHAINS} chains, median over {len(SEEDS)} seeds)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "sgld_convergence.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "sgld_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Markdown report ----
    md = ["# SGLD convergence / chain-length study", "",
          f"**Cell:** TEACHER_N={P.TEACHER_N}, STUDENT_N={P.STUDENT_N}, "
          f"quality={P.TEACHER_FEATURE_QUALITY}. Seeds={SEEDS}.",
          f"**SGLD:** fixed eps={EPS0:.0e} (no decay), {N_CHAINS} chains x "
          f"{SAMPLES_PER_CHAIN} draws, warm-start. Budgets={BUDGETS} epochs.", "",
          "Median across seeds. R-hat -> ~1.1 and a plateau in coverage/width/"
          "map-distance ==> converged credible intervals; continued drift ==> "
          "finite-time warm-start ensemble.", "",
          "| Method | Budget | R-hat | ESS | CIF cov | CIF wid | Ctd1 | Dev | dist(MAP) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for method in METHODS:
        for b in BUDGETS:
            md.append(f"| {method} | {b} | {med(method,b,'rhat'):.2f} | "
                      f"{med(method,b,'ess'):.0f} | {med(method,b,'cov'):.3f} | "
                      f"{med(method,b,'wid'):.3f} | {med(method,b,'ctd1'):.3f} | "
                      f"{med(method,b,'dev'):.2f} | {med(method,b,'map_dist'):.3f} |")
    md += ["", "![convergence](sgld_convergence.png)", "",
           "## Reading", "",
           "- **R-hat panel:** crossing toward 1.1 as budget grows = chains mixing.",
           "- **Coverage / width:** a plateau = the posterior is being characterised, "
           "not endlessly inflating.",
           "- **dist(MAP):** if this keeps growing with budget, chains are wandering "
           "away from the MAP (transient drift), not exploring a stationary basin."]
    (OUT_DIR / "sgld_convergence_report.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved data + figure + report to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
