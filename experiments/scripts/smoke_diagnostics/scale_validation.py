"""Scale-up validation for Bayesian DiSKD.

Runs the Bayesian DiSKD pipeline at three problem sizes (small / medium /
large) and reports posterior-vs-ensemble credible-interval ratios for:
  - CIF (subject x cause x time averaged)
  - cause-specific Ctd
  - predictive deviance

Hypothesis: as N_subjects, n_chains*samples_per_chain, and ensemble_size
all grow, the aggregate-metric ratios (Ctd, deviance) should approach 1.0,
confirming that the residual asymmetry observed at small scale is finite-
sample noise rather than systematic posterior misspecification.

Run from the repository root with:

    PYTHONPATH=src python examples/scale_validation.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import os
import torch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

from diskd import (
    DiSKDStudent,
    DiscreteSurvivalModel,
    MultiChainSampler,
    competing_risk_c_index,
    credible_intervals,
    interval_width,
    posterior_predictions,
    predictive_deviance,
    simulate_competing_risks,
    transform_durations,
)
from diskd.uncertainty import _iter_samples


@dataclass
class Setting:
    name: str
    n_subjects: int
    sgld_epochs: int
    samples_per_chain: int
    ensemble_size: int
    ensemble_epochs: int


SETTINGS = [
    Setting("small",  n_subjects=500,  sgld_epochs=20, samples_per_chain=10, ensemble_size=10, ensemble_epochs=10),
    Setting("medium", n_subjects=1000, sgld_epochs=30, samples_per_chain=15, ensemble_size=20, ensemble_epochs=15),
    Setting("large",  n_subjects=2000, sgld_epochs=40, samples_per_chain=20, ensemble_size=30, ensemble_epochs=20),
]

NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 128
N_CHAINS = 5
SEED = 7


def run_setting(s: Setting):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = simulate_competing_risks(n=s.n_subjects, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()

    t0 = time.time()
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=48,
        epochs=20, batch_size=BATCH_SIZE, device=DEVICE,
    ).fit(train, feature_cols=all_features)
    teacher_time = time.time() - t0

    # SGLD multi-chain
    t0 = time.time()
    base = DiSKDStudent(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
        epochs=s.sgld_epochs, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=teacher.time_grid, optimizer="sgld",
        sgld_burnin_epochs=s.sgld_epochs // 2,
        sgld_samples_per_chain=s.samples_per_chain,
    )
    sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
    sampler.fit(train, feature_cols=student_features)
    sgld_time = time.time() - t0
    lead = sampler.lead_chain
    test_idx = transform_durations(test_durations, lead.time_grid)

    # Ensemble baseline
    t0 = time.time()
    ensemble = []
    for j in range(s.ensemble_size):
        torch.manual_seed(1000 + j)
        np.random.seed(1000 + j)
        m = DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
            epochs=s.ensemble_epochs, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=teacher.time_grid,
        ).fit(train, feature_cols=student_features)
        ensemble.append(m)
    ens_time = time.time() - t0

    # Per-metric evaluation
    cif_sgld = posterior_predictions(lead, sampler.posterior_samples, test, "cif")
    cif_ens = np.stack([m.predict_cif(test) for m in ensemble], axis=0)
    ctd_sgld = []
    dev_sgld = []
    for _, m in _iter_samples(lead, sampler.posterior_samples):
        ctd_sgld.append(competing_risk_c_index(m.predict_cif(test), test_durations, test_events))
        dev_sgld.append(predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_events))
    ctd_sgld = np.stack(ctd_sgld, axis=0)
    dev_sgld = np.array(dev_sgld)
    ctd_ens = np.stack(
        [competing_risk_c_index(m.predict_cif(test), test_durations, test_events) for m in ensemble],
        axis=0,
    )
    dev_ens = np.array(
        [predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_events) for m in ensemble]
    )

    q = (0.025, 0.5, 0.975)
    cif_sgld_q = credible_intervals(cif_sgld, q)
    cif_ens_q = credible_intervals(cif_ens, q)
    ctd_sgld_q = np.quantile(ctd_sgld, q, axis=0)
    ctd_ens_q = np.quantile(ctd_ens, q, axis=0)
    dev_sgld_q = np.quantile(dev_sgld, q)
    dev_ens_q = np.quantile(dev_ens, q)

    return {
        "setting": s,
        "times": (teacher_time, sgld_time, ens_time),
        "n_samples": (len(sampler.posterior_samples), len(ensemble)),
        "cif_width": (interval_width(cif_sgld_q).mean(), interval_width(cif_ens_q).mean()),
        "ctd1": (ctd_sgld_q[:, 0], ctd_ens_q[:, 0]),
        "ctd2": (ctd_sgld_q[:, 1], ctd_ens_q[:, 1]),
        "dev": (dev_sgld_q, dev_ens_q),
    }


def fmt_row(r):
    s = r["setting"]
    cif_s, cif_e = r["cif_width"]
    c1_s, c1_e = r["ctd1"]
    c2_s, c2_e = r["ctd2"]
    d_s, d_e = r["dev"]
    teacher_t, sgld_t, ens_t = r["times"]
    sgld_n, ens_n = r["n_samples"]

    def ratio(a, b):
        return a / max(b, 1e-12)

    return [
        f"=== {s.name.upper()}  (N={s.n_subjects}, sgld_epochs={s.sgld_epochs}, "
        f"samples={sgld_n}, ensemble={ens_n}; teacher {teacher_t:.1f}s + SGLD {sgld_t:.1f}s + ens {ens_t:.1f}s) ===",
        f"  CIF width   :  SGLD={cif_s:.4f}  ensemble={cif_e:.4f}  ratio={ratio(cif_s, cif_e):.2f}x",
        f"  Ctd cause 1 :  SGLD={c1_s[1]:.3f} [{c1_s[0]:.3f}, {c1_s[2]:.3f}] (w={c1_s[2]-c1_s[0]:.3f})"
        f"  |  ens={c1_e[1]:.3f} [{c1_e[0]:.3f}, {c1_e[2]:.3f}] (w={c1_e[2]-c1_e[0]:.3f})"
        f"  ratio={ratio(c1_s[2]-c1_s[0], c1_e[2]-c1_e[0]):.2f}x",
        f"  Ctd cause 2 :  SGLD={c2_s[1]:.3f} [{c2_s[0]:.3f}, {c2_s[2]:.3f}] (w={c2_s[2]-c2_s[0]:.3f})"
        f"  |  ens={c2_e[1]:.3f} [{c2_e[0]:.3f}, {c2_e[2]:.3f}] (w={c2_e[2]-c2_e[0]:.3f})"
        f"  ratio={ratio(c2_s[2]-c2_s[0], c2_e[2]-c2_e[0]):.2f}x",
        f"  Deviance    :  SGLD={d_s[1]:.3f} [{d_s[0]:.3f}, {d_s[2]:.3f}] (w={d_s[2]-d_s[0]:.3f})"
        f"  |  ens={d_e[1]:.3f} [{d_e[0]:.3f}, {d_e[2]:.3f}] (w={d_e[2]-d_e[0]:.3f})"
        f"  ratio={ratio(d_s[2]-d_s[0], d_e[2]-d_e[0]):.2f}x",
    ]


def main() -> None:
    torch.set_num_threads(1)
    results = []
    for s in SETTINGS:
        print(f"\nRunning {s.name}...", flush=True)
        results.append(run_setting(s))
    print()
    print("=" * 132)
    for r in results:
        for line in fmt_row(r):
            print(line)
        print()
    print("=" * 132)
    print("\nRatio summary (closer to 1.00 = better calibration vs ensemble baseline):")
    print(f"  {'setting':10s} {'CIF':>8s} {'Ctd1':>8s} {'Ctd2':>8s} {'Dev':>8s}")
    for r in results:
        cif_r = r["cif_width"][0] / max(r["cif_width"][1], 1e-12)
        c1_r = (r["ctd1"][0][2]-r["ctd1"][0][0]) / max(r["ctd1"][1][2]-r["ctd1"][1][0], 1e-12)
        c2_r = (r["ctd2"][0][2]-r["ctd2"][0][0]) / max(r["ctd2"][1][2]-r["ctd2"][1][0], 1e-12)
        d_r = (r["dev"][0][2]-r["dev"][0][0]) / max(r["dev"][1][2]-r["dev"][1][0], 1e-12)
        print(f"  {r['setting'].name:10s} {cif_r:8.2f} {c1_r:8.2f} {c2_r:8.2f} {d_r:8.2f}")


if __name__ == "__main__":
    main()
