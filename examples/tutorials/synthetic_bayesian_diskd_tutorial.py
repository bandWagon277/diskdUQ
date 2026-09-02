"""Synthetic Bayesian DiSKD tutorial: SGLD multi-chain credible intervals.

Demonstrates the v1 Bayesian extension of DiSKD:

  - Train a deterministic competing-risk teacher with AdamW (same as the
    existing synthetic tutorial).
  - Train a DiSKD student with SGLD using `n_chains` independent chains
    starting from different seeds (mirroring the paper's seed-replicate
    design but interpreted as posterior chains under a flat prior).
  - Collect `samples_per_chain` posterior draws per chain after a burn-in
    fraction; total posterior samples = n_chains * samples_per_chain.
  - Compute credible intervals over CIF, cause-specific C-index, and
    predictive deviance for the test set.
  - Compare against a deep-ensemble baseline (same nominal sample count,
    each member is a fresh AdamW run), which is the "replicate"
    counterpart the paper reports as mean +/- std.

This is a smoke test on synthetic data and is intentionally small enough
to run in ~1-2 minutes on a CPU laptop. Scale `EPOCHS_*` up for any real
experiments.

Run from the repository root with:

    PYTHONPATH=src python examples/synthetic_bayesian_diskd_tutorial.py
"""
from __future__ import annotations

import copy

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
    credible_metric,
    interval_width,
    posterior_predictions,
    predictive_deviance,
    simulate_competing_risks,
    transform_durations,
)


# --- experiment configuration --------------------------------------------------

SEED = 7
N_SUBJECTS = 500
NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 128

TEACHER_EPOCHS = 10
SGLD_EPOCHS_PER_CHAIN = 20   # half burn-in, half sampling
N_CHAINS = 5
SAMPLES_PER_CHAIN = 10        # total 50 posterior samples
SGLD_STEP_SIZE = 1e-5
SGLD_FINAL_STEP_SIZE = 2e-6
SGLD_GAMMA = 1.0

ENSEMBLE_SIZE = 10            # baseline: each member trained from a fresh seed
ENSEMBLE_EPOCHS = 10


def make_teacher(train, all_features) -> DiscreteSurvivalModel:
    teacher = DiscreteSurvivalModel(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=48,
        epochs=TEACHER_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    teacher.fit(train, feature_cols=all_features)
    return teacher


def fit_sgld_multichain(train, teacher, student_features) -> MultiChainSampler:
    base = DiSKDStudent(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=HIDDEN_DIM,
        epochs=SGLD_EPOCHS_PER_CHAIN,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
        time_grid=teacher.time_grid,
        optimizer="sgld",
        sgld_step_size=SGLD_STEP_SIZE,
        sgld_final_step_size=SGLD_FINAL_STEP_SIZE,
        sgld_gamma=SGLD_GAMMA,
        sgld_burnin_epochs=SGLD_EPOCHS_PER_CHAIN // 2,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )
    sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
    sampler.fit(train, feature_cols=student_features)
    return sampler


def fit_ensemble(train, teacher, student_features) -> list[DiSKDStudent]:
    """Deep-ensemble baseline. Mirrors the paper's seed-replicate design."""
    members = []
    for s in range(ENSEMBLE_SIZE):
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        m = DiSKDStudent(
            num_risks=NUM_RISKS,
            num_durations=NUM_DURATIONS,
            hidden_dim=HIDDEN_DIM,
            epochs=ENSEMBLE_EPOCHS,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            teacher_model=teacher,
            teacher_type="competing",
            eta=1.0,
            temperature=2.0,
            time_grid=teacher.time_grid,
        ).fit(train, feature_cols=student_features)
        members.append(m)
    return members


def ensemble_predictions(members: list[DiSKDStudent], test, attr: str) -> np.ndarray:
    """Stack `getattr(m, attr)(test)` across ensemble members."""
    return np.stack([getattr(m, attr)(test) for m in members], axis=0)


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    data = simulate_competing_risks(n=N_SUBJECTS, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)

    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]

    print("=" * 72)
    print("1/4  Training deterministic teacher (AdamW)")
    print("=" * 72)
    teacher = make_teacher(train, all_features)
    print(f"  teacher final NLL: {teacher.history.losses[-1]:.4f}")

    print("=" * 72)
    print(f"2/4  Training SGLD multi-chain student "
          f"({N_CHAINS} chains x {SAMPLES_PER_CHAIN} samples = "
          f"{N_CHAINS * SAMPLES_PER_CHAIN} posterior draws)")
    print("=" * 72)
    sampler = fit_sgld_multichain(train, teacher, student_features)
    print(f"  collected {len(sampler.posterior_samples)} posterior samples")
    print(f"  chain 0 final NLL: {sampler.chains[0].history.losses[-1]:.4f}")

    print("=" * 72)
    print(f"3/4  Training deep-ensemble baseline ({ENSEMBLE_SIZE} members)")
    print("=" * 72)
    ensemble = fit_ensemble(train, teacher, student_features)
    print(f"  member 0 final NLL: {ensemble[0].history.losses[-1]:.4f}")

    print("=" * 72)
    print("4/4  Credible intervals on the test set")
    print("=" * 72)

    # Posterior CIF: [S, N, J, K]
    sgld_cif = posterior_predictions(
        sampler.lead_chain, sampler.posterior_samples, test, predictor="cif"
    )
    ens_cif = ensemble_predictions(ensemble, test, attr="predict_cif")

    print(f"\n  CIF prediction tensor shape: SGLD={sgld_cif.shape}  ensemble={ens_cif.shape}")

    sgld_cif_q = credible_intervals(sgld_cif)        # [3, N, J, K]
    ens_cif_q = credible_intervals(ens_cif)          # [3, N, J, K]
    sgld_width = interval_width(sgld_cif_q).mean()
    ens_width = interval_width(ens_cif_q).mean()
    print(
        f"  Mean 95%-CI width over (subject, cause, time):  "
        f"SGLD={sgld_width:.4f}  ensemble={ens_width:.4f}  "
        f"ratio={sgld_width / max(ens_width, 1e-9):.2f}"
    )

    # C-index credible interval per cause (Figure 4 extension)
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()

    def ctd_fn(model: DiSKDStudent) -> np.ndarray:
        cif = model.predict_cif(test)
        return competing_risk_c_index(cif, test_durations, test_events)

    sgld_ctd_q, sgld_ctd_raw = credible_metric(sampler.lead_chain, sampler.posterior_samples, ctd_fn)
    ens_ctd_raw = np.stack([ctd_fn(m) for m in ensemble], axis=0)
    ens_ctd_q = np.quantile(ens_ctd_raw, [0.025, 0.5, 0.975], axis=0)
    print("\n  Cause-specific Ctd (median [2.5%, 97.5%]):")
    for j in range(NUM_RISKS):
        print(
            f"    cause {j + 1}:  SGLD={sgld_ctd_q[1, j]:.3f} "
            f"[{sgld_ctd_q[0, j]:.3f}, {sgld_ctd_q[2, j]:.3f}]  "
            f"ensemble={ens_ctd_q[1, j]:.3f} "
            f"[{ens_ctd_q[0, j]:.3f}, {ens_ctd_q[2, j]:.3f}]"
        )

    # Predictive deviance credible interval (Table 3 extension)
    student_time_grid = sampler.lead_chain.time_grid
    test_idx = transform_durations(test_durations, student_time_grid)

    def deviance_fn(model: DiSKDStudent) -> float:
        probs = model.predict_interval_probs(test).numpy()
        return predictive_deviance(probs, test_idx, test_events)

    sgld_dev_q, sgld_dev_raw = credible_metric(
        sampler.lead_chain, sampler.posterior_samples, deviance_fn
    )
    ens_dev_raw = np.array([deviance_fn(m) for m in ensemble])
    ens_dev_q = np.quantile(ens_dev_raw, [0.025, 0.5, 0.975])
    print(
        f"\n  Predictive deviance (median [2.5%, 97.5%]):\n"
        f"    SGLD     = {sgld_dev_q[1]:.4f} "
        f"[{sgld_dev_q[0]:.4f}, {sgld_dev_q[2]:.4f}]  "
        f"(width={sgld_dev_q[2] - sgld_dev_q[0]:.4f})\n"
        f"    ensemble = {ens_dev_q[1]:.4f} "
        f"[{ens_dev_q[0]:.4f}, {ens_dev_q[2]:.4f}]  "
        f"(width={ens_dev_q[2] - ens_dev_q[0]:.4f})"
    )

    print("\n" + "=" * 72)
    print("Done. Compare SGLD vs ensemble interval widths above.")
    print("If they're close in magnitude, the SGLD posterior is well-calibrated.")
    print("=" * 72)


if __name__ == "__main__":
    main()
