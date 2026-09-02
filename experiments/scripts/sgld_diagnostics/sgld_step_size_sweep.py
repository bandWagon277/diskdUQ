"""SGLD step-size schedule sweep for Bayesian DiSKD calibration.

Sweeps (sgld_step_size, sgld_final_step_size, sgld_gamma) for the same
teacher/student/data configuration. For each combination, fits a 5-chain
SGLD multi-chain student (10 samples per chain = 50 posterior draws) and
records 95%-credible-interval widths and medians.

Compares against a single fixed deep-ensemble baseline (10 members, AdamW)
which plays the role of the "seed-replicate" reference from the paper.

Goal: identify a step-size schedule whose SGLD posterior CI width is
within 30% of the ensemble baseline width, while preserving the median
estimates. That schedule then becomes the v1 default for the Bayesian
DiSKD extension.

Run from the repository root with:

    PYTHONPATH=src python examples/sgld_step_size_sweep.py
"""
from __future__ import annotations

import itertools
import time

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


# --- experiment configuration --------------------------------------------------

SEED = 7
N_SUBJECTS = 500
NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 128

TEACHER_EPOCHS = 10
SGLD_EPOCHS_PER_CHAIN = 16     # 8 burn-in + 8 sample-collection
N_CHAINS = 5
SAMPLES_PER_CHAIN = 10
ENSEMBLE_SIZE = 10
ENSEMBLE_EPOCHS = 10

# Sweep grid: 4 step sizes x (1 constant + 3 decay gammas) = 16 combos
STEP_SIZES = [5e-7, 1e-6, 5e-6, 1e-5]
DECAY_GAMMAS = [0.55, 0.75, 1.0]


def build_combos():
    combos = []
    for step in STEP_SIZES:
        combos.append({"step": step, "final": None, "gamma": None, "label": "const"})
        for g in DECAY_GAMMAS:
            combos.append(
                {"step": step, "final": step / 5.0, "gamma": g, "label": f"decay g={g}"}
            )
    return combos


def make_teacher(train, all_features):
    return DiscreteSurvivalModel(
        num_risks=NUM_RISKS,
        num_durations=NUM_DURATIONS,
        hidden_dim=48,
        epochs=TEACHER_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    ).fit(train, feature_cols=all_features)


def make_base_student(teacher, combo):
    return DiSKDStudent(
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
        sgld_step_size=combo["step"],
        sgld_final_step_size=combo["final"],
        sgld_gamma=combo["gamma"] if combo["gamma"] is not None else 0.55,
        sgld_burnin_epochs=SGLD_EPOCHS_PER_CHAIN // 2,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )


def run_ensemble_baseline(train, teacher, student_features):
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


def evaluate(sampler_or_models, samples_or_none, test, lead_chain, n_durations):
    """Return (cif_q, ctd_q, dev_q) for posterior (sampler) or ensemble (list)."""
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()
    test_idx = transform_durations(test_durations, lead_chain.time_grid)

    if samples_or_none is not None:
        # Posterior path
        cif = posterior_predictions(lead_chain, samples_or_none, test, "cif")
        ctd_vals = []
        dev_vals = []
        from diskd.uncertainty import _iter_samples
        for _, m in _iter_samples(lead_chain, samples_or_none):
            ctd_vals.append(competing_risk_c_index(m.predict_cif(test), test_durations, test_events))
            dev_vals.append(predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_events))
        ctd_arr = np.stack(ctd_vals, axis=0)
        dev_arr = np.array(dev_vals)
    else:
        # Ensemble path
        cif = np.stack([m.predict_cif(test) for m in sampler_or_models], axis=0)
        ctd_arr = np.stack(
            [competing_risk_c_index(m.predict_cif(test), test_durations, test_events)
             for m in sampler_or_models], axis=0,
        )
        dev_arr = np.array(
            [predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_events)
             for m in sampler_or_models]
        )

    cif_q = credible_intervals(cif)
    ctd_q = np.quantile(ctd_arr, [0.025, 0.5, 0.975], axis=0)
    dev_q = np.quantile(dev_arr, [0.025, 0.5, 0.975])
    return cif_q, ctd_q, dev_q


def fmt_row(label, cif_q, ctd_q, dev_q, ref_cif_width=None, elapsed=None):
    cif_w = interval_width(cif_q).mean()
    ratio_str = f"{cif_w / ref_cif_width:.2f}x" if ref_cif_width else "  -  "
    ctd1_med = ctd_q[1, 0]; ctd1_w = ctd_q[2, 0] - ctd_q[0, 0]
    ctd2_med = ctd_q[1, 1]; ctd2_w = ctd_q[2, 1] - ctd_q[0, 1]
    dev_med = dev_q[1]; dev_w = dev_q[2] - dev_q[0]
    et = f"{elapsed:5.1f}s" if elapsed is not None else "      "
    return (
        f"{label:32s} | {cif_w:.3f} ({ratio_str}) | "
        f"Ctd1 {ctd1_med:.3f} (w={ctd1_w:.3f}) | "
        f"Ctd2 {ctd2_med:.3f} (w={ctd2_w:.3f}) | "
        f"Dev {dev_med:6.2f} (w={dev_w:.2f}) | {et}"
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    data = simulate_competing_risks(n=N_SUBJECTS, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]

    print(f"Setup: n={N_SUBJECTS}, num_risks={NUM_RISKS}, K={NUM_DURATIONS}, "
          f"chains={N_CHAINS}, samples/chain={SAMPLES_PER_CHAIN}, "
          f"epochs/chain={SGLD_EPOCHS_PER_CHAIN}")

    print("\n--- training shared teacher ---")
    t0 = time.time()
    teacher = make_teacher(train, all_features)
    print(f"  teacher NLL = {teacher.history.losses[-1]:.4f}  ({time.time() - t0:.1f}s)")

    print(f"\n--- ensemble baseline ({ENSEMBLE_SIZE} members) ---")
    t0 = time.time()
    ensemble = run_ensemble_baseline(train, teacher, student_features)
    ens_cif_q, ens_ctd_q, ens_dev_q = evaluate(ensemble, None, test, ensemble[0], NUM_DURATIONS)
    ref_width = interval_width(ens_cif_q).mean()
    print(fmt_row("ensemble (ref)", ens_cif_q, ens_ctd_q, ens_dev_q, ref_width, time.time() - t0))

    combos = build_combos()
    print(f"\n--- SGLD sweep: {len(combos)} combos ---")
    print("=" * 152)
    header = (f"{'config':32s} | width (ratio) | "
              f"Ctd1 (median, width)   | Ctd2 (median, width)   | "
              f"Dev (median, width)      | time")
    print(header)
    print("-" * 152)

    results = []
    for i, combo in enumerate(combos, 1):
        label = f"step={combo['step']:.0e} {combo['label']}"
        t0 = time.time()
        base = make_base_student(teacher, combo)
        sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
        sampler.fit(train, feature_cols=student_features)
        cif_q, ctd_q, dev_q = evaluate(sampler, sampler.posterior_samples, test, sampler.lead_chain, NUM_DURATIONS)
        elapsed = time.time() - t0
        line = fmt_row(label, cif_q, ctd_q, dev_q, ref_width, elapsed)
        print(f"[{i:2d}/{len(combos)}] {line}", flush=True)
        results.append((label, interval_width(cif_q).mean(), ctd_q, dev_q))

    print("=" * 152)
    print("\nDone. Rows closest to ratio 1.00 (i.e., SGLD CI ≈ ensemble CI) are the")
    print("best-calibrated step-size schedules. Pick one and update SGLD defaults.")


if __name__ == "__main__":
    main()
