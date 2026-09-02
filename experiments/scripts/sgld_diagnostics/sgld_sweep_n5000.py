"""SGLD step-size sweep at N=5000 to match SRTR paper Table 1 / Figure 4 scale.

The paper's post-COVID SRTR student cohort has n = 5,041 (Appendix D.4).
This script reruns the step-size sweep at that scale on synthetic data,
so the resulting default schedule is directly applicable when the same
pipeline is later run on the real SRTR data.

Setup compared to `sgld_step_size_sweep.py`:
  - N=5,000 (up from 500)
  - Teacher and chain epochs increased so the deeper model can fit
  - Step-size grid shifted to smaller values, since theoretical guidance
    for SGLD recommends scaling step ~ O(1/N).

Run from the repository root with:

    PYTHONPATH=src python examples/sgld_sweep_n5000.py
"""
from __future__ import annotations

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
from diskd.uncertainty import _iter_samples


SEED = 7
N_SUBJECTS = 5000
NUM_RISKS = 2
NUM_DURATIONS = 12
HIDDEN_DIM = 32
BATCH_SIZE = 256

TEACHER_EPOCHS = 15
SGLD_EPOCHS_PER_CHAIN = 30
N_CHAINS = 5
SAMPLES_PER_CHAIN = 10
ENSEMBLE_SIZE = 20
ENSEMBLE_EPOCHS = 15

# Step-size grid shifted down because N grew 10x.
STEP_SIZES = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
DECAY_GAMMAS = [0.55, 1.0]


def build_combos():
    combos = []
    for step in STEP_SIZES:
        combos.append({"step": step, "final": None, "gamma": None, "label": "const"})
        for g in DECAY_GAMMAS:
            combos.append({"step": step, "final": step / 5.0, "gamma": g, "label": f"decay g={g}"})
    return combos


def make_teacher(train, all_features):
    return DiscreteSurvivalModel(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=48,
        epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
    ).fit(train, feature_cols=all_features)


def make_base_student(teacher, combo):
    return DiSKDStudent(
        num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
        epochs=SGLD_EPOCHS_PER_CHAIN, batch_size=BATCH_SIZE, device=DEVICE,
        teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
        time_grid=teacher.time_grid, optimizer="sgld",
        sgld_step_size=combo["step"],
        sgld_final_step_size=combo["final"],
        sgld_gamma=combo["gamma"] if combo["gamma"] is not None else 1.0,
        sgld_burnin_epochs=SGLD_EPOCHS_PER_CHAIN // 2,
        sgld_samples_per_chain=SAMPLES_PER_CHAIN,
    )


def run_ensemble(train, teacher, student_features):
    members = []
    for j in range(ENSEMBLE_SIZE):
        torch.manual_seed(1000 + j)
        np.random.seed(1000 + j)
        m = DiSKDStudent(
            num_risks=NUM_RISKS, num_durations=NUM_DURATIONS, hidden_dim=HIDDEN_DIM,
            epochs=ENSEMBLE_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            teacher_model=teacher, teacher_type="competing", eta=1.0, temperature=2.0,
            time_grid=teacher.time_grid,
        ).fit(train, feature_cols=student_features)
        members.append(m)
    return members


def evaluate(lead_model, posterior_samples, test, test_durations, test_events, test_idx):
    cif = posterior_predictions(lead_model, posterior_samples, test, "cif")
    ctd = []
    dev = []
    for _, m in _iter_samples(lead_model, posterior_samples):
        ctd.append(competing_risk_c_index(m.predict_cif(test), test_durations, test_events))
        dev.append(predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_events))
    return {
        "cif_q": credible_intervals(cif),
        "ctd_q": np.quantile(np.stack(ctd, axis=0), [0.025, 0.5, 0.975], axis=0),
        "dev_q": np.quantile(np.array(dev), [0.025, 0.5, 0.975]),
    }


def evaluate_ensemble(members, test, test_durations, test_events, test_idx):
    cif = np.stack([m.predict_cif(test) for m in members], axis=0)
    ctd = np.stack(
        [competing_risk_c_index(m.predict_cif(test), test_durations, test_events) for m in members], axis=0,
    )
    dev = np.array(
        [predictive_deviance(m.predict_interval_probs(test).numpy(), test_idx, test_events) for m in members]
    )
    return {
        "cif_q": credible_intervals(cif),
        "ctd_q": np.quantile(ctd, [0.025, 0.5, 0.975], axis=0),
        "dev_q": np.quantile(dev, [0.025, 0.5, 0.975]),
    }


def fmt(label, r, ref_cif_width=None, elapsed=None):
    cif_w = interval_width(r["cif_q"]).mean()
    ratio = f"{cif_w / ref_cif_width:.2f}x" if ref_cif_width else "  -  "
    c1m, c1w = r["ctd_q"][1, 0], r["ctd_q"][2, 0] - r["ctd_q"][0, 0]
    c2m, c2w = r["ctd_q"][1, 1], r["ctd_q"][2, 1] - r["ctd_q"][0, 1]
    dm, dw = r["dev_q"][1], r["dev_q"][2] - r["dev_q"][0]
    et = f"{elapsed:5.1f}s" if elapsed is not None else "     "
    return (f"{label:32s} | {cif_w:.4f} ({ratio}) | Ctd1 {c1m:.3f} (w={c1w:.3f}) | "
            f"Ctd2 {c2m:.3f} (w={c2w:.3f}) | Dev {dm:6.3f} (w={dw:.3f}) | {et}")


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    data = simulate_competing_risks(n=N_SUBJECTS, seed=SEED, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=SEED)
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]
    test_durations = test["duration"].to_numpy()
    test_events = test["event"].to_numpy()

    print(f"Setup: N={N_SUBJECTS}  (train={len(train)}, test={len(test)})")
    print(f"  teacher epochs={TEACHER_EPOCHS}, SGLD epochs/chain={SGLD_EPOCHS_PER_CHAIN}, "
          f"chains={N_CHAINS}, samples/chain={SAMPLES_PER_CHAIN}, ensemble={ENSEMBLE_SIZE}")

    print("\n--- teacher ---")
    t0 = time.time()
    teacher = make_teacher(train, all_features)
    test_idx = transform_durations(test_durations, teacher.time_grid)
    print(f"  teacher NLL = {teacher.history.losses[-1]:.4f}  ({time.time() - t0:.1f}s)")

    print(f"\n--- ensemble baseline ({ENSEMBLE_SIZE} members) ---")
    t0 = time.time()
    ensemble = run_ensemble(train, teacher, student_features)
    ens = evaluate_ensemble(ensemble, test, test_durations, test_events, test_idx)
    ref_width = interval_width(ens["cif_q"]).mean()
    print(fmt("ensemble (ref)", ens, ref_width, time.time() - t0))

    combos = build_combos()
    print(f"\n--- SGLD sweep: {len(combos)} combos ---")
    print("=" * 152)
    header = (f"{'config':32s} | width (ratio)   | Ctd1 (median, width)   | Ctd2 (median, width)   | "
              f"Dev (median, width)     | time")
    print(header)
    print("-" * 152)

    for i, combo in enumerate(combos, 1):
        label = f"step={combo['step']:.0e} {combo['label']}"
        t0 = time.time()
        base = make_base_student(teacher, combo)
        sampler = MultiChainSampler(base, n_chains=N_CHAINS, seeds=list(range(N_CHAINS)))
        sampler.fit(train, feature_cols=student_features)
        result = evaluate(sampler.lead_chain, sampler.posterior_samples, test, test_durations, test_events, test_idx)
        print(f"[{i:2d}/{len(combos)}] {fmt(label, result, ref_width, time.time() - t0)}", flush=True)

    print("=" * 152)


if __name__ == "__main__":
    main()
