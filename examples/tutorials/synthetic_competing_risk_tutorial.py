"""Synthetic DiSKD-C tutorial: competing-risk teacher to competing-risk student.

This script uses only simulated data. It trains a full competing-risk teacher,
then trains a smaller competing-risk student with the teacher's time-dependent
cause-specific hazards as DiSKD guidance.

Run from the repository root with:

    python examples/synthetic_competing_risk_tutorial.py
"""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split

from diskd import DiSKDStudent, DiscreteSurvivalModel, simulate_competing_risks


def main() -> None:
    # Keep this tutorial responsive on CPU-only machines and shared clusters.
    torch.set_num_threads(1)

    # Fix the neural-network initialization so the tutorial output is stable.
    torch.manual_seed(7)

    # Simulate a two-cause survival dataset. Event labels are:
    #   0 = censored, 1 = first event cause, 2 = second event cause.
    data = simulate_competing_risks(n=500, seed=7, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=7)

    # The teacher sees all simulated covariates. The student intentionally sees
    # fewer covariates, which creates a simple reason for teacher guidance to
    # help in this toy setup.
    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]

    # Fit a competing-risk teacher on all features. The teacher produces
    # interval probabilities with shape [N, J + 1, K], where the final channel
    # is the no-event category.
    teacher = DiscreteSurvivalModel(
        num_risks=2,
        num_durations=12,
        hidden_dim=48,
        epochs=3,
        batch_size=128,
        device="cpu",
    ).fit(train, feature_cols=all_features)

    # Fit a DiSKD-C student. `teacher_type="competing"` uses the teacher's
    # cause-specific interval probabilities and matches teacher || student KL
    # over the at-risk intervals. Sharing the teacher time grid keeps the
    # student and teacher hazards aligned.
    student = DiSKDStudent(
        num_risks=2,
        num_durations=12,
        hidden_dim=32,
        epochs=3,
        batch_size=128,
        device="cpu",
        teacher_model=teacher,
        teacher_type="competing",
        eta=1.0,
        temperature=2.0,
        time_grid=teacher.time_grid,
    ).fit(train, feature_cols=student_features)

    # Public prediction APIs return survival [N, K] and CIF [N, J, K].
    survival = student.predict_survival(test.head(5))
    cif = student.predict_cif(test.head(5))

    print(f"Teacher final loss: {teacher.history.losses[-1]:.4f}")
    print(f"Student final loss: {student.history.losses[-1]:.4f}")
    print(f"Survival shape: {survival.shape}")
    print(f"CIF shape: {cif.shape}")
    print(f"Risk-1 CIF at final interval for first subject: {cif[0, 0, -1]:.4f}")


if __name__ == "__main__":
    main()
