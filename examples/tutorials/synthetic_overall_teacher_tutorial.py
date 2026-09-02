"""Synthetic DiSKD-O tutorial: overall-risk teacher to competing-risk student.

This script uses only simulated data. It collapses competing-risk labels into
an any-event label for the teacher, then distills the teacher's overall-event
hazard into a competing-risk student.

Run from the repository root with:

    python examples/synthetic_overall_teacher_tutorial.py
"""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split

from diskd import DiSKDStudent, DiscreteSurvivalModel, simulate_competing_risks


def main() -> None:
    # Keep this tutorial responsive on CPU-only machines and shared clusters.
    torch.set_num_threads(1)

    # Fix the neural-network initialization so the tutorial output is stable.
    torch.manual_seed(11)

    # Generate a toy competing-risk dataset with censoring. The competing-risk
    # student will use the original event labels: 0=censored, 1..J=causes.
    data = simulate_competing_risks(n=500, seed=11, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=11)

    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]

    # The overall-risk teacher is trained with a binary any-event target. It
    # predicts a single discrete hazard P(any event in interval k | at risk).
    teacher_train = train.copy()
    teacher_train["event_any"] = (teacher_train["event"] > 0).astype(int)
    teacher = DiscreteSurvivalModel(
        num_risks=1,
        num_durations=12,
        hidden_dim=48,
        epochs=3,
        batch_size=128,
        device="cpu",
    ).fit(teacher_train, feature_cols=all_features, event_col="event_any")

    # The competing-risk student still receives cause labels. DiSKD-O matches
    # the teacher's overall hazard to the student's aggregate event hazard
    # sum_j lambda_j(t_k), while the internal likelihood learns cause allocation.
    student = DiSKDStudent(
        num_risks=2,
        num_durations=12,
        hidden_dim=32,
        epochs=3,
        batch_size=128,
        device="cpu",
        teacher_model=teacher,
        teacher_type="overall",
        eta=1.0,
        time_grid=teacher.time_grid,
    ).fit(train, feature_cols=student_features)

    survival = student.predict_survival(test.head(5))
    cif = student.predict_cif(test.head(5))

    print(f"Overall teacher final loss: {teacher.history.losses[-1]:.4f}")
    print(f"Student final loss: {student.history.losses[-1]:.4f}")
    print(f"Survival shape: {survival.shape}")
    print(f"CIF shape: {cif.shape}")
    print(f"Risk-2 CIF at final interval for first subject: {cif[0, 1, -1]:.4f}")


if __name__ == "__main__":
    main()
