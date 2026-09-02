"""Synthetic tutorial: fixed-horizon binary teacher to competing-risk student.

This script uses only simulated data. It trains a binary classifier that
predicts P(event cause 1 by a chosen horizon), then distills that probability
into the competing-risk student's cause-1 CIF at the same horizon.

Run from the repository root with:

    python examples/synthetic_binary_horizon_tutorial.py
"""

from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diskd import DiSKDStudent, fit_time_grid, simulate_competing_risks


def main() -> None:
    # Keep this tutorial responsive on CPU-only machines and shared clusters.
    torch.set_num_threads(1)

    # Fix the neural-network initialization so the tutorial output is stable.
    torch.manual_seed(13)

    # Generate a toy competing-risk dataset. The binary teacher below is not a
    # survival model; it only predicts whether cause 1 occurred by one horizon.
    data = simulate_competing_risks(n=600, seed=13, censor_max=0.05)
    train, test = train_test_split(data, test_size=0.25, random_state=13)

    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]

    # Build the student's discrete time grid first, then choose a horizon index
    # on that grid. The distillation loss will match the student's CIF at this
    # same index, so the binary teacher target and student target are aligned.
    time_grid = fit_time_grid(train["duration"].values, num_durations=12)
    horizon_index = 7
    horizon_time = float(time_grid.cuts[horizon_index])

    # The binary teacher target is 1 if cause 1 was observed by the horizon and
    # 0 otherwise. This is a compact tutorial target; production analyses may
    # need additional censoring-specific handling for fixed-horizon labels.
    binary_target = ((train["event"] == 1) & (train["duration"] <= horizon_time)).astype(int)
    binary_teacher = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=13),
    )
    binary_teacher.fit(train[all_features], binary_target)

    # DiSKDStudent accepts a callable teacher for `teacher_type="binary_horizon"`.
    # The callable returns one probability per row in the input DataFrame.
    def predict_risk1_by_horizon(frame):
        return binary_teacher.predict_proba(frame[all_features])[:, 1]

    # The competing-risk student receives cause labels and matches its cause-1
    # CIF at `horizon_index` to the binary teacher probability.
    student = DiSKDStudent(
        num_risks=2,
        num_durations=12,
        hidden_dim=32,
        epochs=3,
        batch_size=128,
        device="cpu",
        teacher_model=predict_risk1_by_horizon,
        teacher_type="binary_horizon",
        binary_risk_index=0,
        binary_horizon_index=horizon_index,
        eta=1.0,
        time_grid=time_grid,
    ).fit(train, feature_cols=student_features)

    cif = student.predict_cif(test.head(5))
    risk1_at_horizon = cif[:, 0, horizon_index]

    print(f"Binary teacher horizon index: {horizon_index}")
    print(f"Binary teacher horizon time: {horizon_time:.4f}")
    print(f"Student final loss: {student.history.losses[-1]:.4f}")
    print(f"CIF shape: {cif.shape}")
    print(f"Risk-1 CIF at horizon for first subject: {risk1_at_horizon[0]:.4f}")


if __name__ == "__main__":
    main()
