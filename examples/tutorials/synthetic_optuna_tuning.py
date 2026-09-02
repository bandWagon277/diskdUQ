"""Synthetic Optuna tuning tutorial for DiSKD.

This script uses only simulated data. It first trains a small competing-risk
teacher, then uses Optuna to select student network and training hyperparameters
from a compact search space.

Run from the repository root after installing the tuning extra:

    pip install -e ".[tuning]"
    python examples/synthetic_optuna_tuning.py
"""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split

from diskd import (
    DiscreteSurvivalModel,
    simulate_competing_risks,
    tune_diskd_student,
)


def main() -> None:
    # Keep the tutorial responsive on CPU-only machines and shared clusters.
    torch.set_num_threads(1)
    torch.manual_seed(17)

    # Use simulated competing-risk data. Event labels are 0=censored and
    # 1..J=event causes.
    data = simulate_competing_risks(n=420, seed=17, censor_max=0.05)
    train_valid, test = train_test_split(data, test_size=0.25, random_state=17)
    train, valid = train_test_split(train_valid, test_size=0.25, random_state=18)

    all_features = [c for c in data.columns if c.startswith("x")]
    student_features = all_features[:8]

    # Fit a fixed teacher first. Tuning below searches the student
    # hyperparameters, not the teacher hyperparameters. Optuna selects student
    # hyperparameters with CV inside `train`; `valid` is reserved for final
    # refit early stopping.
    teacher = DiscreteSurvivalModel(
        num_risks=2,
        num_durations=10,
        hidden_dim=32,
        epochs=2,
        batch_size=128,
        device="cpu",
    ).fit(train, feature_cols=all_features)

    # The tuning helper runs two stages on the same training set. Stage 1
    # searches only neural-network/training parameters with CV. Stage 2 fixes
    # those parameters and searches only distillation parameters with CV.
    model_search_space = {
        "backbone": ["mlp", "time_mlp", "transformer"],
        # Keep hidden dimensions divisible by all nhead choices below.
        "hidden_dim": [16, 32],
        "hidden_layers": {"type": "int", "low": 1, "high": 2},
        "dropout": {"type": "float", "low": 0.0, "high": 0.2},
        "nhead": [2, 4],
        "lr": {"type": "float", "low": 1e-3, "high": 5e-3, "log": True},
        "batch_size": [64, 128],
        "epochs": {"type": "fixed", "value": 2},
        "optimizer": ["adamw", "adam"],
    }
    # Distillation hyperparameters are searched separately from the neural
    # network hyperparameters. Omit `eta` only when you want to fix it via
    # `eta=...` or `fixed_distillation_params={"eta": ...}`.
    distillation_search_space = {
        "eta": {"type": "float", "low": 0.0, "high": 2.0},
        "temperature": [1.0, 2.0],
    }

    result = tune_diskd_student(
        train,
        valid_data=valid,
        teacher_model=teacher,
        teacher_type="competing",
        num_risks=2,
        num_durations=10,
        feature_cols=student_features,
        model_search_space=model_search_space,
        distillation_search_space=distillation_search_space,
        cv_folds=3,
        model_n_trials=3,
        distillation_n_trials=3,
        early_stopping_patience=2,
        seed=17,
        num_threads=1,
    )

    cif = result.best_model.predict_cif(test.head(5))

    print(f"Best validation NLL: {result.best_value:.4f}")
    print(f"Best params: {result.best_params}")
    print(f"CIF shape: {cif.shape}")


if __name__ == "__main__":
    main()
