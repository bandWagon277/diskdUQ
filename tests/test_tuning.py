import numpy as np
import pytest
import torch

from diskd import (
    default_distillation_search_space,
    default_model_search_space,
    default_search_space,
    simulate_competing_risks,
    suggest_params,
    tune_discrete_survival_model,
    tune_diskd_student,
)

optuna = pytest.importorskip("optuna")
torch.set_num_threads(1)


class ConstantCompetingTeacher:
    def __init__(self, num_risks: int, num_durations: int):
        self.num_risks = num_risks
        self.num_durations = num_durations

    def predict_interval_probs(self, data):
        n = len(data)
        event = torch.full((n, self.num_risks, self.num_durations), 0.06)
        no_event = torch.full((n, 1, self.num_durations), 1.0 - 0.06 * self.num_risks)
        return torch.cat([event, no_event], dim=1)


def _data():
    return simulate_competing_risks(n=40, seed=19, censor_max=0.05)


def _features(data):
    return [c for c in data.columns if c.startswith("x")][:5]


def _tiny_space(include_distillation: bool = False):
    space = {
        "backbone": ["mlp"],
        "hidden_dim": [8],
        "hidden_layers": {"type": "fixed", "value": 1},
        "dropout": {"type": "fixed", "value": 0.0},
        "lr": {"type": "fixed", "value": 1e-3},
        "batch_size": {"type": "fixed", "value": 64},
        "epochs": {"type": "fixed", "value": 1},
        "optimizer": ["adamw"],
        "device": {"type": "fixed", "value": "cpu"},
    }
    if include_distillation:
        space["eta"] = {"type": "fixed", "value": 0.5}
        space["temperature"] = [1.0]
    return space


def _tiny_transformer_space(include_distillation: bool = False):
    space = _tiny_space(include_distillation=include_distillation)
    space["backbone"] = ["transformer"]
    space["hidden_dim"] = [8]
    space["nhead"] = [2]
    return space


def test_default_search_space_includes_distillation_terms():
    space = default_search_space(include_distillation=True, teacher_type="competing")
    assert "backbone" in space
    assert "transformer" in space["backbone"]
    assert "nhead" in space
    assert "eta" in space
    assert "temperature" in space


def test_default_search_spaces_separate_model_and_distillation_terms():
    model_space = default_model_search_space()
    distillation_space = default_distillation_search_space(teacher_type="competing")
    assert "eta" not in model_space
    assert "temperature" not in model_space
    assert "backbone" not in distillation_space
    assert "eta" in distillation_space
    assert "temperature" in distillation_space


def test_suggest_params_supports_fixed_and_categorical_specs():
    trial = optuna.trial.FixedTrial({"backbone": "mlp", "optimizer": "adamw"})
    params = suggest_params(
        trial,
        {
            "backbone": ["mlp"],
            "optimizer": {"type": "categorical", "choices": ["adamw"]},
            "epochs": {"type": "fixed", "value": 1},
        },
        fixed_params={"lr": 1e-3},
    )
    assert params == {"backbone": "mlp", "optimizer": "adamw", "epochs": 1, "lr": 1e-3}


def test_tune_discrete_survival_model_smoke():
    torch.manual_seed(19)
    data = _data()
    result = tune_discrete_survival_model(
        data,
        feature_cols=_features(data),
        num_risks=2,
        num_durations=5,
        search_space=_tiny_space(include_distillation=False),
        cv_folds=2,
        n_trials=1,
        validation_size=0.25,
        seed=19,
        num_threads=1,
    )
    assert result.best_model is not None
    assert np.isfinite(result.best_value)
    assert result.best_params["backbone"] == "mlp"


def test_tune_diskd_student_smoke():
    torch.manual_seed(23)
    data = _data()
    result = tune_diskd_student(
        data,
        teacher_model=ConstantCompetingTeacher(num_risks=2, num_durations=5),
        teacher_type="competing",
        feature_cols=_features(data),
        num_risks=2,
        num_durations=5,
        search_space=_tiny_space(include_distillation=True),
        cv_folds=2,
        n_trials=1,
        validation_size=0.25,
        seed=23,
        num_threads=1,
    )
    assert result.best_model is not None
    assert np.isfinite(result.best_value)
    assert result.best_params["eta"] == 0.5


def test_tune_diskd_student_tunes_eta_by_default():
    torch.manual_seed(24)
    data = _data()
    result = tune_diskd_student(
        data,
        teacher_model=ConstantCompetingTeacher(num_risks=2, num_durations=5),
        teacher_type="competing",
        feature_cols=_features(data),
        num_risks=2,
        num_durations=5,
        model_search_space=_tiny_space(include_distillation=False),
        distillation_search_space={
            "eta": [0.25, 0.75],
            "temperature": [1.0],
        },
        cv_folds=2,
        n_trials=1,
        validation_size=0.25,
        seed=24,
        num_threads=1,
    )
    assert result.best_model is not None
    assert result.best_params["eta"] in {0.25, 0.75}
    assert "eta" in result.study.best_trial.params
    assert result.stage1_study is not None
    assert result.stage2_study is result.study
    assert "eta" not in result.stage1_study.best_trial.params
    assert result.stage1_best_params is not None
    assert "eta" not in result.stage1_best_params
    assert result.stage2_best_params is not None
    assert "eta" in result.stage2_best_params


def test_tune_diskd_student_fixed_eta_is_not_suggested():
    torch.manual_seed(25)
    data = _data()
    result = tune_diskd_student(
        data,
        teacher_model=ConstantCompetingTeacher(num_risks=2, num_durations=5),
        teacher_type="competing",
        feature_cols=_features(data),
        num_risks=2,
        num_durations=5,
        model_search_space=_tiny_space(include_distillation=False),
        distillation_search_space={
            "eta": [0.25, 0.75],
            "temperature": [1.0],
        },
        eta=0.5,
        cv_folds=2,
        n_trials=1,
        validation_size=0.25,
        seed=25,
        num_threads=1,
    )
    assert result.best_model is not None
    assert result.best_params["eta"] == 0.5
    assert "eta" not in result.study.best_trial.params


def test_tune_discrete_survival_model_transformer_smoke():
    torch.manual_seed(29)
    data = _data()
    result = tune_discrete_survival_model(
        data,
        feature_cols=_features(data),
        num_risks=2,
        num_durations=5,
        search_space=_tiny_transformer_space(include_distillation=False),
        cv_folds=2,
        n_trials=1,
        validation_size=0.25,
        seed=29,
        num_threads=1,
    )
    assert result.best_model is not None
    assert np.isfinite(result.best_value)
    assert result.best_params["backbone"] == "transformer"
    assert result.best_params["nhead"] == 2


def test_tune_diskd_student_uses_external_valid_for_final_early_stopping():
    torch.manual_seed(31)
    data = _data()
    train = data.iloc[:30].reset_index(drop=True)
    valid = data.iloc[30:].reset_index(drop=True)
    result = tune_diskd_student(
        train,
        valid_data=valid,
        teacher_model=ConstantCompetingTeacher(num_risks=2, num_durations=5),
        teacher_type="competing",
        feature_cols=_features(data),
        num_risks=2,
        num_durations=5,
        model_search_space=_tiny_space(include_distillation=False),
        distillation_search_space={
            "eta": [0.25],
            "temperature": [1.0],
        },
        cv_folds=2,
        early_stopping_patience=1,
        n_trials=1,
        validation_size=0.25,
        seed=31,
        num_threads=1,
    )
    assert result.best_model is not None
    assert result.best_model.history.val_losses is not None
    assert len(result.best_model.history.val_losses) >= 1
