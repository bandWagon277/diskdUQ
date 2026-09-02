"""Optional Optuna tuning helpers for DiSKD models.

Optuna is an optional dependency. Install it with `pip install diskd[tuning]`
or `pip install -e ".[tuning]"` when working from a clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

from .losses import CompetingRiskNLLLoss, SingleRiskNLLLoss
from .models import DiSKDStudent, DiscreteSurvivalModel
from .preprocessing import transform_durations


SearchSpace = dict[str, Any]

MODEL_PARAM_KEYS = {
    "backbone",
    "hidden_dim",
    "hidden_layers",
    "dropout",
    "nhead",
    "lr",
    "batch_size",
    "epochs",
    "optimizer",
    "device",
    "time_grid",
}
STUDENT_PARAM_KEYS = {"eta", "temperature", "binary_risk_index", "binary_horizon_index"}


@dataclass
class TuningResult:
    """Result returned by Optuna tuning helpers."""

    best_model: DiscreteSurvivalModel | DiSKDStudent | None
    best_params: dict[str, Any]
    best_value: float
    study: Any
    stage1_study: Any | None = None
    stage2_study: Any | None = None
    stage1_best_params: dict[str, Any] | None = None
    stage2_best_params: dict[str, Any] | None = None


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for tuning. Install with `pip install diskd[tuning]` "
            "or `pip install -e \".[tuning]\"` from a local clone."
        ) from exc
    return optuna


def default_model_search_space() -> SearchSpace:
    """Return a moderate default search space for neural-network parameters.

    Users can pass a smaller or larger `search_space` to the tuning functions.
    Values can be specified as lists for categorical search, or dictionaries:

    - `{"type": "categorical", "choices": [...]}`
    - `{"type": "int", "low": 1, "high": 4}`
    - `{"type": "float", "low": 1e-4, "high": 1e-2, "log": True}`
    - `{"type": "fixed", "value": ...}`
    """
    return {
        "backbone": ["mlp", "time_mlp", "transformer"],
        "hidden_dim": [32, 64, 128, 256],
        "hidden_layers": {"type": "int", "low": 1, "high": 4},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "nhead": [2, 4, 8],
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": [32, 64, 128],
        "epochs": [10, 20, 50],
        "optimizer": ["adamw", "adam"],
    }


def default_distillation_search_space(teacher_type: str = "competing") -> SearchSpace:
    """Return the default search space for DiSKD distillation parameters.

    `eta` is tuned by default. Set `eta=...`, pass
    `fixed_distillation_params={"eta": ...}`, or include `eta` in `fixed_params`
    to use a fixed value instead.
    """
    space: SearchSpace = {
        "eta": {"type": "float", "low": 0.0, "high": 5.0},
    }
    if teacher_type == "competing":
        space["temperature"] = [1.0, 2.0, 4.0]
    return space


def default_search_space(include_distillation: bool = False, teacher_type: str = "competing") -> SearchSpace:
    """Return a combined search space.

    This helper is kept for backward compatibility. New tuning code should
    prefer `default_model_search_space()` and
    `default_distillation_search_space(...)` so DiSKD parameters are selected
    separately from neural-network parameters.
    """
    space = default_model_search_space()
    if include_distillation:
        space.update(default_distillation_search_space(teacher_type=teacher_type))
    return space


def suggest_params(trial: Any, search_space: SearchSpace | None = None, fixed_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Suggest trial parameters from a search space and apply fixed overrides."""
    fixed_params = fixed_params or {}
    params = {}
    for name, spec in (search_space or {}).items():
        if name in fixed_params:
            continue
        value = _suggest_one(trial, name, spec)
        if value is not None:
            params[name] = value
    params.update(fixed_params)
    return params


def validation_nll(
    model: DiscreteSurvivalModel,
    data: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
) -> float:
    """Evaluate internal validation NLL for a fitted model."""
    if model.time_grid is None:
        raise RuntimeError("Model must have a fitted time grid before validation.")
    logits = model.predict_logits(data)
    idx = torch.tensor(transform_durations(data[duration_col].values, model.time_grid), dtype=torch.long)
    events = torch.tensor(data[event_col].to_numpy(dtype=np.int64), dtype=torch.long)
    loss_fn = SingleRiskNLLLoss() if model.num_risks == 1 else CompetingRiskNLLLoss()
    return float(loss_fn(logits, idx, events).item())


def tune_discrete_survival_model(
    data: pd.DataFrame,
    *,
    valid_data: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    num_risks: int = 1,
    num_durations: int = 20,
    search_space: SearchSpace | None = None,
    fixed_params: dict[str, Any] | None = None,
    model_search_space: SearchSpace | None = None,
    fixed_model_params: dict[str, Any] | None = None,
    cv_folds: int = 5,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    n_trials: int = 20,
    validation_size: float = 0.25,
    seed: int = 0,
    refit: bool = True,
    num_threads: int | None = 1,
    study_name: str = "diskd_model_tuning",
    verbose: bool = False,
) -> TuningResult:
    """Tune a `DiscreteSurvivalModel` with Optuna and training-set CV.

    If `valid_data` is supplied, it is reserved for early stopping during the
    final refit and is not used as the Optuna objective.
    """
    optuna = _require_optuna()
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_df, early_stop_df = _split_tuning_data(data, valid_data, validation_size, seed)
    _validate_cv_config(train_df, cv_folds)
    space = _resolve_model_space(search_space, model_search_space)
    fixed = _resolve_fixed_model_params(fixed_params, fixed_model_params)

    def objective(trial):
        params = suggest_params(trial, space, fixed)
        torch.manual_seed(seed + trial.number)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        return _cv_objective(
            params,
            train_df,
            feature_cols,
            duration_col,
            event_col,
            cv_folds,
            seed + trial.number,
            lambda p: _make_base_model(p, num_risks, num_durations),
            trial,
        )

    study = _create_study(optuna, study_name, seed)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best_params = _best_params_from_study(study, space, fixed)
    best_model = None
    if refit:
        torch.manual_seed(seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        best_model = _make_base_model(best_params, num_risks, num_durations)
        best_model.fit(
            train_df,
            feature_cols=feature_cols,
            duration_col=duration_col,
            event_col=event_col,
            valid_data=early_stop_df,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
    return TuningResult(best_model=best_model, best_params=best_params, best_value=float(study.best_value), study=study)


def tune_diskd_student(
    data: pd.DataFrame,
    *,
    teacher_model: Any,
    teacher_type: str = "competing",
    valid_data: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    num_risks: int = 2,
    num_durations: int = 20,
    search_space: SearchSpace | None = None,
    fixed_params: dict[str, Any] | None = None,
    model_search_space: SearchSpace | None = None,
    distillation_search_space: SearchSpace | None = None,
    fixed_model_params: dict[str, Any] | None = None,
    fixed_distillation_params: dict[str, Any] | None = None,
    eta: float | None = None,
    temperature: float | None = None,
    cv_folds: int = 5,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    n_trials: int = 20,
    model_n_trials: int | None = None,
    distillation_n_trials: int | None = None,
    validation_size: float = 0.25,
    seed: int = 0,
    refit: bool = True,
    num_threads: int | None = 1,
    study_name: str = "diskd_student_tuning",
    verbose: bool = False,
) -> TuningResult:
    """Tune a `DiSKDStudent` with Optuna and training-set CV.

    Neural-network parameters and DiSKD distillation parameters are tuned in
    two stages on the same training set. Stage 1 runs CV over only model
    parameters. Stage 2 fixes the selected model parameters and runs CV over
    only distillation parameters. By default, `eta` is tuned in Stage 2. Pass
    `eta=...` or `fixed_distillation_params={"eta": ...}` to keep it fixed.

    If `valid_data` is supplied, it is reserved for early stopping during the
    final refit and is not used as the Optuna objective.
    """
    optuna = _require_optuna()
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_df, early_stop_df = _split_tuning_data(data, valid_data, validation_size, seed)
    _validate_cv_config(train_df, cv_folds)
    model_space = _resolve_model_space(search_space, model_search_space)
    distillation_space = _resolve_distillation_space(search_space, distillation_search_space, teacher_type)
    fixed_model = _resolve_fixed_model_params(fixed_params, fixed_model_params)
    fixed_distillation = _resolve_fixed_distillation_params(
        fixed_params,
        fixed_distillation_params,
        eta=eta,
        temperature=temperature,
    )

    model_trials = n_trials if model_n_trials is None else model_n_trials
    distillation_trials = n_trials if distillation_n_trials is None else distillation_n_trials

    def model_objective(trial):
        model_params = _with_teacher_time_grid(suggest_params(trial, model_space, fixed_model), teacher_model)
        torch.manual_seed(seed + trial.number)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        return _cv_objective(
            model_params,
            train_df,
            feature_cols,
            duration_col,
            event_col,
            cv_folds,
            seed + trial.number,
            lambda p: _make_base_model(p, num_risks, num_durations),
            trial,
        )

    model_study = _create_study(optuna, f"{study_name}_model", seed)
    model_study.optimize(model_objective, n_trials=model_trials, n_jobs=1, show_progress_bar=False)
    best_model_params = _best_params_from_study(model_study, model_space, fixed_model)

    def distillation_objective(trial):
        distillation_params = suggest_params(trial, distillation_space, fixed_distillation)
        params = _with_teacher_time_grid({**best_model_params, **distillation_params}, teacher_model)
        torch.manual_seed(seed + 10000 + trial.number)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        return _cv_objective(
            params,
            train_df,
            feature_cols,
            duration_col,
            event_col,
            cv_folds,
            seed + trial.number,
            lambda p: _make_student_model(p, num_risks, num_durations, teacher_model, teacher_type),
            trial,
        )

    distillation_study = _create_study(optuna, f"{study_name}_distillation", seed + 1)
    distillation_study.optimize(
        distillation_objective,
        n_trials=distillation_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_distillation_params = _best_params_from_study(
        distillation_study,
        distillation_space,
        fixed_distillation,
    )
    best_params = {**best_model_params, **best_distillation_params}
    best_model = None
    if refit:
        torch.manual_seed(seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        best_model = _make_student_model(best_params, num_risks, num_durations, teacher_model, teacher_type)
        best_model.fit(
            train_df,
            feature_cols=feature_cols,
            duration_col=duration_col,
            event_col=event_col,
            valid_data=early_stop_df,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
    return TuningResult(
        best_model=best_model,
        best_params=best_params,
        best_value=float(distillation_study.best_value),
        study=distillation_study,
        stage1_study=model_study,
        stage2_study=distillation_study,
        stage1_best_params=best_model_params,
        stage2_best_params=best_distillation_params,
    )


def _suggest_one(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, (list, tuple)):
        return trial.suggest_categorical(name, list(spec))
    if not isinstance(spec, dict):
        return spec
    kind = spec.get("type")
    if kind == "fixed":
        return spec["value"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "float":
        kwargs = {"log": bool(spec.get("log", False))}
        if "step" in spec:
            kwargs["step"] = spec["step"]
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)
    raise ValueError(f"Unsupported search-space spec for {name!r}: {spec!r}")


def _fixed_values_from_space(search_space: SearchSpace) -> dict[str, Any]:
    out = {}
    for name, spec in search_space.items():
        if isinstance(spec, dict) and spec.get("type") == "fixed":
            out[name] = spec["value"]
    return out


def _best_params_from_study(study: Any, search_space: SearchSpace, fixed_params: dict[str, Any] | None) -> dict[str, Any]:
    best_params = _fixed_values_from_space(search_space)
    best_params.update(study.best_trial.params)
    best_params.update(fixed_params or {})
    return best_params


def _best_params_from_spaces(
    study: Any,
    model_search_space: SearchSpace,
    fixed_model_params: dict[str, Any],
    distillation_search_space: SearchSpace,
    fixed_distillation_params: dict[str, Any],
) -> dict[str, Any]:
    best_params = _fixed_values_from_space(model_search_space)
    best_params.update(_fixed_values_from_space(distillation_search_space))
    best_params.update(study.best_trial.params)
    best_params.update(fixed_model_params)
    best_params.update(fixed_distillation_params)
    return best_params


def _resolve_model_space(
    legacy_search_space: SearchSpace | None,
    model_search_space: SearchSpace | None,
) -> SearchSpace:
    if model_search_space is not None:
        return dict(model_search_space)
    if legacy_search_space is not None:
        return {k: v for k, v in legacy_search_space.items() if k in MODEL_PARAM_KEYS}
    return default_model_search_space()


def _resolve_distillation_space(
    legacy_search_space: SearchSpace | None,
    distillation_search_space: SearchSpace | None,
    teacher_type: str,
) -> SearchSpace:
    if distillation_search_space is not None:
        return dict(distillation_search_space)
    if legacy_search_space is not None:
        legacy = {k: v for k, v in legacy_search_space.items() if k in STUDENT_PARAM_KEYS}
        if legacy:
            return legacy
    return default_distillation_search_space(teacher_type=teacher_type)


def _resolve_fixed_model_params(
    legacy_fixed_params: dict[str, Any] | None,
    fixed_model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    fixed = {k: v for k, v in (legacy_fixed_params or {}).items() if k in MODEL_PARAM_KEYS}
    fixed.update(fixed_model_params or {})
    return fixed


def _resolve_fixed_distillation_params(
    legacy_fixed_params: dict[str, Any] | None,
    fixed_distillation_params: dict[str, Any] | None,
    *,
    eta: float | None,
    temperature: float | None,
) -> dict[str, Any]:
    fixed = {k: v for k, v in (legacy_fixed_params or {}).items() if k in STUDENT_PARAM_KEYS}
    fixed.update(fixed_distillation_params or {})
    if eta is not None:
        fixed["eta"] = eta
    if temperature is not None:
        fixed["temperature"] = temperature
    return fixed


def _cv_objective(
    params: dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: list[str] | None,
    duration_col: str,
    event_col: str,
    cv_folds: int,
    seed: int,
    make_model,
    trial: Any,
) -> float:
    fold_values = []
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(train_df)):
        fold_train = train_df.iloc[fit_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[score_idx].reset_index(drop=True)
        try:
            model = make_model(params)
            model.fit(
                fold_train,
                feature_cols=feature_cols,
                duration_col=duration_col,
                event_col=event_col,
            )
            fold_values.append(validation_nll(model, fold_valid, duration_col=duration_col, event_col=event_col))
        except Exception as exc:
            trial.set_user_attr(f"fold_{fold}_error", repr(exc))
            return float("inf")
    return float(np.mean(fold_values))


def _validate_cv_config(train_df: pd.DataFrame, cv_folds: int) -> None:
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if cv_folds > len(train_df):
        raise ValueError("cv_folds cannot exceed the number of training rows.")


def _split_tuning_data(
    data: pd.DataFrame,
    valid_data: pd.DataFrame | None,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if valid_data is None:
        train_df, valid_df = train_test_split(data, test_size=validation_size, random_state=seed, shuffle=True)
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
    return data.reset_index(drop=True), valid_data.reset_index(drop=True)


def _create_study(optuna: Any, study_name: str, seed: int):
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)


def _with_teacher_time_grid(params: dict[str, Any], teacher_model: Any) -> dict[str, Any]:
    params = dict(params)
    if "time_grid" not in params and getattr(teacher_model, "time_grid", None) is not None:
        params["time_grid"] = teacher_model.time_grid
    return params


def _make_base_model(params: dict[str, Any], num_risks: int, num_durations: int) -> DiscreteSurvivalModel:
    model_params = _model_params(params)
    return DiscreteSurvivalModel(num_risks=num_risks, num_durations=num_durations, **model_params)


def _make_student_model(
    params: dict[str, Any],
    num_risks: int,
    num_durations: int,
    teacher_model: Any,
    teacher_type: str,
) -> DiSKDStudent:
    model_params = _model_params(params)
    if "time_grid" not in model_params and getattr(teacher_model, "time_grid", None) is not None:
        model_params["time_grid"] = teacher_model.time_grid
    student_params = {k: v for k, v in params.items() if k in STUDENT_PARAM_KEYS}
    return DiSKDStudent(
        num_risks=num_risks,
        num_durations=num_durations,
        teacher_model=teacher_model,
        teacher_type=teacher_type,
        **student_params,
        **model_params,
    )


def _model_params(params: dict[str, Any]) -> dict[str, Any]:
    model_params = {k: v for k, v in params.items() if k in MODEL_PARAM_KEYS}
    if model_params.get("backbone") != "transformer":
        model_params.pop("nhead", None)
        return model_params

    hidden_dim = int(model_params.get("hidden_dim", 64))
    nhead = int(model_params.get("nhead", 4))
    if hidden_dim % nhead != 0:
        raise ValueError("For transformer backbones, hidden_dim must be divisible by nhead.")
    return model_params
