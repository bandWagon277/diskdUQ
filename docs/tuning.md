# Hyperparameter Tuning

DiSKD can be used in two modes:

- Fixed/default hyperparameters: instantiate `DiscreteSurvivalModel` or
  `DiSKDStudent` directly and pass values such as `backbone`, `hidden_dim`,
  `lr`, `batch_size`, and `epochs`.
- Optuna search: install the optional tuning extra and call
  `tune_discrete_survival_model(...)` or `tune_diskd_student(...)`. For
  DiSKD students, neural-network parameters and distillation parameters are
  searched in separate spaces.

The tuning workflow is:

1. Provide an already separated training set and, optionally, a validation set
   via `valid_data`.
2. Stage 1 runs Optuna over only neural-network/training parameters with
   K-fold CV inside the training set.
3. Stage 2 fixes the selected neural-network/training parameters, then runs
   Optuna over only DiSKD distillation parameters such as `eta` and
   `temperature`, again with K-fold CV inside the same training set.
4. After selecting hyperparameters, refit one final model on the full training
   set. If `valid_data` is provided, it is used for early stopping during this
   final refit, not for hyperparameter selection.

Optuna is optional so the core package remains lightweight.

```bash
pip install -e ".[tuning]"
```

## Data Splits

The `data` argument is the data used for cross-validation during tuning. If
`valid_data` is supplied, it is held out of Optuna and used only for final
refit early stopping. If `valid_data` is omitted, the helper reserves a random
holdout from `data` for final early stopping and runs CV on the remaining rows.

Teacher predictions are not precomputed by the caller. During each DiSKD trial,
the student queries `teacher_model` on that fold's internal training rows and
uses the resulting teacher signal in the distillation loss.

## Tuned Parameters

The default tuning space covers:

- `backbone`: `mlp`, `time_mlp`, or `transformer`
- `hidden_dim`
- `hidden_layers`
- `dropout`
- `nhead` for transformer backbones
- `lr`
- `batch_size`
- `epochs`
- `optimizer`: `adamw` or `adam`

For DiSKD students, the default distillation space includes:

- `eta`
- `temperature` for competing-risk teacher distillation

`eta` is tuned by default. To fix it, pass `eta=...`,
`fixed_distillation_params={"eta": ...}`, or the backward-compatible
`fixed_params={"eta": ...}`. `temperature` follows the same rule.

`num_risks`, `num_durations`, and the time grid are fixed during one tuning
run. Teacher and student models should use aligned time grids for
time-dependent distillation.

`cv_folds` controls the number of folds used inside the training set. The
default is 5. `n_trials` applies to both stages by default; pass
`model_n_trials` or `distillation_n_trials` to set stage-specific budgets.

## Custom Search Space

Pass separate dictionaries to `model_search_space` and
`distillation_search_space` to override the defaults:

```python
model_search_space = {
    "backbone": ["mlp", "time_mlp", "transformer"],
    "hidden_dim": [32, 64, 128],
    "hidden_layers": {"type": "int", "low": 1, "high": 3},
    "dropout": {"type": "float", "low": 0.0, "high": 0.4},
    "nhead": [2, 4, 8],
    "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "batch_size": [32, 64, 128],
    "epochs": {"type": "fixed", "value": 20},
    "optimizer": ["adamw", "adam"],
}

distillation_search_space = {
    "eta": {"type": "float", "low": 0.0, "high": 5.0},
    "temperature": [1.0, 2.0, 4.0],
}

result = tune_diskd_student(
    train,
    valid_data=valid,
    teacher_model=teacher,
    teacher_type="competing",
    model_search_space=model_search_space,
    distillation_search_space=distillation_search_space,
    cv_folds=5,
    model_n_trials=30,
    distillation_n_trials=30,
    early_stopping_patience=5,
)
```

Supported spec forms:

- list or tuple: categorical choices
- `{"type": "categorical", "choices": [...]}`
- `{"type": "int", "low": ..., "high": ...}`
- `{"type": "float", "low": ..., "high": ..., "log": True}`
- `{"type": "fixed", "value": ...}`

`fixed_model_params` and `fixed_distillation_params` can be used to force
specific values on top of default or custom search spaces. The older
`search_space` and `fixed_params` arguments are still accepted; when used with
`tune_diskd_student`, their keys are split internally into model and
distillation groups.

When `backbone="transformer"`, `hidden_dim` must be divisible by `nhead`.
The default ranges satisfy this constraint. If you provide custom ranges, keep
that compatibility in mind or invalid trials will be assigned an infinite
objective value.

## Objective

The tuning objective in each stage is mean internal negative log-likelihood
across CV folds inside the training set. Stage 1 uses internal-only student
training to select the neural-network/training parameters. Stage 2 uses DiSKD
training with the Stage 1 parameters fixed to select distillation parameters.
Both stages rank trials by held-out fold NLL on the observed survival outcome.

This means `eta` and `temperature` are selected for validation performance on
the internal outcome, not for matching the teacher as strongly as possible. A
large teacher-student KL improvement is useful only if it improves the held-out
survival likelihood under the target event coding.

If `valid_data` is omitted, the tuning helper creates a random holdout from the
provided data and reserves it for final refit early stopping. Hyperparameter
selection still uses CV on the remaining training rows. If `valid_data` is
supplied, the provided `data` is treated as the full training set for CV and the
external validation set is used only for final refit early stopping.
