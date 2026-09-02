# DiSKD API Notes

This document records the tensor conventions used by the public package and
examples.

## Event Coding

Observed event labels use:

- `0`: censored observation
- `1..J`: competing-risk event causes

Single-risk models treat any positive event value as an observed event. The
examples usually create an explicit `event_any` column for single-risk teachers
so the target remains easy to inspect.

## Time Grid

`fit_time_grid(durations, num_durations)` stores the right endpoint of each
discrete interval. `transform_durations(...)` maps each observed duration to a
zero-indexed interval id in `[0, K - 1]`.

Teacher and student hazards must be evaluated on the same grid for
time-dependent distillation. The examples pass `time_grid=teacher.time_grid` or
construct a shared grid explicitly.

## Training And Tuning

Model `fit(...)` methods accept optional `valid_data`,
`early_stopping_patience`, and `early_stopping_min_delta` arguments. Early
stopping monitors validation internal NLL on the observed survival outcome.

Optuna helpers use K-fold CV inside the provided training data for
hyperparameter selection. `tune_diskd_student(...)` uses two stages on the same
training set: first model/training parameters, then distillation parameters with
the selected model parameters fixed. An external `valid_data` passed to a
tuning helper is reserved for final refit early stopping and is not used as the
trial objective.

High-level `fit(...)` calls follow the same data path:

1. Select feature columns.
2. Fit or reuse a `TimeGrid`.
3. Transform durations to zero-indexed interval ids.
4. Fit the feature preprocessor on the training data.
5. Train the neural model with the internal likelihood and, for
   `DiSKDStudent`, the selected teacher-guidance term.

For teacher-guided training, the teacher is queried on the internal student
rows during `fit(...)`. Teacher and student time grids should be aligned; pass
`time_grid=teacher.time_grid` when using a fitted `DiscreteSurvivalModel` as
the teacher.

## Tensor Shapes

- Single-risk logits: `[N, K]`
- Competing-risk logits: `[N, J, K]`
- Full competing-risk interval probabilities: `[N, J + 1, K]`
- The final probability channel is the no-event category.
- `predict_hazard(data)` returns `[N, K]` for single-risk models and
  `[N, J, K]` for competing-risk models.
- `predict_survival(data)` returns `[N, K]`.
- `predict_cif(data)` returns `[N, J, K]`.
- `predict_cif(data, risk=j)` returns `[N, K]` and uses one-indexed risk labels.

Competing-risk logits are converted to interval probabilities by appending a
fixed no-event logit of zero and applying a softmax over event causes plus the
no-event category. `predict_hazard(...)` omits that no-event category and
returns only event hazards.

## Losses

`CompetingRiskNLLLoss` accumulates interval cross-entropy over intervals where
the subject is at risk. For an observed event, the target is the event cause at
the observed interval. For censoring, all included intervals use the no-event
target.

`CompetingRiskKDLoss` implements DiSKD-C. It adds a teacher-to-student KL term
between full interval distributions, including the no-event category. Temperature
is applied only to the KL term.

`OverallToCompetingRiskKDLoss` implements DiSKD-O. It matches an overall-event
teacher hazard to the student's aggregate event hazard, `sum_j lambda_j(t_k)`.

`BinaryHorizonToCompetingRiskKDLoss` matches a binary teacher probability
`P(event j by tau_h)` to the student's CIF for the same event and horizon.
`binary_risk_index` and `binary_horizon_index` are zero-indexed in
`DiSKDStudent`.

For all KD losses, `eta=0` reduces to the internal likelihood.

The KD losses return `(NLL + eta * KD) / (1 + eta)`. For a fixed `eta`, this is
equivalent for optimization to the unnormalized penalized objective
`NLL + eta * KD`; the normalization keeps loss magnitudes easier to compare
across `eta` values.

## Metrics

`predictive_deviance(interval_probs, idx_durations, events)` computes
`-2 * log_likelihood` under the same discrete-time event coding used by the
losses.

For single-risk predictions, `brier_score(survival, idx_durations, events)`
and `negative_binomial_log_likelihood(...)` return IPCW score curves over the
discrete evaluation grid. `integrated_brier_score(...)` and
`integrated_negative_binomial_log_likelihood(...)` integrate those curves.
These helpers require `survival` with shape `[N, K]` and event labels
`0=censored`, `1=event`.

`concordance_index(durations, events, risk_scores)` treats higher scores as
higher event risk. For competing risks, use
`competing_risk_c_index(cif, durations, events)` to return one C-index per
event cause, or pass `event_of_interest=j` to get a single cause-specific
C-index. The competing-risk C-index is cause-specific: comparable pairs are
anchored by subjects who experienced event cause `j` before another subject's
observed time.

The package intentionally exposes IBS and INBLL only for single-risk survival
predictions. For competing risks, the public API supports the discrete
likelihood/deviance and cause-specific C-index vector.

## Teacher Types

`DiSKDStudent(..., teacher_type="competing")` expects a teacher model with
`predict_interval_probs(data)` returning competing-risk interval probabilities.

`DiSKDStudent(..., teacher_type="overall")` expects either a single-risk teacher
or a competing-risk teacher whose event probabilities can be summed across
causes.

`DiSKDStudent(..., teacher_type="binary_horizon")` accepts either a callable
that returns one probability per input row, or an object with `predict_proba`.
The tutorial uses a callable wrapper so the binary teacher can keep its own
feature preprocessing.
