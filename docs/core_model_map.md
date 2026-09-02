# Core Model And Experiment Map

This document records what belongs to the reusable DiSKD model layer and what
belongs to experiment settings.

## Core Package: `src/diskd`

| File | Main objects | Role |
|---|---|---|
| `models.py` | `DiscreteSurvivalModel`, `DiSKDStudent` | High-level fit/predict wrappers. Owns preprocessing, time grid, optimizer choice, teacher querying, and prediction APIs. |
| `losses.py` | `SingleRiskNLLLoss`, `CompetingRiskNLLLoss`, KD losses | Internal likelihood losses and teacher discrepancy losses. KD losses normalize `(NLL + eta * KD) / (1 + eta)`. |
| `networks.py` | `MLPBackbone`, `TimeEmbeddingMLP`, `TransformerBackbone` | Neural logits for single-risk or competing-risk discrete hazards. |
| `samplers.py` | `SGLD` | SGLD optimizer with `welling_teh`, `literal`, adaptive, preconditioned, and cyclical variants. |
| `multichain.py` | `MultiChainSampler`, `WarmStartMultiChainSampler` | Independent SGLD chain orchestration and pooled posterior samples. |
| `uncertainty.py` | Posterior prediction and diagnostics | Draw-level predictions, credible intervals, R-hat, ESS, coverage helpers. |
| `simulation.py` | Synthetic competing-risk cohorts | Shared synthetic DGP for teacher/student/test cohorts. |
| `_ground_truth.py` | `true_cif_at_grid` | Closed-form true CIF for simulation-only coverage diagnostics. |
| `metrics.py` | Deviance, C-index, Brier, log-likelihood | Evaluation utilities. |
| `preprocessing.py` | `TimeGrid`, `FeaturePreprocessor` | Duration binning and feature preprocessing. |
| `utils.py` | CIF/survival/probability transforms | Tensor transforms shared by losses, metrics, and correction probes. |
| `tuning.py` | Optuna helpers | Optional hyperparameter search. |

## Current Model Design

The production-facing model API is:

```python
from diskd import DiscreteSurvivalModel, DiSKDStudent

teacher = DiscreteSurvivalModel(num_risks=2, num_durations=12)
student = DiSKDStudent(
    teacher_model=teacher,
    teacher_type="competing",
    eta=1.0,
    temperature=2.0,
)
```

`DiscreteSurvivalModel` trains an internal-only discrete survival model.
`DiSKDStudent` extends it with teacher guidance:

- `teacher_type="competing"`: teacher gives cause-specific interval probabilities.
- `teacher_type="overall"`: teacher gives overall event probability.
- `teacher_type="binary_horizon"`: teacher gives one event-by-horizon probability.

For competing risks, logits have shape `[N, J, K]`. A no-event channel is
appended internally before softmax, producing interval probabilities
`[N, J + 1, K]`. CIF is derived from the interval event probabilities and the
previous-interval survival.

## Bayesian Components

The Bayesian/generalized-Bayesian part is modular:

- Full-network SGLD: use `DiSKDStudent(..., optimizer="sgld")`.
- Warm-start SGLD: fit an AdamW MAP, then pass the state dict to
  `WarmStartMultiChainSampler`.
- Last-layer-only SGLD: experiment subclasses freeze all parameters except
  `net.head`.
- Last-layer Godambe/sandwich correction: experiment scripts freeze the
  backbone, compute low-dimensional curvature/score matrices for the head, and
  propagate covariance to CIF by a delta method.
- Omega calibration diagnostics: experiment subclasses multiply SGLD loss scale
  by `sgld_omega`.

The package deliberately keeps the posterior engine separate from non-Bayesian
comparators. Deep ensembles, bootstrap ensembles, and MC dropout live in the
UQ baseline scripts only.

## Experiment Categories

| Directory | Purpose |
|---|---|
| `experiments/scripts/manuscript/` | Manuscript/proposal simulation cells and low-dimensional controls. |
| `experiments/scripts/eta_and_borrowing/` | Eta selection, teacher borrowing, horizon, and teacher-quality diagnostics. |
| `experiments/scripts/bayesian_correction/` | Last-layer sandwich/Godambe and omega-vs-sandwich probes. |
| `experiments/scripts/sgld_diagnostics/` | SGLD convergence, parameter-space, warm-start, step-size, and omega studies. |
| `experiments/scripts/uq_baselines/` | Comparator UQ engines. These are baselines, not main Bayesian methods. |
| `experiments/scripts/plotting/` | Plotting from saved experiment outputs. |
| `experiments/scripts/smoke_diagnostics/` | Small sanity checks for DGP or scale behavior. |

## What Should Stay Out Of `src/diskd`

Keep these in setting files or experiment scripts:

- sample sizes such as `TEACHER_N`, `STUDENT_N`, and `TEST_N`
- seed grids
- eta, omega, and SGLD grids
- output directories
- simulation headline cells
- Slurm/account/GPU settings
- oracle true-CIF calibration used only in simulation diagnostics

