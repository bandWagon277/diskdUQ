# DiSKD Reorganized Research Repo

This folder is a runnable reorganization of the current DiSKD project code.
The main cleanup is separation of:

- core model and methodology code: `src/diskd/`
- experiment scripts: `experiments/scripts/`
- experiment settings: `experiments/settings/`
- cluster launchers: `cluster/`
- written findings and proposal notes: `reports/`

The original project tree is not modified. This repo copy is intended to be the
clean working version for method development, simulation studies, and meetings.

## Layout

```text
.
├── src/diskd/                  # reusable package: models, losses, SGLD, metrics
├── tests/                      # package-level regression tests
├── docs/                       # API notes and core model map
├── examples/tutorials/         # small public tutorials
├── experiments/
│   ├── run_with_env.sh         # standard local entry point
│   ├── settings/               # named experiment cells
│   └── scripts/                # research probes grouped by question
├── cluster/                    # generic Slurm runner for settings files
├── reports/                    # selected markdown/txt findings
└── references/                 # proposal/guideline reference notes
```

## Core Method Boundary

`src/diskd/` is the reusable implementation layer. It should not contain
experiment-specific sample sizes, seeds, result paths, or proposal cells.

The core statistical object is the generalized Bayesian DiSKD posterior:

```text
Pi(theta | D, teacher) proportional to
  pi_0(theta) exp{-omega * sum_i [r_i(theta) + eta q_i(theta)]}
```

where `r_i` is the internal discrete-time survival likelihood loss and `q_i`
is a teacher discrepancy loss. In the current package, KD losses are normalized
as `(NLL + eta * KD) / (1 + eta)` for optimization stability; the SGLD wrapper
rescales gradients by `(1 + eta)` so `optimizer="sgld"` targets the intended
omega=1 generalized posterior unless a script explicitly overrides the loss
scale.

Bayesian or generalized-Bayesian inference code lives in:

- `src/diskd/samplers.py`: SGLD optimizer and drift conventions.
- `src/diskd/multichain.py`: cold/warm-start multi-chain SGLD.
- `src/diskd/uncertainty.py`: posterior draws, intervals, R-hat, ESS.
- `experiments/scripts/bayesian_correction/`: last-layer Godambe/sandwich and
  omega calibration probes.

Non-Bayesian UQ engines are kept under
`experiments/scripts/uq_baselines/`. They are comparator baselines, not the
main method claim.

## Quick Start

```bash
cd diskd_reorganized_repo
python -m pip install -e ".[dev,experiments,tuning]"
export PYTHONPATH=src
python examples/tutorials/synthetic_competing_risk_tutorial.py
python -m pytest -q
```

The standard experiment entry point is:

```bash
bash experiments/run_with_env.sh experiments/settings/00_smoke_competing_tutorial.env
```

Each `.env` file defines `SCRIPT=...`, `OUT_DIR=...`, and the relevant sample
sizes, seeds, SGLD settings, eta grid, or comparator engines. You can override
any setting at runtime:

```bash
SEEDS=42,43,44 DEVICE=cuda bash experiments/run_with_env.sh \
  experiments/settings/03_study_a_sgld_diagnostic.env
```

## Main Experiment Settings

| Setting | Purpose | Script |
|---|---|---|
| `00_smoke_competing_tutorial.env` | Fast API smoke test | `examples/tutorials/synthetic_competing_risk_tutorial.py` |
| `01_guideline_control.env` | Guideline simulation control cell | `experiments/scripts/manuscript/guideline_sim.py` |
| `02_eta_selection.env` | Eta borrowing selection by internal validation deviance | `experiments/scripts/eta_and_borrowing/eta_selection_probe.py` |
| `03_study_a_sgld_diagnostic.env` | Full-network vs last-layer-only SGLD diagnostic | `experiments/scripts/sgld_diagnostics/sgld_parameter_diagnostic_probe.py` |
| `04_uq_baseline_comparator.env` | Deep/bootstrap/MC-dropout/SGLD comparator baseline table | `experiments/scripts/uq_baselines/uq_engine_comparator_probe.py` |
| `05_last_layer_godambe.env` | Bayesian last-layer sandwich/Godambe correction | `experiments/scripts/bayesian_correction/sandwich_correction_probe.py` |
| `06_sgld_omega_full_lastlayer.env` | Omega sweep for full vs last-layer SGLD | `experiments/scripts/sgld_diagnostics/sgld_omega_sweep_full_vs_lastlayer.py` |
| `07_wellspecified_linear.env` | Low-dimensional well-specified control | `experiments/scripts/manuscript/linear_wellspec_probe.py` |

Important interpretation: `uq_engine_comparator_probe.py` includes an oracle
scalar calibration based on closed-form true CIF. That scalar is diagnostic for
simulation only; it is not a deployable correction method.

## Cluster Usage

Use one generic Slurm wrapper and choose the setting file:

```bash
sbatch --export=ALL,SETTING=experiments/settings/03_study_a_sgld_diagnostic.env \
  cluster/run_setting.slurm
```

The cluster script activates `CONDA_ENV=diskd` by default and then calls
`experiments/run_with_env.sh`.

## Current Code Map

For the detailed model and experiment map, read:

- `docs/core_model_map.md`
- `docs/api.md`
- `experiments/README.md`
- `reports/study_a_bc_findings.md`
- `reports/sandwich_correction_findings.md`
- `reports/omega_vs_sandwich_explanation.txt`

