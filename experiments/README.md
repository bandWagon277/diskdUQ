# Experiments

Experiments are split into two layers:

- `scripts/`: executable probes, copied from the current project and grouped by
  research question.
- `settings/`: named `.env` files that define sample sizes, seeds, eta grids,
  SGLD budgets, output directories, and the script to run.

Use the runner from the repo root:

```bash
bash experiments/run_with_env.sh experiments/settings/03_study_a_sgld_diagnostic.env
```

The runner sets `PYTHONPATH=src`, sources the setting file, creates `OUT_DIR`,
and executes `SCRIPT`. Runtime overrides are allowed:

```bash
DEVICE=cuda SEEDS=42,43,44 bash experiments/run_with_env.sh \
  experiments/settings/06_sgld_omega_full_lastlayer.env
```

## Script Groups

| Group | Main question |
|---|---|
| `manuscript/` | What are the headline simulation regimes and low-dimensional controls? |
| `eta_and_borrowing/` | When should the student borrow from the teacher, and how should eta be selected? |
| `bayesian_correction/` | Can generalized posterior variance be corrected by last-layer sandwich/Godambe ideas? |
| `sgld_diagnostics/` | Is SGLD weak because of parameter dimension, chain geometry, or posterior calibration? |
| `uq_baselines/` | How do non-Bayesian UQ baselines compare to the Bayesian method? |
| `plotting/` | How are saved outputs converted into figures? |
| `smoke_diagnostics/` | Are basic DGP/model scale assumptions sane? |

## Interpretation Rules

- `src/diskd` is the method implementation. Do not add one-off settings there.
- `settings/*.env` defines experimental cells. Add new cells here first.
- `OUT_DIR` should point under `outputs/` unless a paper reproduction requires
  a different path.
- Closed-form true CIF calibration is simulation-only. It is useful for
  diagnosing spread but should not be described as deployable.
- `uq_baselines` are comparators. The main Bayesian method development is in
  SGLD, last-layer posterior approximations, sandwich/Godambe corrections, and
  omega calibration diagnostics.

