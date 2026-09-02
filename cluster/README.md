# Cluster Launchers

Use the generic setting-based Slurm wrapper:

```bash
sbatch --export=ALL,SETTING=experiments/settings/03_study_a_sgld_diagnostic.env \
  cluster/run_setting.slurm
```

Useful overrides:

```bash
sbatch --export=ALL,SETTING=experiments/settings/06_sgld_omega_full_lastlayer.env,DEVICE=cuda,SEEDS=42,43,44 \
  cluster/run_setting.slurm
```

The wrapper assumes:

- Slurm partition/account are compatible with the local cluster defaults.
- `CONDA_ENV=diskd` exists, unless `PYTHON=/path/to/python` is provided.
- The job is submitted from the repo root.

