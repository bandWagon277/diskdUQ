# Reorganization Notes

Changes made in this cleaned repo copy:

1. Copied the reusable package from `src/diskd/` without changing model code.
2. Copied package tests into `tests/`.
3. Moved tutorial scripts into `examples/tutorials/`.
4. Grouped research probes under `experiments/scripts/` by research question.
5. Added `experiments/settings/*.env` so experimental cells are no longer
   encoded only in Slurm scripts or one-off command lines.
6. Added `experiments/run_with_env.sh` as the standard local runner.
7. Added `cluster/run_setting.slurm` as a generic Slurm launcher for any setting.
8. Added `docs/core_model_map.md` to document the core-model boundary.
9. Copied selected findings into `reports/`; raw logs and generated figures are
   excluded.

The original project tree remains unchanged.

