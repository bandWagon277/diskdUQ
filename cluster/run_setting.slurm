#!/bin/bash
#SBATCH --job-name=diskd-setting
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu
#SBATCH --account=jiankang1

set -euo pipefail
mkdir -p logs

module load cuda/12.1.1 || true

CONDA_ENV="${CONDA_ENV:-diskd}"
if [ -z "${PYTHON:-}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        conda activate "$CONDA_ENV"
    else
        echo "ERROR: conda env '$CONDA_ENV' not found." >&2
        exit 2
    fi
fi

export DEVICE="${DEVICE:-cuda}"
SETTING="${SETTING:-experiments/settings/03_study_a_sgld_diagnostic.env}"

echo "setting: $SETTING"
echo "device: $DEVICE"
bash experiments/run_with_env.sh "$SETTING"

