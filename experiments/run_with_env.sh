#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTING="${1:-experiments/settings/00_smoke_competing_tutorial.env}"
if [ "$#" -gt 0 ]; then
    shift
fi

if [[ "$SETTING" != /* ]]; then
    SETTING="$ROOT/$SETTING"
fi
if [[ ! -f "$SETTING" ]]; then
    echo "ERROR: setting file not found: $SETTING" >&2
    exit 2
fi

set -a
source "$SETTING"
set +a

if [[ -z "${SCRIPT:-}" ]]; then
    echo "ERROR: setting file must define SCRIPT=..." >&2
    exit 2
fi
if [[ "$SCRIPT" != /* ]]; then
    SCRIPT="$ROOT/$SCRIPT"
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: script not found: $SCRIPT" >&2
    exit 2
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DEFAULT_OUT="$ROOT/outputs/$(basename "${SETTING%.env}")"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT}"
if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT/$OUT_DIR"
fi
export OUT_DIR
mkdir -p "$OUT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "ERROR: no python executable found; set PYTHON=/path/to/python" >&2
        exit 2
    fi
fi
if [[ -z "${CONDA_PREFIX:-}" && "$PYTHON" == */envs/*/bin/python ]]; then
    CONDA_PREFIX="${PYTHON%/bin/python}"
    export CONDA_PREFIX
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

echo "repo: $ROOT"
echo "setting: $SETTING"
echo "script: $SCRIPT"
echo "out_dir: $OUT_DIR"
echo "python: $("${PYTHON}" -c 'import sys; print(sys.executable)')"

exec "$PYTHON" "$SCRIPT" "$@"
