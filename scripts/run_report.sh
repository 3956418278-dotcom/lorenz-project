#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <config> <validation-run-id> [sampling-run-id] [frequency-run-id]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
CMD=("${PYTHON_BIN:-python}" -m lorenz_sine.cli report --config "$1" --input-run "$2")
if [[ -n "${3:-}" ]]; then CMD+=(--sampling-run "$3"); fi
if [[ -n "${4:-}" ]]; then CMD+=(--frequency-run "$4"); fi
exec "${CMD[@]}"
