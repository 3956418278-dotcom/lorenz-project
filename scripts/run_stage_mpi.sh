#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <stage> <mpi-ranks> <config> [stage arguments...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE="$1"
NPROC="$2"
CONFIG="$3"
shift 3

if (( NPROC < 3 )); then
  echo "dask-mpi requires at least 3 ranks (scheduler, client, worker)" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_HASH="$("${PYTHON_BIN}" -c "from lorenz_sine.config import load_config; print(load_config('$CONFIG', task='$STAGE')['config_hash'])")"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%N)_${STAGE}_${CONFIG_HASH}}"
RUN_DIR="results/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" results/packages cache/matplotlib cache/dask-worker-space

CMD=(mpirun -np "${NPROC}" "${PYTHON_BIN}" -m lorenz_sine.cli "${STAGE}" \
  --config "${CONFIG}" --resume "${RUN_ID}" "$@")
set +e
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
CODE=${PIPESTATUS[0]}
set -e
"${PYTHON_BIN}" -m lorenz_sine.cli package --run-id "${RUN_ID}" || true
echo "RUN_ID=${RUN_ID}"
exit "${CODE}"
