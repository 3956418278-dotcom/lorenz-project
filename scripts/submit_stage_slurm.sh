#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <stage> <config> [stage arguments...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE="$1"
CONFIG="$2"
shift 2

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
ENV_NAME="${ENV_NAME:-lorenz-sine}"
NTASKS="${NTASKS:-48}"
TIME_LIMIT="${TIME_LIMIT:-230}"
MEM_PER_CPU="${MEM_PER_CPU:-4000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_HASH="$("${PYTHON_BIN}" -c "from lorenz_sine.config import load_config; print(load_config('$CONFIG', task='$STAGE')['config_hash'])")"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%N)_${STAGE}_${CONFIG_HASH}}"
RUN_DIR="results/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" results/packages cache/matplotlib cache/dask-worker-space

CLI=("${PYTHON_BIN}" -m lorenz_sine.cli "${STAGE}" --config "${CONFIG}" --resume "${RUN_ID}" "$@")
printf -v CLI_QUOTED '%q ' "${CLI[@]}"
printf -v ROOT_QUOTED '%q' "${PROJECT_ROOT}"
printf -v ENV_QUOTED '%q' "${ENV_NAME}"
printf -v RUN_QUOTED '%q' "${RUN_ID}"
WRAP="cd ${ROOT_QUOTED}; export PYTHONPATH=${ROOT_QUOTED}/code:\${PYTHONPATH:-}; export PYTHONDONTWRITEBYTECODE=1; export MPLBACKEND=Agg; export MPLCONFIGDIR=${ROOT_QUOTED}/cache/matplotlib; mamba activate ${ENV_QUOTED}; srun -n \$SLURM_NTASKS ${CLI_QUOTED}; CODE=\$?; ${PYTHON_BIN} -m lorenz_sine.cli package --run-id ${RUN_QUOTED} || true; exit \$CODE"

sbatch -n "${NTASKS}" --hint=compute_bound -t "${TIME_LIMIT}" \
  --mem-per-cpu="${MEM_PER_CPU}" -o "${RUN_DIR}/slurm-%j.out" --wrap="${WRAP}"
echo "RUN_ID=${RUN_ID}"
