#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PYTHONPATH:-}"

CONFIG="${1:-configs/server.json}"
MODE="${2:-all}"
ENV_NAME="${ENV_NAME:-hcmc_jas}"
NTASKS="${NTASKS:-256}"
TIME_LIMIT="${TIME_LIMIT:-230}"
MEM_PER_CPU="${MEM_PER_CPU:-4000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_HASH="$("$PYTHON_BIN" -c "from lorenz_sine.config import load_config; print(load_config('$CONFIG')['config_hash'])")"
RUN_ID="$(date +%Y%m%d-%H%M%S)_${MODE}_${CONFIG_HASH}"
RUN_DIR="results/runs/${RUN_ID}"
mkdir -p "$RUN_DIR" "results/packages"

CMD="mamba activate $ENV_NAME; time srun -n \$SLURM_NTASKS $PYTHON_BIN -m lorenz_sine.cli run --mode $MODE --config $CONFIG --resume $RUN_ID"

sbatch -n "$NTASKS" --hint=compute_bound -t "$TIME_LIMIT" --mem-per-cpu="$MEM_PER_CPU" \
  -o "$RUN_DIR/slurm-%j.out" \
  --wrap="$CMD; $PYTHON_BIN -m lorenz_sine.cli package --run-id $RUN_ID"
