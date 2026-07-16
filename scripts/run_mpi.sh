#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PYTHONPATH:-}"

NPROC="${1:-48}"
CONFIG="${2:-configs/server.json}"
MODE="${3:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_HASH="$("$PYTHON_BIN" -c "from lorenz_sine.config import load_config; print(load_config('$CONFIG')['config_hash'])")"
RUN_ID="$(date +%Y%m%d-%H%M%S)_${MODE}_${CONFIG_HASH}"
mkdir -p "results/runs/${RUN_ID}" "results/packages"

CMD=(mpirun -np "$NPROC" "$PYTHON_BIN" -m lorenz_sine.cli run --mode "$MODE" --config "$CONFIG" --resume "$RUN_ID")
set +e
"${CMD[@]}" 2>&1 | tee "results/runs/${RUN_ID}/run.log"
CODE=${PIPESTATUS[0]}
set -e
"$PYTHON_BIN" -m lorenz_sine.cli package --run-id "$RUN_ID" || true
exit "$CODE"
