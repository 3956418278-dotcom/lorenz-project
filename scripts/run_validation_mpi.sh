#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 <mpi-ranks> <config> <response-run-id> <higher-order-run-id>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_stage_mpi.sh" validate "$1" "$2" \
  --response-run "$3" --higher-order-run "$4"
