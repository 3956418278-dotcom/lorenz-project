#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 <config> <response-run-id> <higher-order-run-id>" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-3}"
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" validate "$1" \
  --response-run "$2" --higher-order-run "$3"
