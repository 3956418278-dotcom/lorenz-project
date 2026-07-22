#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <config> <validation-run-id> [sampling-run-id] [frequency-run-id]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-1}"
ARGS=(report "$1" --input-run "$2")
if [[ -n "${3:-}" ]]; then ARGS+=(--sampling-run "$3"); fi
if [[ -n "${4:-}" ]]; then ARGS+=(--frequency-run "$4"); fi
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" "${ARGS[@]}"
