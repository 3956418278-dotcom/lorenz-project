#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "usage: $0 <config> <amplitude-run-id>" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-3}"
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" response "$1" --amplitude-run "$2"
