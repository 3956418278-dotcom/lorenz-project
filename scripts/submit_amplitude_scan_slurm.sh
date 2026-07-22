#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "usage: $0 <config> <steady-run-id>" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-48}"
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" amplitude-scan "$1" --steady-run "$2"
