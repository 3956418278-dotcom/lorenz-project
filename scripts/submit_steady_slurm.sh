#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "usage: $0 <config> <spectrum-run-id>" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-48}"
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" steady "$1" --spectrum-run "$2"
