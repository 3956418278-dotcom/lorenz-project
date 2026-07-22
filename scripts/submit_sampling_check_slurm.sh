#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-6}"
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" sampling-check "${1:-configs/sampling_check_server.json}"
