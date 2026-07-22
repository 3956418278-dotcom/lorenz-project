#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NTASKS="${NTASKS:-48}"
exec "${SCRIPT_DIR}/submit_stage_slurm.sh" spectrum "${1:-configs/spectrum_server.json}"
