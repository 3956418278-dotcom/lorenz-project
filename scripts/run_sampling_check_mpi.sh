#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_stage_mpi.sh" sampling-check "${1:-6}" "${2:-configs/sampling_check_server.json}"
