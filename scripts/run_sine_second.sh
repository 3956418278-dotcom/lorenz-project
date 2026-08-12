#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
LORENZ_PYTHON_BIN="${LORENZ_PYTHON:-python}"
exec "${LORENZ_PYTHON_BIN}" -m lorenz_sine.sine_second "$@"
