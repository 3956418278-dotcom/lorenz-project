#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/production/single_axis_coefficients_omega_5p938_h4_h6_h8_b512_v1.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

NPROC="${NPROC:-48}"
if (( NPROC < 3 )); then
  echo "dask-mpi requires at least 3 ranks (scheduler, client, worker)" >&2
  exit 2
fi

cd "${RELEASE_ROOT}"
mkdir -p outputs runtime/cache/matplotlib runtime/cache/dask-worker-space

export PYTHONPATH="${RELEASE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${RELEASE_ROOT}/runtime/cache/matplotlib"
export DASK_TEMPORARY_DIRECTORY="${RELEASE_ROOT}/runtime/cache/dask-worker-space"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec mpirun -np "${NPROC}" "${PYTHON_BIN}" -m lorenz.high_order_probe \
  --config "${CONFIG}" "$@"
