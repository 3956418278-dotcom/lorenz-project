#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <experiment-config.json> [--resume <artifact-directory>]" >&2
  exit 2
fi
CONFIG="$1"
shift

if [[ -f "${CONFIG}" ]]; then
  CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
elif [[ -f "${RELEASE_ROOT}/${CONFIG}" ]]; then
  CONFIG="${RELEASE_ROOT}/${CONFIG}"
else
  echo "experiment config does not exist: ${CONFIG}" >&2
  exit 2
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
