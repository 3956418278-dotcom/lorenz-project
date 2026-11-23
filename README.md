# Lorenz-63 Forced ODE & Harmonic Response Toolkit

A lightweight numerical Python/C99 toolkit for integrating periodically forced Lorenz-63 trajectories, performing uniform-phase Fourier projection and phase-space moment accumulation, and fitting matched-block polynomial response models.

## Features

- **Native C99 DOP853 fast path (`src/lorenz/_dop853.c`, `src/lorenz/_fast_dop853.py`)**:
  - Implements the 8th-order Dormand–Prince (`DOP853`) adaptive Runge–Kutta tableau with 7th-order dense output directly in C99 (`ctypes`), with transparent fallback to `scipy.integrate.solve_ivp`.
  - Fused single-frequency cycle-Fourier projection, phase-space moment accumulation, and two-tone arbitrary-frequency DFT projection kernels.
- **Reproducible Ensemble Sampling (`src/lorenz/ensemble.py`)**:
  - Order-independent `PCG64DXSM` block streams keyed by immutable integer block IDs (`SeedSequence.spawn_key`) with parallel spinup support.
- **Condition Planners & Response Conventions (`src/lorenz/conditions.py`, `src/lorenz/response.py`)**:
  - Paired sign contrasts (`+h`, `-h`, unforced reference), mirrored two-axis phase-pair designs, and linear/quadratic tensor reconstruction utilities.
- **Resumable Checkpointing & Parallel Backends (`src/lorenz/high_order_probe.py`, `src/lorenz/parallel.py`, `src/lorenz/persistence.py`)**:
  - Rolling task scheduling over local `ProcessPoolExecutor`, local Dask, or `dask-mpi`, with chunked restart files and SHA-256 artifact manifests.
- **Statistical & Coordinate Utilities (`analysis/regression.py`, `analysis/response_coordinates.py`)**:
  - Crossed-block generalized least-squares (GLS) polynomial fits, joint Hotelling-$F$ lack-of-fit / nested model tests, and Fourier-to-sinusoid coordinate conversions.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate lorenz-toolkit
```

A C99 compiler (`gcc` or `clang`) enables automatic compilation of the native DOP853 extension; if unavailable, the solver automatically falls back to SciPy. To force the pure-SciPy reference path, set `LORENZ_DISABLE_C_SOLVER=1`.

## Quickstart

### 1. Dry-run a configuration

```bash
PYTHONPATH=src python -m lorenz.high_order_probe \
  --config configs/example_single_axis.json \
  --dry-run
```

### 2. Run a small local example

```bash
PYTHONPATH=src python -m lorenz.high_order_probe \
  --config configs/example_single_axis.json
```

Artifacts are written under `outputs/` (ignored by Git).

### 3. Direct Python API

```python
import numpy as np
from lorenz.core import simulate_phase_samples
from lorenz.response import phase_fourier

cfg = {
    "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "solver": {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11},
}

samples = simulate_phase_samples(
    initial_state=np.array([1.0, 1.0, 20.0]),
    forcing_vector=np.array([0.0, 0.5, 0.0]),
    omega=2.0,
    phase=0.0,
    discard_time=5.0,
    n_cycle=16,
    n_phase=32,
    cfg=cfg,
)
coeffs = phase_fourier(samples.values, harmonics=[0, 1, 2], phase_offset=samples.phase_offset)
```

## Running Tests

```bash
PYTHONPATH=.:src pytest -v
```

## Repository Structure

```text
.
├── src/lorenz/                 # ODE solver, C99 DOP853 kernel, ensemble & Fourier sampling, persistence
├── analysis/                   # Generic crossed-block GLS regression & coordinate utilities
├── configs/                    # Minimal synthetic example configurations
├── tests/                      # Self-contained unit & parity tests
├── environment.yml
└── pytest.ini
```
