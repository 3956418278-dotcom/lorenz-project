# Lorenz Sine Response

This repository contains the cleaned Lorenz-63 sinusoidal forcing experiment.

## Layout

- `code/lorenz_sine/`: Python package for simulation, spectrum diagnostics, steady-state checks, response estimation, frequency scans, boundary estimates, reporting, storage, and CLI.
- `configs/`: experiment parameter files.
- `scripts/`: MPI and Slurm launch scripts.
- `cache/`: shared spinup and phase-sample caches.
- `results/`: per-run outputs and downloadable packages.

## Smoke Test

```bash
PYTHONPATH=code python -m lorenz_sine.cli run \
  --mode all \
  --config configs/smoke.json \
  --smoke-test
```

## MPI

```bash
bash scripts/run_mpi.sh 48 configs/server.json all
```

## Slurm

```bash
bash scripts/submit_slurm.sh configs/server.json all
```

Each run writes to `results/runs/<run_id>/` and packages that directory as
`results/packages/<run_id>.tar.gz`. Shared caches are written only under
`cache/`.
