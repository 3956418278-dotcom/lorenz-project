# Lorenz-63 Sinusoidal Response Workflow

This repository estimates first- and second-order statistical response while
separately testing sampling convergence, periodic steady state, higher-order
contamination, detectability, and finite-amplitude truncation validity.  The
old ratio-only `boundary.py` calculation is retired; configured ratios are
reported only as decision diagnostics in the validation stage.

## Repository partitions

- `code/lorenz_sine/`: source code and scientific analysis.
- `configs/`: task-specific formal candidates and small smoke configurations.
- `scripts/`: local MPI and Slurm stage launchers.
- `cache/`: reusable numerical caches plus Matplotlib/Dask runtime caches.
- `results/runs/<run_id>/`: one task run and all of its outputs.
- `results/packages/<run_id>.tar.gz`: independently downloadable run package.

Nothing in the runtime requires Git.  All paths are resolved from the
repository root, and headless plotting uses `MPLBACKEND=Agg`.

## Environment

Create the declared environment instead of assuming server packages:

```bash
mamba env create -f environment.yml
mamba activate lorenz-sine
```

The environment declares NumPy, SciPy, Matplotlib, Dask, Distributed,
`dask-mpi`, and `mpi4py`.

## Deployment

With Git:

```bash
git clone <repository-url> lorenz-project
cd lorenz-project
mamba env create -f environment.yml
```

Without Git, create an upload archive locally:

```bash
tar -czf lorenz-project-upload.tar.gz \
  --exclude=.git --exclude='cache/*.npz' --exclude='results/runs/*' \
  --exclude='results/packages/*' lorenz-project/
scp lorenz-project-upload.tar.gz user@server:/path/to/work/
```

Then on the server:

```bash
cd /path/to/work
tar -xzf lorenz-project-upload.tar.gz
cd lorenz-project
mamba env create -f environment.yml
```

Git metadata is never read to start a run.  Software package versions are
recorded in each manifest.

## Independent stage commands

Every command prints `RUN_ID=<run_id>`.  Substitute that exact ID into the
next command; the code never scans for a “latest” run.

```bash
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/cache/matplotlib"

python -m lorenz_sine.cli spectrum \
  --config configs/spectrum_server.json

python -m lorenz_sine.cli sampling-check \
  --config configs/sampling_check_server.json

python -m lorenz_sine.cli steady \
  --config configs/steady_server.json \
  --spectrum-run <spectrum_run_id>

python -m lorenz_sine.cli amplitude-scan \
  --config configs/amplitude_scan_server.json \
  --steady-run <steady_run_id>

python -m lorenz_sine.cli response \
  --config configs/response_server.json \
  --amplitude-run <amplitude_run_id>

python -m lorenz_sine.cli higher-order \
  --config configs/higher_order_server.json \
  --response-run <response_run_id>

python -m lorenz_sine.cli frequency-scan \
  --config configs/frequency_scan_server.json \
  --response-run <response_run_id>

python -m lorenz_sine.cli validate \
  --config configs/validation_server.json \
  --response-run <response_run_id> \
  --higher-order-run <higher_order_run_id>

python -m lorenz_sine.cli report \
  --config configs/report_server.json \
  --input-run <validation_run_id> \
  --sampling-run <sampling_check_run_id> \
  --frequency-run <frequency_scan_run_id>

python -m lorenz_sine.cli package --run-id <run_id>
```

If `steady` has explicit configured frequencies, `--spectrum-run` is optional.
If its frequency list is empty, omitting that flag is an error.  No command
falls back to `omega=1.0`, and no downstream command silently recomputes a
parent stage.

## Numerical sampling meaning

- `n_phase`: equally spaced phase points per complete forcing period; it must
  be a power of two.
- `n_record_cycles`: complete post-transient forcing periods used for the
  phase average; it must be a power of two.
- `n_record_samples = n_phase * n_record_cycles`: aligned post-transient
  sample count; it must be a power of two.
- `fft_length`: FFT grid length and therefore frequency resolution
  `sample_rate / fft_length`; it must be a power of two.
- `welch_segment_length`: explicit power-of-two Welch window length.
- `welch_overlap_samples`: explicit overlap, strictly smaller than the segment.

For periodic forcing the record contains an integer number of periods, the
fundamental is FFT bin `n_record_cycles`, and the second harmonic is bin
`2*n_record_cycles`.  Each run records frequency resolution, Nyquist
frequency, target-bin checks, and the alignment metadata.  Non-power-of-two
values are rejected before a run is created.

`sampling-check` scans solver tolerances, spinup, phase resolution, cycle
count, FFT length, and Welch length on a deliberately small design.  Its
recommendations are advisory JSON output and never override a formal config.

## Statistical design

The steady stage saves phase means and Fourier coefficients for every
`omega/n_skip/block/seed`.  Seed standard errors and confidence intervals use
the finite-sample Student t distribution at the configured confidence level.
Adjacent-block differences and their confidence intervals drive an explicit,
configurable plateau recommendation, which is also visualized for inspection.

The amplitude stage saves paired `M(+A)`, `M(-A)`, and `M(0)` simulations and
their odd/even decompositions.  The response stage fits all three models

```text
constant
constant + A^2
constant + A^2 + A^4
```

for both L and H.  It saves per-seed coefficients, t and bootstrap confidence
intervals, residuals, R², BIC, leave-one-amplitude prediction error, and
sensitivity to removing the largest or smallest amplitude.  Included and
excluded amplitudes and reasons are explicit.

The higher-order stage evaluates odd cubic residuals, even quartic residuals,
the full quadratic-truncation residual, phase/output/harmonic/overall norms,
and local log-log slopes with seed confidence intervals.  The validation stage
then distinguishes:

- noise-dominated;
- linear-valid;
- second-order-detectable;
- quadratic-truncation-valid;
- higher-order-contaminated.

All detection and relative-error thresholds live in the validation config and
are labelled as configurable decision rules, not natural constants.

## Resume and output contract

Resume the same task with exactly the same config and parent runs:

```bash
python -m lorenz_sine.cli amplitude-scan \
  --config configs/amplitude_scan_server.json \
  --steady-run <steady_run_id> \
  --resume <amplitude_run_id>
```

Per-seed checkpoints are written as each task completes.  A later seed failure
does not discard earlier checkpoints, and resume validates the task, config
hash, and parent IDs before skipping valid parameter points.

Each run contains:

```text
results/runs/<run_id>/
├── config.json
├── manifest.json
├── status.json
├── runtime.json
├── run.log
├── checkpoints/
├── data/
├── tables/
├── figures/
├── report.pdf       # report task
└── result.npz       # compatibility bundle; canonical result is data/result.npz
```

## MPI

`dask-mpi` uses one rank for the scheduler and one for the client:

```text
MPI ranks = desired workers + 2
```

For the formal `n_seed=46` integration stages, 48 ranks provide 46 workers:

```bash
bash scripts/run_spectrum_mpi.sh 48 configs/spectrum_server.json
bash scripts/run_steady_mpi.sh 48 configs/steady_server.json <spectrum_run_id>
bash scripts/run_amplitude_scan_mpi.sh 48 configs/amplitude_scan_server.json <steady_run_id>
```

Sampling-check defaults to six ranks for four worker slots.  Response,
higher-order, frequency-scan, and validation are predominantly client-side
post-processing; their MPI wrappers exist for consistent remote submission,
but normally use only three ranks.  Run the consolidated report serially:

```bash
bash scripts/run_report.sh configs/report_server.json <validation_run_id> \
  <sampling_run_id> <frequency_run_id>
```

## Slurm

Stage launchers accept the environment, task count, time, memory, and Python
executable through environment variables:

```bash
ENV_NAME=lorenz-sine NTASKS=48 TIME_LIMIT=230 MEM_PER_CPU=4000 \
  bash scripts/submit_steady_slurm.sh \
  configs/steady_server.json <spectrum_run_id>

ENV_NAME=lorenz-sine NTASKS=48 TIME_LIMIT=230 MEM_PER_CPU=4000 \
  bash scripts/submit_amplitude_scan_slurm.sh \
  configs/amplitude_scan_server.json <steady_run_id>
```

The spectrum and amplitude/steady launchers default to 48 tasks, sampling
check to 6, post-processing to 3, and report to 1.  Logs stay in the new run
directory; both success and failure paths package the available run state.

## Smoke workflow

Use only `configs/smoke/` for local verification.  The smoke stages must still
be invoked independently with explicit run IDs, exactly like the formal
commands.  Do not use the formal server configs as smoke tests.
