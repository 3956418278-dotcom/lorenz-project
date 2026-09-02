# Lorenz high-order Dask/MPI release

This release contains the minimal runtime path for
`python -m lorenz.high_order_probe`, taken from branch
`cleanup/minimal-ssm-lorenz` at commit
`7604305bde63d47e22f53e388555cadcaf602e77`.

The numerical integration and response definitions are unchanged. The
release-only execution adapter sends the existing
block-condition integration functions to Dask workers.
With 48 MPI ranks, rank 0 is the scheduler, rank 1 is the client, and the
remaining 46 ranks are one-thread workers.

The package includes the completed-lineage single-axis configuration and a
separate mirrored phase-pair configuration:
`configs/production/mixed_phase_pairs_omega_5p938_b32768_v1.json`.
The mixed design uses common scales `1,1.25,1.5,1.75,2` with base component
amplitudes `xy=(4,2)`, `xz=(4,2.5)`, and `yz=(2,2.5)`, giving 31 conditions
per block including the shared unforced condition.
No config from a completed local experiment is included.

## Upload and install

```bash
rsync -av lorenz-high-order-dask-mpi.tar.gz user@server:/path/to/run/
ssh user@server
cd /path/to/run
tar -xzf lorenz-high-order-dask-mpi.tar.gz
cd lorenz-high-order-dask-mpi
conda env create -f environment.yml
conda activate lorenz-high-order-mpi
```

The `mpirun` executable and `mpi4py` must use the same MPI implementation.
On a module-managed cluster, load its MPI module before creating or activating
the environment as required by the site.

## Run

Pass the required production configuration explicitly. The launcher never
selects an old experiment implicitly:

```bash
./run.sh configs/production/mixed_phase_pairs_omega_5p938_b32768_v1.json
./run.sh configs/production/mixed_phase_pairs_omega_5p938_b32768_v1.json \
  --resume outputs/production/<artifact-directory>
```

The underlying server command is:

```bash
mpirun -np 48 python -m lorenz.high_order_probe --config <config.json>
```

Set `NPROC` or `PYTHON_BIN` to override the rank count or interpreter. Use
`--dry-run` to validate the configuration and MPI startup without integrating:

```bash
./run.sh configs/production/mixed_phase_pairs_omega_5p938_b32768_v1.json --dry-run
```

Generated scientific artifacts are written under `outputs/`; Dask and
Matplotlib caches are written under `runtime/cache/`. Neither location is
inside `src/`.

The mixed run performs its 300-replicate paired-block coefficient analysis
after checkpoint completion. It can also be repeated without integration:

```bash
PYTHONPATH=src python -m lorenz.high_order_probe \
  --analyze-artifact outputs/production/mixed_phase_pairs_omega_5p938_b32768_v1/<artifact-directory>
```

## Included runtime surface

- the `lorenz.high_order_probe` package dependency closure;
- the existing single-axis configuration and the `omega=5.938`, `xy/xz/yz`,
  mirrored-phase, `B=32768` production configuration;
- NumPy, SciPy, Matplotlib, Dask/Distributed, dask-mpi, and mpi4py environment
  declarations.

Historical experiment configs, tests, analysis scripts, prior outputs, MATLAB
files, and project documentation are intentionally excluded.

This configuration retains every block's condition-mean Fourier coefficients
in the existing chunk/checkpoint layout and writes the numerical JSON/CSV
summaries and manifest. It does not retain dense trajectories, per-cycle
arrays, or physical-frequency spectra, and it does not generate figures or
PDFs.
