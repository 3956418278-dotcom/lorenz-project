# Current State

## Project Definition

The project measures first- and second-order frequency response of the
sinusoidally forced Lorenz-63 system. The confirmed tensor definitions,
Fourier signs and factors, ensemble estimand, and replication unit are in
`PROJECT.md` and are unchanged.

Current scope is monochromatic forcing with full-state observation `(x,y,z)`:

- `chi1(omega)`: complex first-order susceptibility;
- `chi2(2*omega; omega,omega)`: complex effective input-symmetric
  second-harmonic susceptibility identified by same-direction contractions;
- `Qdc(omega)`: real rectification coefficient, reported separately rather
  than as a complete ordered-frequency DC susceptibility.

An initial-state block is the sampling replication unit. Conditions are
crossed and contrasted within block; uncertainty is summarized across blocks.
Finite spinup, forced-transient removal, solver accuracy, and phase resolution
are numerical-bias questions separate from sampling uncertainty.

## Current Capabilities and Ownership

- `src/lorenz/core.py` owns Lorenz integration and phase/dense sampling.
- `src/lorenz/ensemble.py` owns reproducible block generation and the common
  run-config-to-ensemble adapter.
- `src/lorenz/response.py` owns paired odd/even contrasts, Fourier conventions,
  finite-strength response normalization, and tensor reconstruction.
- `src/lorenz/statistics.py` owns block-level cos/sin statistics, covariance,
  Hotelling tests, and Benjamini-Hochberg adjustment.
- `src/lorenz/strength_study.py` owns crossed-strength sampling and retained
  Fourier/dense summaries; `strength_series.py` and `strength_bootstrap.py` own
  cross-strength fits and whole-block simultaneous inference.
- `src/lorenz/retention.py` owns condition-axis identity, retained-object
  policy, chunk layout, and selected-condition spectrum loading.
- `src/lorenz/response_plots.py` owns reusable response rendering, including
  high-order probe figures.
- `src/lorenz/high_order_probe.py` composes the one-frequency high-order
  signal-discovery run, extension compatibility, inference tables, and artifact
  persistence. `experiments/run_high_order_probe.py` is its thin CLI.
- `experiments/plot_high_order_probe_figure_a.py` is a streaming view of a
  completed artifact. It reuses shared condition/contrast/noise definitions and
  writes presentation outputs under `outputs/figures/`, outside immutable
  simulation artifacts.
- `experiments/analyze_single_axis_coefficients.py` is the read-only
  coefficient analysis for the `omega=5.938` completed artifact. It uses the
  crossed block covariance over all seven amplitudes and currently restricts
  inputs to the three coordinate axes.

The active standalone high-order production configurations are:

- `configs/production/high_order_probe_omega_6p7_v1.json`;
- `configs/production/high_order_probe_omega_5p938_v1.json`.

Each describes the complete seven-strength design directly. Incremental
base/extension configuration history is retained only in executed artifacts'
`config_snapshot.json` and manifests.

## Confirmed Numerical and Statistical Evidence

The conservative working numerical profile remains provisional outside the
protocols where it was checked:

- initial-state spinup: `40` Lorenz time;
- forced discard: `160` Lorenz time;
- observation: at least `64` cycles and about `400` physical time for the
  dense/high-order designs;
- phase grid: `32` samples per cycle;
- solver: DOP853, `rtol=1e-9`, `atol=1e-11`.

These are working design parameters, not universal residual-bias bounds or a
proof of exact sampling from the SRB measure.

### Perturbative-response evidence

The x-direction dense pilot covered 12 response-independent frequencies
`0.5`--`22.6`, strengths `{0.25,0.5,1,2}` with six-strength anchor extensions,
and `B=64` blocks.

- `chi1_xx` was identified at `{0.5,1.0,1.4,2.0,4.0,5.6,8.0,16.0,22.6}` and
  not at `{0.7,2.8,11.3}` under stored whole-block simultaneous 95% bounds.
- The elevated `5.6--8` band relative to `2.8--4` and the falloff above `8`
  were resolved. A low-frequency plateau/dip, separation of `5.6` from `8`,
  and high-tail flatness were not resolved.
- Second harmonic and DC were not identified anywhere. Stored upper bounds,
  rather than point estimates, are the scientific result.
- Fitted higher-order coefficients and direct harmonics `n=3,4,5` were not
  identified. No cell satisfied the provisional joint requirement of leading
  response identification and a simultaneous <=20% higher-order bound.

Canonical analysis artifact:
`outputs/exploratory/x_response_merged_v1/20260815T133813_merged/`.

### High-order signal-discovery evidence

Two completed `B=256` probes use six forcing directions, outputs `(x,y,z)`,
harmonics `n=0..5`, and strengths `{2,4,8,10,12,14,16}`. Inference uses the
joint signed cos/sin Hotelling statistic with BH `q<=0.05` over a 630-cell
non-DC family at each frequency.

- At `omega=6.7`, detected-cell counts for `n=1..5` are
  `91,59,45,15,22`.
- At `omega=5.938`, detected-cell counts for `n=1..5` are
  `92,49,46,3,8`.
- Fourth and fifth harmonics are established at sufficiently large forcing,
  but no sharp monotone onset is claimed.
- Their strongest responses remain below the median amplitude of one unforced
  block at the corresponding frequency. Detection comes from coherent paired
  ensemble response, not from exceeding single-block chaotic fluctuations.
- Every redundant extension-unforced chunk matches its immutable base exactly
  for condition means, retained cycle Fourier coefficients, complex spectra,
  and spectrum segment counts.

Canonical artifacts:

- `outputs/production/high_order_probe_omega_6p7_h10_h12_h14_h16_extension_v1/20260822T061155_75ca70131a4b/`;
- `outputs/production/high_order_probe_omega_5p938_h10_h12_h14_h16_extension_v1/20260822T061155_b1e645f07e98/`.

These are large-amplitude signal-discovery results. Especially `h>=8` is not
interpreted as a perturbative tensor estimate.

### Single-axis coefficient estimates at `omega=5.938`

The first coefficient analysis uses the self-contained final `B=256` artifact
once; its base and extension artifacts share block IDs and are provenance, not
additional replicates. All strengths `{2,4,8,10,12,14,16}` enter the fit.

- Signed first-harmonic odd contrasts require raw powers `{h,h^3,h^5}` to pass
  the global crossed-block Hotelling-F lack-of-fit check over
  Lorenz-parity-allowed components; the subsequent `h^7` term gives a marginal
  but retained improvement (`nested p=0.0403`). The selected
  `{h,h^3,h^5,h^7}` model has lack-of-fit `p=0.500`.
- A constant `even(n=2)/h^2` coefficient is inconsistent with the amplitude
  series (`p=1.33e-14`). Raw powers `{h^2,h^4}` are the lowest adequate model
  (`p=0.540`), so the reported quadratic coefficient accounts for visible
  quartic contamination rather than using the `h=16` ratio directly.
- Adding `h^6` is marginal (`nested p=0.0683`) and is retained in the model and
  fit-range sensitivity records. Pointwise 95% intervals are conditional on
  the selected `{h^2,h^4}` model.
- For the three Lorenz-parity-allowed diagonal-input second-harmonic
  coefficients `Q` defined by `even_hat_i(2 omega)=h^2 Q_i,jj+O(h^4)`, the
  signed `(cos,sin)` estimates are: `Q_z,xx=(0.0001812,-0.0001670)`,
  `Q_z,yy=(0.0008267,0.0001895)`, and
  `Q_z,zz=(0.0003194,-0.0001990)`. All nine raw estimates and intervals are in
  the analysis artifact.

Analysis artifact:
`outputs/analysis/single_axis_coefficients_omega_5p938_v1/`.

## Current Scientific Decision

The project has not established an overlap between:

1. an asymptotic strength range where higher-order contamination is bounded;
2. an identification range where the response exceeds chaotic sampling noise.

The completed high-order probes establish visibility of harmonics `n=1..5` at
large forcing. The single-axis calculation now provides model-based
zero-strength extrapolations, but it does not establish a directly resolved
small-amplitude perturbative window: `h=2` is weak for second order, while the
large amplitudes require higher-power correction. The reported pointwise
intervals therefore do not include polynomial-order uncertainty.

Cross-input `xy`, `xz`, and `yz` coefficients have deliberately not been
calculated. A future expensive amplitude-scaling design remains unconfirmed
and requires user judgment; no new simulation is authorized by the current
analysis.

The complete two-frequency second-order surface and multi-tone forcing remain
outside scope.

## Working Constraints

- Completed artifacts are immutable evidence; presentation and re-analysis
  outputs belong under `outputs/figures/` or another derived-data area.
- Exploration and formal evidence remain explicitly separated.
- Retention for a future perturbative production design must be selected from
  the storage-scaling report rather than inherited automatically.
- Linux numerical checks use
  `/home/feng/miniforge3/envs/ody/bin/python` (NumPy 2.4.3, SciPy 1.15.2),
  which lacks pytest.
- The complete pytest suite runs in the Windows Python environment documented
  by `environment.yml`.
- Personal tool settings and temporary arrays are ignored and not part of the
  repository. A credential removed from the current tree remains present in
  Git history until separately rotated and history-cleaned with explicit
  approval.
