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
- `src/lorenz/high_order_probe.py` composes one-frequency harmonic runs over
  configured paired forcing directions, extension compatibility, inference
  tables, and artifact persistence. `experiments/run_high_order_probe.py` is
  its thin CLI.
- `experiments/plot_high_order_probe_figure_a.py` is a streaming view of a
  completed artifact. It reuses shared condition/contrast/noise definitions and
  writes presentation outputs under `outputs/figures/`, outside immutable
  simulation artifacts.
- `experiments/analyze_single_axis_coefficients.py` is the read-only
  coefficient analysis for the `omega=5.938` completed artifact. It uses the
  crossed block covariance over all seven amplitudes and currently restricts
  inputs to the three coordinate axes.
- `experiments/analyze_quadratic_b512_refinement.py` is the focused read-only
  comparison of the nested `B=256` and `B=512`, `h={4,6,8}` single-axis
  designs. It reuses the coefficient analysis definitions and reports only the
  three allowed `z`-output diagonal-input quadratic coefficients.

The active standalone high-order production configurations are:

- `configs/production/high_order_probe_omega_6p7_v1.json`;
- `configs/production/high_order_probe_omega_5p938_v1.json`.

These describe the complete seven-strength designs directly. Incremental
base/extension configuration history is retained only in executed artifacts'
`config_snapshot.json` and manifests.

The completed single-axis refinement configurations are:

- `configs/production/single_axis_coefficients_omega_5p938_h5_h6_h7_v1.json`
  (`B=256`, filling the amplitude gap);
- `configs/production/single_axis_coefficients_omega_5p938_h4_h6_h8_b512_v1.json`
  (`B=512`, refining the candidate quadratic regime).

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

The coefficient analysis combines the immutable seven-amplitude artifact with
the completed axis-only `h={5,6,7}` artifact. Both use exactly the ordered block
IDs `2700000..2700255`; all numerical, ensemble, sampling, dense, and retention
settings agree, and their independently regenerated unforced coefficients are
bitwise identical. The merged crossed-block grid is
`{2,4,5,6,7,8,10,12,14,16}`. Mixed directions were not simulated.

The zero-strength audit compares raw powers `{h,h^3}` versus `{h,h^3,h^5}`
and `{h^2,h^4}` versus `{h^2,h^4,h^6}` on 13 progressive upper-amplitude
windows, both including `h=2` and excluding it. Its 416 fits retain signed
cosine and sine separately with block-level pointwise 95% intervals.

- For first order, `{h,h^3,h^5}` passes the global lack-of-fit check
  (`p=0.800`); the next `h^7` improvement is borderline but not retained at the
  prespecified threshold (`p=0.0507`). Every allowed first-order coefficient is
  nevertheless amplitude-window-sensitive once the newly available low
  windows are tested. `L_yx`, `L_yy`, and both parts of `L_zz` lack a common
  overlap across all fit intervals.
- For second order, quartic correction remains necessary. The added data now
  support a global `h^6` contribution (`nested p=0.00152`), confirming that
  higher-even-order curvature is not just an `h=16` effect.
- `Q_z,xx` is borderline: all fit intervals overlap and its largest order and
  window shifts are about `1.77` and `1.93` median sampling half-widths.
  `Q_z,yy` and `Q_z,zz` remain model-sensitive, driven primarily by their sine
  intercepts; `Q_z,yy` sine has no common all-fit interval.

A focused candidate-regime diagnostic excludes `h=2` from fitting, uses
`h={4,5,6,7,8}` for `Q(h)=Q+C4*h^2`, and treats the `h=4..16` fit only as a
large-amplitude reference. On `h=4..8`, constant `Q(h)` is jointly adequate
(`p=0.717`), adding `C4` is not jointly supported (`p=0.227`), and five of six
component-wise `C4` intervals include zero. The exception is `Q_z,yy` sine:
`C4=-6.67e-6` with 95% interval `[-1.30e-5,-3.75e-7]`; its low-range and
full-range intercept intervals do not overlap. Thus there is no single regime
that is both measured and contamination-free for all three complex
coefficients. `Q_z,yy` cosine is the clearest stable measured component;
`Q_z,xx` and both `Q_z,zz` low-range intercepts remain sampling-limited.

Adding `h=5,6,7` therefore does not resolve the zero-strength intercept. It
shows that sparse amplitude placement was not the sole cause: sampling noise
still destabilizes short high-order fits, while systematic curvature persists
across the longer windows. No single final intercept is selected.

The subsequent nested `B=512` refinement at `h={4,6,8}` preserves the first
256 block IDs exactly. Its regenerated unforced coefficients and all raw
first-256 odd/even contrasts are bitwise identical to the earlier `B=256`
artifacts. For the six signed components of `Q_z,xx`, `Q_z,yy`, and `Q_z,zz`,
the fitted model is `even(h)/h^2 = Q + C4*h^2` with full crossed-block
covariance.

- The `Q` interval half-width ratios, `B=512` over matched `B=256`, are
  `0.692--0.733`, consistent with the expected `0.705` sampling reduction.
  Every nested central-value shift is compatible with zero.
- `Q_z,yy` cosine and sine and `Q_z,zz` cosine are measured at `B=512`; the
  two `Q_z,xx` components and `Q_z,zz` sine remain sampling-limited.
- Only `Q_z,yy` sine has a resolved finite-amplitude slope:
  `C4=-5.73e-6`, 95% interval `[-1.09e-5,-5.71e-7]`. The corresponding fitted
  change from `h=4` to `h=8` is `-2.75e-4`, about 43% of its extrapolated
  intercept.
- Across all six signed components, the constant model remains jointly
  adequate (`p=0.398`), adding `C4` is not jointly required (`p=0.236`), and
  the `Q+C4*h^2` model has no resolved residual curvature (`p=0.579`). These
  global statements do not erase the individually resolved `Q_z,yy` sine
  slope.

Analysis artifact:
`outputs/analysis/single_axis_coefficients_omega_5p938_h5_h6_h7_v2/`.

Refinement analysis artifact:
`outputs/analysis/single_axis_quadratic_b512_refinement_v1/`.

Single-axis production artifacts:

- `outputs/production/single_axis_coefficients_omega_5p938_h4_h6_h8_b512_v1/20260824T125050_d90049066676/`;
- `outputs/production/single_axis_coefficients_omega_5p938_h5_h6_h7_v1/20260824T074945_7878c8a09d92/`.

## Current Scientific Decision

For practical single-axis second-order estimation, `h=4..8` is now the
supported working regime when its finite-amplitude `C4` correction is retained;
it is not a universal flat `even/h^2` plateau. The doubled block count behaves
as a sampling refinement, and the three-point `Q+C4*h^2` model shows no
resolved still-higher curvature inside this range. Remaining uncertainty is
sampling-dominated for `Q_z,xx` and `Q_z,zz` sine. For `Q_z,yy` sine, the
dominant qualification is the resolved `Q`--`C4` extrapolation dependence,
which more blocks cannot remove. Fits extending to `h>=10` remain
large-amplitude references rather than definitions of this regime.

These results support the candidate regime but do not establish a general
small-amplitude perturbative window for every response coefficient. Pointwise
intervals quantify sampling for the specified fit; polynomial-order and
amplitude-window spread remain separate extrapolation uncertainties.

Cross-input `xy`, `xz`, and `yz` coefficients have deliberately not been
calculated. No further production simulation is currently authorized.

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
