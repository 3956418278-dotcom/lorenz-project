# Current State

## Confirmed

- Goal: use numerical experiments on sinusoidally forced Lorenz-63 dynamics to
  compute interpretable and traceable first- and second-order response tensors.
- Lorenz parameters are `sigma=10`, `rho=28`, `beta=8/3`; the observable is the
  full state `(x, y, z)`.
- Formal results must address distinguishability from chaotic fluctuations,
  generality across forcing frequency, dependence on forcing strength,
  conclusion-matched uncertainty, and preservation of numerical tensors with
  provenance.
- Scope: monochromatic forcing. A complete two-frequency second-order
  susceptibility surface and multi-tone forcing are outside the current scope.
- Primary tensors: complex `chi1(omega)` and complex monochromatic
  second-harmonic `chi2(2*omega; omega, omega)`.
- Separate second-order output: real rectification coefficient tensor
  `Qdc(omega)`. It is not described as a complete ordered-frequency DC
  susceptibility.
- Phase-resolved directional responses are underlying estimates and
  diagnostics. Non-target harmonics are retained to diagnose finite-strength
  and higher-order contamination, sampling error, and incomplete transients.
- Fourier uses the `exp(-i*n*theta)` coefficient convention; forcing is
  `h*a*sin(theta)`; the second-order Volterra term has no `1/2!` factor. The
  exact normalization formulas are in `PROJECT.md`.
- For the monochromatic same-frequency protocol, direction contractions
  identify only the effective input-index symmetric part of the quadratic
  tensor. This does not assert a general input-index symmetry of second-order
  susceptibility.
- The unforced physical/SRB measure `mu0` defines and generates the
  initial-state ensemble. After applying a fixed forcing protocol and discarding
  its transient, the estimand is the forced system's asymptotic
  phase-conditioned expectation.
- The initial-state block is the sampling replication unit. A seed only
  identifies and reproduces a block; different seed labels do not establish
  independence without a justified generation procedure.
- Across-block variation estimates sampling uncertainty. Spinup,
  forced-transient, integration, and other numerical bias are evaluated by
  separate convergence checks.
- For the current provisional strength family, identification uses whole-block
  joint bootstrap intervals with simultaneous 95% coverage. Asymptotic
  adequacy requires the simultaneous upper bound on the first fitted
  higher-order contribution relative to the resolved leading contribution to
  be at most 20%. This criterion has not yet been extended to the eventual
  frequency/direction family.

## Current Capability

- Lorenz integration, vector sinusoidal forcing, and sampling from an explicit
  initial state on a forcing-phase grid using physical discard time. Cycle
  samples, actual
  sample times, and forcing phase offset are retained.
- Reproducible initial-state block generation from a configurable
  symmetry-respecting proposal using block-ID-addressed `SeedSequence` children
  and `PCG64DXSM`; raw proposals and post-spinup states remain distinct with
  in-memory provenance.
- Paired odd/even phase contrasts and uniform-grid Fourier coefficients.
- Phase-aware Fourier extraction: nonzero forcing phase is included in the
  coefficient basis rather than silently rotating the reported susceptibility.
- Unnormalized odd-fundamental, even-second-harmonic, and even-DC Fourier
  contrasts suitable for later cross-strength inference.
- Direct extraction of finite-strength directional `chi1`, second-harmonic
  `chi2`, and `Qdc` using the confirmed normalization.
- Reconstruction of a linear tensor and the effective monochromatic quadratic
  tensor from suitable direction sets.
- Block-preserving finite-strength normalization, per-block tensor
  reconstruction, joint realification of `chi1`/`chi2`/`Qdc`, and separate
  unbiased block sample covariance and covariance of the block mean.
- An explicit block/cycle adapter requires `(block, ..., cycle, phase)`, carries
  immutable block IDs through estimation, and prevents an observable axis from
  being silently treated as replication.
- Exploratory unforced-spinup diagnostics support replicated batches with
  explicit batch and block identity, nested endpoints, alternate proposals,
  and reproducible raw/derived outputs.
- Forced-transient diagnostics preserve block and cycle identity, compare
  nested physical-time discard windows through raw Fourier contrasts, and
  distinguish target, parity-forbidden, and allowed higher harmonics. Their
  paired sensitivity uncertainty remains separate from production covariance.
- Numerical-convergence diagnostics separately compare nested observation
  prefixes, nested and shifted phase subgrids, and solver profiles using
  block-level Fourier contrasts rather than pathwise chaotic trajectories.
- Shared strength-study primitives integrate configured crossed strengths,
  preserve per-cycle Fourier summaries, fit block-level raw contrasts with
  `h+h^3` or `h^2+h^4` terms, and keep target and non-target harmonics.
- The x-direction dense response pilot composes those primitives with: uniform
  per-frequency cells, block-level retention (cycle-mean Fourier plus
  across-cycle variances; per-cycle kept only for a small representative
  subset), dense fixed-dt summaries per block (folded phase-conditioned mean on
  256 theta bins, Welch PSD of the residual, small raw segments), and a
  per-frequency detection/adequacy family (identification per strength,
  fitted c1/c2/c3/c4 coefficients, direct harmonics n=3,4,5, nested-prefix
  adequacy, four-state interpretation).
- A retained-data contract (`RETENTION.md`, `src/lorenz/retention.py`) for the
  production run: structured condition identity (condition table, sampling
  grids, solver/spinup settings) plus chunk-by-chunk block-level files holding
  per-cycle complex Fourier coefficients `(block, condition, cycle, state,
  harmonic)`, estimator phase-grid samples `(block, condition, cycle, phase,
  state)`, complex per-block physical-frequency spectra `S_b(Omega)` with the
  explicit frequency grid `(block, condition, state, frequency_bin)`, and
  optionally complete dense trajectories `(block, condition, state, time)`.
  Retention is explicit in configuration (`retention.*_blocks`:
  `"all"`/count/null; per-cycle Fourier and phase samples default to "all";
  dense trajectories must be set explicitly). Derived summaries remain
  reproducible from the raw contract. The storage-scaling calculator
  (`src/lorenz/storage_scale.py`) reports each object class as a function of
  B, directions, strengths, observation rule, dt, phase resolution, harmonic
  count, and spectrum bin count (reference: the executed x-pilot design at
  full retention is 4.33 GiB uncompressed, of which phase samples 2.01 GiB,
  dense trajectories 1.39 GiB, per-cycle Fourier 773 MiB, spectra 81 MiB).
- Direct block-level visualizations (`src/lorenz/response_plots.py`):
  per-block spectra with coherent ensemble mean and its uncertainty, raw
  time-domain trajectories, measured frequency-response curves, normalized
  strength dependence, and the complete retained harmonic set with per-cycle
  distributions. Lines/markers are estimates, bands are uncertainty, clouds
  are empirical block distributions, and reference lines are labeled.
- Whole-block bootstrap inference resamples every crossed strength, condition,
  component, real/imaginary coordinate, and observation prefix jointly. It
  reports signal identification separately from higher-order adequacy and
  fails closed when a leading denominator is unresolved.
- Direction-design and reconnaissance capabilities reconstruct block-level
  tensors across crossed forcing directions and frequencies, diagnose Lorenz
  parity sectors, and preserve the unforced second harmonic separately from
  the working second-harmonic estimator. The current reconnaissance runner has
  not been executed.
- Analytic synthetic-signal verification of Fourier signs, factors, complex
  susceptibility recovery, and direction reconstruction. The seven targeted
  high-order probe checks pass in the current numerical environment.

- The production high-order probe supports six forcing directions, arbitrary
  strength counts, retained block-level complex spectra, signed cos/sin
  Hotelling tests, multiplicity-aware BH q-values, resumable chunked execution,
  and immutable extension artifacts. Extension runs merge new strengths onto a
  canonical condition axis and require an exact redundant-unforced check against
  their base artifact before inference.
- Figure-A-style diagnostics are generated as one three-panel `x | y | z`
  figure per direction and strength. They show the coherent odd/even spectra,
  whole-block-resampled ensemble noise, single-block chaotic-background bands,
  harmonic guides, and exact known-frequency responses from `condition_means`.

## High-Order Signal-Discovery Results

Two completed `B=256` probes cover six forcing directions, outputs `(x,y,z)`,
harmonics `n=0..5`, and strengths `{2,4,8,10,12,14,16}` at `omega=6.7` and
`omega=5.938`. Detection uses the joint signed cos/sin Hotelling statistic with
BH `q<=0.05` over a separate 630-cell non-DC family at each frequency. These are
large-amplitude signal-discovery probes; especially `h>=8` is not interpreted
as a perturbative tensor estimate.

- At `omega=6.7`, harmonics `n=1,2,3,4,5` are all detected somewhere in the
  family, with respectively `91,59,45,15,22` detected cells. Both `n=4` and
  `n=5` are present across multiple cells by `h=12`; isolated lower-strength
  detections and nonmonotone counts mean that no sharp monotone onset is
  claimed.
- At `omega=5.938`, harmonics `n=1,2,3,4,5` are all detected somewhere, with
  respectively `92,49,46,3,8` detected cells. `n=4` is first established at
  `h=14`; `n=5` has detections from `h=10`, and both are detected at `h=14` and
  `h=16`.
- The strongest `n=4` and `n=5` responses remain below the median amplitude of
  a single unforced block at their harmonic frequencies: response/background
  ratios are about `0.395/0.247` at `omega=6.7` and `0.329/0.194` at
  `omega=5.938`. Their detection comes from coherent paired ensemble response,
  not from exceeding the fluctuations of an individual chaotic block.
- The redundant unforced extension condition matches its immutable base
  exactly in all eight chunks for `condition_means`, retained cycle Fourier
  coefficients, dense complex spectra, and spectrum segment counts at both
  frequencies.

Canonical artifacts:

- `outputs/production/high_order_probe_omega_6p7_h10_h12_h14_h16_extension_v1/20260822T061155_75ca70131a4b/`
- `outputs/production/high_order_probe_omega_5p938_h10_h12_h14_h16_extension_v1/20260822T061155_b1e645f07e98/`

Each artifact contains the 630-cell numerical/significance summaries and core
figures. Its post-run `figure_A_by_direction_strength/` view contains 42 PNGs
(six directions by seven strengths); these presentation files were generated
after the immutable simulation manifest.

Established next issue: the probes show that fourth- and fifth-harmonic signals
can be resolved at sufficiently large forcing, but they do not establish a
small-amplitude scaling regime or perturbative coefficients. Any tensor study
must separately test amplitude scaling and numerical/statistical adequacy at
smaller strengths.

## Not Yet Confirmed

- Formal acceptance tolerances for residual spinup and forced-transient bias.
- Use of paired `+h`, `-h`, and unforced trajectories as the formal estimator,
  including the uncertainty estimator appropriate to deterministic chaotic
  divergence and within-trajectory dependence.
- Frequency domain and sampling scheme, forcing directions, strength range,
  phase resolution, cycle counts, and independent sample counts for the final
  perturbative tensor experiment. The completed probe settings are confirmed
  only for large-amplitude signal discovery.
- Operational criteria for a usable zero-strength limit, response
  distinguishability, convergence, multiplicity across frequencies/tensor
  components, and evidence strength for final conclusions.
- The high-order probe has persistent configuration and raw/derived schemas;
  their suitability and runtime strategy for the final perturbative tensor
  experiment are not yet confirmed.

## Current Blocker

The susceptibility, ensemble, replication-unit, block-generation mechanism,
and block-level covariance boundaries are implemented. Replicated-batch
exploration supports `T_spinup = 40` physical Lorenz time as the conservative
working candidate for generating initial-state blocks. This is a parameter for
the next convergence study, not a claim that finite-time endpoints exactly
follow `mu0`.

For one provisional method-validation protocol (`omega=2`, `h=0.5`, forcing in
the x direction), exploration supports the following conservative working
candidates: `forced discard = 160` physical time, `64` observed cycles (about
`201` physical time), `32` phase points per cycle, and `DOP853` with
`rtol=1e-9`, `atol=1e-11`. These do not establish convergence over the eventual
frequency, strength, or direction design.

The `B=256` provisional refinement found no identification window under the
confirmed simultaneous 95% / 20% rule. First-order signal is identifiable at
several strengths, but its simultaneous higher-order adequacy bound fails.
Symmetry-allowed second-harmonic and DC components remain unidentified.

The proposed 256-cycle extension at `omega=2` is deferred. That single
frequency/direction shows weak second-order identifiability under the current
estimator, but it does not distinguish generally high estimator variance from
a frequency- or direction-specific weak response.

Minimal reconnaissance found weak second-order identifiability across the
entire fixed octave grid, with `omega=2` near the middle rather than an isolated
poor point. Held-out blocks then confirmed that the fixed estimator
`A2=(C(+h,2)+C(-h,2))/2` broadly reduces second-harmonic variance while
preserving the estimand when the stationary unforced n=2 coefficient is zero.
`A2` is now the working second-harmonic estimator; the unforced coefficient is
always retained and tested as a finite-transient/numerical-bias diagnostic.

This variance reduction alone did not identify second harmonic at `B=32`.
Reanalysis shows that observation length still reduces variance near `1/L`,
but x-direction forcing probes only two of five symmetry-allowed linear entries
and one of eight symmetry-allowed entries in each effective quadratic tensor.
An x-only observation extension therefore cannot determine the full-tensor
sampling budget.

The executed x-direction dense response pilot: 12 response-independent
half-octave frequencies `0.5`-`22.6`, x-forcing, strengths
`{0.25, 0.5, 1, 2}` at B=64 (observation `max(64 cycles, 400 time)`; anchors
`{0.5, 1, 4, 8}` extended to six strengths), with block-level Fourier and
dense spectral retention. Artifacts:
`outputs/exploratory/x_response_dense_v1/`, `x_response_c2_extras_v1/`,
`x_response_merged_v1/` (corrected merged diagnostics; the first merged run
was superseded after a coefficient-summary filtering bug was found and fixed).

Established (stored whole-block simultaneous 95% bounds, not point +/- SE):

- chi1_xx(omega) is detected at {0.5, 1.0, 1.4, 2.0, 4.0, 5.6, 8.0, 16.0,
  22.6} and not detected at {0.7, 2.8, 11.3}. Paired same-block comparisons
  resolve an elevated band at `5.6-8` over `2.8-4` and the falloff above `8`;
  the low-omega plateau versus `2.8-4`, `5.6` versus `8`, and the high tail
  flatness are not resolved, so no dip/peak labels are claimed.
- The fitted c1 is stable across the 4-6 strength fits within ~1-1.5 SE (the
  3-strength fit is not a valid c1 estimate); it is described as the leading
  coefficient of the truncated strength-series fit.
- Second order and DC are not detected anywhere. Stored upper bounds:
  `chi2z <= 0.004-0.33` across the grid (tightest `0.034` at `omega=0.5`,
  `0.035` at `8`); `Qdc z <= 0.008-0.066`. The fitted c2 decays toward zero as
  the fit range grows: c2 is consistent with zero, so only the bounds are
  scientific content.
- The fitted c3/c4 and the direct harmonics n=3,4,5 are not detected at any
  frequency or strength. Negligibility (provisional 20% rule) is established
  nowhere: no `(omega, h)` cell is "bounded small". An exploratory sub-family
  bootstrap shows the global family (critical 4.16 versus 3.68-3.87 for
  natural sub-families) is not the cause of second-order non-detection;
  `|c2|/SE <= 2.0` everywhere.

Unresolved: the validity region (identified AND higher-order-negligible) does
not exist under the provisional rule; second order is bounded but not
measured; the DC common high-precision `mu0` baseline remains deferred; the
Welch background is validated only as a descriptive scale.

The six-direction high-order signal-discovery extension is now complete at two
frequencies. It establishes visibility of `n=1..5` at large amplitude but does
not resolve the earlier perturbative-identification problem. The next decision
is an amplitude-scaling design that tests where the resolved signals cease to
be distinguishable and whether a defensible small-amplitude regime exists.

The completed high-order artifacts retain per-block continuous complex spectra,
so the full noise-versus-response comparison is now available for those probe
settings. Retention for a future perturbative production design must still be
chosen explicitly from the storage-scaling report rather than inherited
automatically.

## Delegated Design Conclusions

- Recommended estimand: the phase-conditioned expectation of the selected
  observable under the periodically forced statistical steady state. Finite
  initialization and transient-removal bias require convergence checks outside
  ordinary sampling confidence intervals.
- Recommended independent unit: an initial-state block sampled independently
  from the target unforced regime. Conditions, strengths, and preferably
  directions are crossed within a block. A seed identifies a block but is not
  itself the statistical object, and chaotic divergence means pairing may or
  may not reduce variance.
- Retain `block x condition x cycle x phase/observable`, or at least per-cycle
  Fourier summaries. Average and contrast within a block, then estimate joint
  real/imaginary covariance across blocks. Reconstruct tensors within each
  block before cross-block uncertainty summaries.
- Fit raw Fourier contrasts across strength rather than susceptibility values
  already divided by `h` or `h^2`. Odd contrasts use odd powers and even
  contrasts use even powers. This avoids unnecessary heteroskedastic scaling
  and supports explicit estimates of higher-order contamination.
- Distinguish the asymptotic range, where the local response expansion is
  adequate, from the identification window, where response also exceeds
  chaotic sampling noise. Failure of these ranges to overlap is a valid
  non-identification result, especially for second order whose normalized noise
  grows approximately as `1/h^2`.
- Non-target harmonics diagnose phase alignment, aliasing, transients, and some
  higher orders, but cannot replace cross-strength checks because higher-order
  terms also contribute at the target harmonics.
- The current integrator retains cycle samples, but `phase_mean` discards them
  too early for formal estimation. Formal sampling also needs transient,
  phase-resolution, unforced-baseline, validation, and provenance contracts.
- Independent proposal draws followed by a common deterministic spinup preserve
  independence by construction. Finite spinup changes representativeness of
  `mu0`, not that independence; accept spinup through ensemble-level doubling
  and alternate-proposal sensitivity rather than trajectory decorrelation.
- Forced discard is measured in physical Lorenz time. Compare nested discard
  windows with equal post-discard observation lengths; fixed discarded-cycle
  counts would impose frequency-dependent physical transients.
- Observation windows require both a minimum physical duration and a minimum
  completed-cycle count. Phase grids use nested refinement plus shifted-subgrid
  checks, and solver refinement compares block-level Fourier statistics rather
  than pathwise chaotic trajectories.

## Existing Exploratory Evidence

- `T_spinup=40` is the conservative working candidate; this is not a proof of
  exact sampling from `mu0`.
  Artifact: `outputs/exploratory/unforced_spinup_refinement/20260814T054012_fbf00b2df957/`.
- `forced discard=160` is the working candidate for the provisional protocol,
  not a cross-protocol residual-bias bound.
  Artifact: `outputs/exploratory/forced_transient_late_windows/20260814T060949_7b459a19b61b/`.
- The numerical working profile is 32 phase points, at least 64 cycles and about
  200 physical time, and `DOP853` with `rtol=1e-9`, `atol=1e-11`. Observation
  length and solver choices remain provisional because response noise limited
  their convergence comparisons.
  Artifact: `outputs/exploratory/numerical_convergence_provisional/20260814T062105_afb486be58b6/`.
- The 256-block strength refinement found no identification window under the
  provisional simultaneous 95% / 20% rule. Its fixed-SE max bootstrap is not a
  fully studentized bootstrap and the two-power model does not bound all higher
  powers.
  Artifact: `outputs/exploratory/strength_identifiability_refinement/20260814T080709_12e8192abb9f/`.
- Within-block variance was close to `1/L` through 64 cycles, but long-length
  projections and point curvature were unstable; x-only lengthening is
  deferred.
  Artifact: `outputs/exploratory/within_block_observation_efficiency/20260814T085830_fae3c8131769/`.
- On the fixed octave grid, second harmonic and DC were weak at every frequency
  and `omega=2` was not an isolated poor point.
  Artifact: `outputs/exploratory/frequency_reconnaissance/20260814T132507_9e8a11beb461/`.
- Held-out blocks support working estimator
  `A2=(C(+h,2)+C(-h,2))/2`, with the unforced n=2 coefficient retained as a bias
  diagnostic. This reduces second-harmonic variance broadly but did not itself
  establish identification. DC still needs a common uncertain `mu0` baseline;
  separate per-frequency baselines are not justified.
  Artifact: `outputs/exploratory/variance_reduction_heldout/20260814T141349_2694329f4136/`.
- A supervisor-discussion figure set renders the above stored evidence for the
  three questions (distinguishability, numerical accuracy vs noise, higher-order
  significance). It is a view of existing artifacts, not new evidence; the one
  illustrative panel (phase-resolved reconstruction) is labeled as such.
  Artifact: `outputs/figures/supervisor_evidence_20260815/` (`make_figures.py`,
  `fig{1..4}*.png/pdf`, `notes.md`).
- x-direction dense response pilot artifacts: dense grid
  `outputs/exploratory/x_response_dense_v1/20260815T120543_3e1cb15bc6f3/`,
  c2-extras `outputs/exploratory/x_response_c2_extras_v1/20260815T122332_1a3ae1d04b23/`,
  corrected merged analysis
  `outputs/exploratory/x_response_merged_v1/20260815T133813_merged/`.
  Result figures and the canonical result summary:
  `outputs/figures/x_response_results/` (`make_figures.py`, `result_extract.py`,
  `result_checks.py`, `verify_results.py`, `fig{1..4}*.png/pdf`, `notes.md`).
- Block-level figure set built from the same executed artifacts with the new
  production plotting functions: `outputs/figures/x_response_block_level/`
  (`make_figures.py`, `fig1_block_spectra_limited`, `fig1_supplement_complex_2blocks`,
  `fig2_raw_time_domain`, `fig3_frequency_response`, `fig4_strength_dependence`,
  `fig5_harmonic_content`, `notes.md`). fig1 is reconstruction-limited (the
  executed artifacts lack the complex per-block spectrum S_b(Omega)); the
  complete figure requires the next production run's retained spectrum.

## Working Constraints

- `reference/`, Git history, and historical experiments are consulted only to
  answer a concrete information gap and do not define the current project.
- Exploration and formal evidence remain explicitly separated.
- Project body, durable collaboration state, and generated experiment data stay
  in their respective repository partitions.
- Existing modifications under `reference/matlab_ssm/` predate current work and
  are not part of active changes.
- The usable Linux numerical environment for the probe is currently
  `/home/feng/miniforge3/envs/ody/bin/python` (NumPy 2.4.3, SciPy 1.15.2).
  It lacks pytest; the base environment has pytest but no NumPy, while the
  `ml` environment has an incompatible SciPy build. Targeted probe checks were
  therefore invoked directly in `ody`; dependency versions are not locked.
- This retention/visualization work ran on the Windows machine's Python
  3.14.3 (`E:\Downloads\py\python.exe`, numpy 2.4.3, scipy 1.17.1, matplotlib
  3.10.8, pytest installed locally; run tests with `PYTHONPATH=src python -m
  pytest tests/`). One pre-existing Windows-only test failure remains:
  `test_artifacts.py::test_json_ready_and_atomic_write_preserve_project_json_bytes`
  expects POSIX `Path` stringification ("relative/path"); on Windows
  `str(Path(...))` yields backslashes. The remaining suite passes (117
  tests).

## Context Policy

The main context retains confirmed definitions, current capability, blockers,
constraints, delegated conclusions, and project-level decisions. Work whose
local detail substantially exceeds its durable project impact should run in a
delegated context when available. Delegated results return as conclusions,
supporting evidence, assumptions or limitations, project impact, and unresolved
questions; detailed logs and intermediate analysis remain in that context or
in reproducible project artifacts.
