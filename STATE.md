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
- Strength-identifiability diagnostics fit block-level raw contrasts with
  `h+h^3` or `h^2+h^4` terms, retain joint real/imaginary covariance, compare
  adjacent normalized responses, and keep target and non-target harmonics.
- Whole-block bootstrap inference resamples every crossed strength, condition,
  component, real/imaginary coordinate, and observation prefix jointly. It
  reports signal identification separately from higher-order adequacy and
  fails closed when a leading denominator is unresolved.
- Within-block observation diagnostics use cycle prefixes and non-overlapping
  windows without treating cycles as replication. They estimate variance
  scaling across blocks and preserve the distinction between point-estimate
  instability, window-position variation, and sampling uncertainty.
- Frequency reconnaissance uses a response-independent octave grid, shared
  blocks, a single across-frequency bootstrap family, and observation windows
  that record both minimum cycles and physical duration.
- Held-out variance-reduction validation treats the second-harmonic unforced
  coefficient as a retained bias diagnostic rather than automatically adding
  it to the response estimator. DC baseline subtraction remains a distinct
  two-source estimation problem.
- Analytic synthetic-signal verification of Fourier signs, factors, complex
  susceptibility recovery, and direction reconstruction. Current test result:
  `91 passed`.

## Not Yet Confirmed

- Formal acceptance tolerances for residual spinup and forced-transient bias.
- Use of paired `+h`, `-h`, and unforced trajectories as the formal estimator,
  including the uncertainty estimator appropriate to deterministic chaotic
  divergence and within-trajectory dependence.
- Frequency domain and sampling scheme, forcing directions, strength range,
  phase resolution, cycle counts, and independent sample counts.
- Operational criteria for a usable zero-strength limit, response
  distinguishability, convergence, multiplicity across frequencies/tensor
  components, and evidence strength for final conclusions.
- Persistent experiment configuration, raw/derived result schemas, and runtime
  strategy for formal experiments.

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

The next blocker is minimal forcing-direction reconnaissance. The recommended
fixed exploratory design uses the six full-rank directions
`ex, ey, ez, (ex+ey)/sqrt(2), (ex+ez)/sqrt(2), (ey+ez)/sqrt(2)` at the
response-independent low/mid/high frequencies `0.5, 2, 8`, with the same 32
blocks, six strengths, working A2 estimator, and numerical profile. Existing x
trajectories can be reused. The projected new cost is 5,760 integrations, about
73 minutes and 202 MiB of raw summaries. This is an important resource choice
and is not yet authorized. DC continues to require unforced mean subtraction;
its common high-precision `mu0` baseline should be designed only after the
frequency/direction family is known. The simultaneous 95% and 20% criteria
remain provisional design criteria, not the final scientific meaning of
negligible higher order.

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

An unforced-spinup refinement used 384 blocks for each of two independent
proposal families, divided into six 64-block batches, with nested endpoints at
physical times `10, 20, 40, 80, 160`. At `T=40`, within-proposal drift toward
`T=160` and cross-proposal sensitivity were comparable to observed same-time
between-batch variation. Full-ensemble standardized proposal sensitivity was
`0.0039`; state means, lobe balance, and normalized vector-field balance were
also mutually consistent. This supports `T_spinup=40` as a conservative
working candidate at the diagnostic resolution of this study. The `T=160`
endpoint is only an internal reference, the six-batch distributions are
descriptive, and solver convergence was not tested by this experiment. The
reproducible output is under
`outputs/exploratory/unforced_spinup_refinement/20260814T054012_fbf00b2df957/`.

A forced-transient late-window refinement used the same 48 blocks at discard
times `40, 80, 160, 240, 320`, with non-overlapping windows after time `80`.
Each window contained 24 cycles and 24 phases (`75.398` physical time) under the
provisional protocol `omega=2`, `h=0.5`, x-direction. Parity-forbidden odd DC
retained same-direction onset memory at discard `40` and `80`, then fluctuated
around zero from `160` onward. Across target, parity-forbidden, and allowed
higher harmonics, late-window drift was comparable to its paired sensitivity
standard error and showed no continuing monotone decay. This supports
`forced discard=160` as a working candidate for subsequent numerical checks,
not as a cross-protocol convergence conclusion or residual-bias bound. The
reproducible output is under
`outputs/exploratory/forced_transient_late_windows/20260814T060949_7b459a19b61b/`.

A numerical-convergence pilot used 32 blocks under the same provisional
protocol with discard `160`. A 32-point phase grid differed from 64 points by
about `1.1e-4` to `1.6e-4` RMS on target contrasts and passed a shifted-subgrid
check; 16 points produced differences of roughly 3--21% of the reference mean.
For 32 versus 64 cycles, target paired-drift/SE ratios were `0.45`, `1.25`, and
`0.68`; baseline versus tighter solver ratios were `0.89`, `0.53`, and `1.33`.
This supports 32 phase points clearly and selects 64 cycles plus the baseline
solver conservatively. The 64-cycle target contrasts were themselves only
`0.60--0.85` cross-block SE from zero, so observation-window and solver choices
are working parameters rather than formal bias bounds. Reproducible output is
under
`outputs/exploratory/numerical_convergence_provisional/20260814T062105_afb486be58b6/`.

A crossed-block strength pilot used `B=32` and strengths
`0.25, 0.5, 1, 2, 4` under provisional `omega=2`, x-direction forcing. It did
not establish an identification window. First-order x/y signal became visible
at strong forcing, but fitted cubic/linear contribution diagnostics could not
exclude material target-harmonic contamination. Symmetry-allowed
second-harmonic and DC z components remained unstable relative to block
variation; rough square-root projections were commonly 150--250 blocks for a
ratio-three scale. For x-direction forcing, Lorenz equivariance makes the
first-order z component and the even/DC x/y components exact population null
controls. Their finite-sample excursions, including about 3.3 SE for even-DC
x/y at `h=2`, demonstrate that marginal ratio thresholds are not acceptable as
formal family-wide tests. Non-target harmonics stayed below about 1.92 SE, but
that does not bound higher-order terms at target harmonics. The run took about
180 seconds and is reproducible under
`outputs/exploratory/strength_identifiability_provisional/20260814T070309_1c4629df883e/`.

The confirmed strength refinement used 256 crossed blocks, strengths
`0.25, 0.5, 1, 2, 3, 4`, and 9,999 whole-block bootstrap draws. Its joint family
contained 210 scalar coordinates and had a simultaneous 95% critical value of
`3.456595`. First-order odd fundamental signal was identifiable at
`h=0.5,1,2,3,4`, but higher-/leading-order upper ratios at prefix maxima
`h=1,2,3,4` were `13.027, 1.765, 1.149, 0.794`, all above 20%. The allowed
second-harmonic and DC z components were never identifiable, so their adequacy
ratios failed closed. All exact symmetry-null controls retained zero. The run
took 1,474 seconds and produced 53 MB of raw summaries under
`outputs/exploratory/strength_identifiability_refinement/20260814T080709_12e8192abb9f/`.
The bootstrap uses a fixed observed standard error rather than a fully
studentized resampled standard error; it constrains the fitted first correction
but not all still-higher powers.

Reanalysis of the same 256-block artifact found family-wide variance scaling
close to `Var proportional to 1/L` over 1--64 cycles: median exponent `1.006`
(10--90% `0.942--1.039`) and median L=64 correlation inflation `0.975`
(10--90% `0.837--1.310`). Median family SE RMS fell from `0.1913` at 8 cycles
to `0.0681` at 64. However, first-order point higher-/leading ratios varied
substantially across 16- and 32-cycle windows before reaching `0.140` (h=2)
and `0.165` (h=4) at 64 cycles. A staged projection gives simultaneous upper
ratios of about `0.673/0.435` at L=256, `0.357/0.291` at L=1024, and
`0.236/0.226` at L=4096. Direct 20% projections require roughly
9,600--12,400 cycles and 34--44 hours, well outside the observed scaling range.
The reproducible reanalysis is under
`outputs/exploratory/within_block_observation_efficiency/20260814T085830_fae3c8131769/`.

Minimal frequency reconnaissance used the fixed grid
`omega=0.5,1,2,4,8`, the same 32 blocks, and respectively
`64,64,64,128,255` cycles so every point had at least 64 cycles and about 200
physical time. One 1,050-coordinate whole-block family had simultaneous 95%
critical value `3.80817`. At every frequency, symmetry-allowed second harmonic
and DC remained unidentified and no strength passed 20% adequacy; `omega=2`
second-order S/N was near the middle of the grid. First-order signal was visible
at strong forcing at every frequency, no structural null control excluded zero,
and non-target signal/joint-bound stayed below `0.948`. A post-hoc comparison
suggested that omitting the theoretically zero unforced n=2 coefficient could
reduce radial variance by roughly 64--75%, but this is a held-out hypothesis,
not accepted evidence. The 19.3-minute reproducible run is under
`outputs/exploratory/frequency_reconnaissance/20260814T132507_9e8a11beb461/`.

A pre-specified held-out study used 32 new, disjoint blocks on the same five
frequencies. For all symmetry-allowed second-harmonic z coordinates, the
variance ratio of `A2` to the previous unforced-subtracted estimator ranged
from `0.225` to `0.530` with median `0.323`; simultaneous upper bounds were
below one for 28 of 30 frequency-strength cells, with no point reversal in the
two inconclusive cells. The separately calibrated unforced-n=2 diagnostic had
no exclusion from zero. Neither estimator identified second harmonic or passed
20% adequacy at `B=32`. For DC, paired/equal-B-independent variance ratios had
median `0.926`, all intervals included one, and correlations with the unforced
baseline were unresolved. A separate baseline at 10% of forced-estimator
variance was projected to require 318--874 blocks per frequency, so separate
frequency baselines are not justified. The four 95% diagnostic families were
calibrated separately, not jointly. The reproducible held-out artifact is under
`outputs/exploratory/variance_reduction_heldout/20260814T141349_2694329f4136/`.

No-new-trajectory decision analysis found A2 variance scaling near `1/L` across
the available windows, but conditional identification projections were highly
unstable and no current point higher-/leading ratio was below 20%. A four- to
eight-fold x-only observation extension could improve raw identification but
cannot resolve strength-curvature adequacy or unobserved tensor sectors. The
six-direction set above is the smallest full-rank effective-quadratic design;
it has no quadratic residual degrees of freedom. A later balanced nine-direction
sum/difference design would add residual checks, but its current full-grid cost
is not justified before the reconnaissance.

One diagnostic audit used `sigma=10`, `rho=28`, `beta=8/3`, `omega=2`, forcing
direction `(1,0,0)`, strength `0.5`, 12 seeds, and up to 128 cycles. Standard
errors decreased with longer cycle averages, but target components were not
reliably separated from background fluctuations. This result is useful only as
a runtime/noise warning; its parameters, estimator, and numerical values are
not accepted project definitions or formal scientific evidence.

## Working Constraints

- `reference/`, Git history, and historical experiments are consulted only to
  answer a concrete information gap and do not define the current project.
- Exploration and formal evidence remain explicitly separated.
- Project body, durable collaboration state, and generated experiment data stay
  in their respective repository partitions.
- Existing modifications under `reference/matlab_ssm/` predate current work and
  are not part of active changes.
- The usable environment is currently
  `/home/feng/miniforge3/envs/ml/bin/python`; the default `python` lacks NumPy
  and SciPy, and dependency versions are not locked.

## Context Policy

The main context retains confirmed definitions, current capability, blockers,
constraints, delegated conclusions, and project-level decisions. Work whose
local detail substantially exceeds its durable project impact should run in a
delegated context when available. Delegated results return as conclusions,
supporting evidence, assumptions or limitations, project impact, and unresolved
questions; detailed logs and intermediate analysis remain in that context or
in reproducible project artifacts.
