# Lorenz Sinusoidal Response Project

## Confirmed goal

Use numerical experiments on the sinusoidally forced Lorenz-63 system to
compute interpretable and traceable first- and second-order response tensors.

Results used in scientific conclusions must:

- establish whether the response is distinguishable from chaotic background
  fluctuations;
- support conclusions across a frequency design rather than a few special
  frequencies;
- characterize dependence on forcing strength;
- report uncertainty and evidence strength appropriate to each conclusion;
- preserve numerical tensor values and their provenance, not only figures.

Exploratory runs may be used for debugging, scale estimation, and formal-design
choices, but they are not evidence for final conclusions.

The Lorenz-63 regime is

```text
sigma = 10, rho = 28, beta = 8/3.
```

The observable is the full state `g(X) = (x, y, z)`, so the susceptibility
output index refers to those three coordinates.

## Response definition and conventions

The final primary response tensors are frequency-domain susceptibilities for
monochromatic forcing. Phase-resolved responses are retained as underlying
estimates and diagnostics, not used as the final tensor definition.

For forcing direction `a`, signed strength `h`, angular frequency `omega`, and
phase `theta = omega*t + phase`, the dynamics are

```text
dX/dt = F(X) + h*a*sin(theta).
```

Let `m(h, a, theta)` be the phase-conditioned ensemble mean and let `m0` be
the unforced mean evaluated on the same phase grid. The Fourier-series
convention is

```text
x_hat[n] = (1 / 2*pi) integral_0^(2*pi) x(theta) exp(-i*n*theta) dtheta.
```

The corresponding discrete estimator is the unweighted mean over a uniform
phase grid. With the forcing convention above,

```text
f_hat[+1] = h*a/(2i),    f_hat[-1] = -h*a/(2i).
```

The response expansion uses the Volterra convention without a `1/2!` factor:

```text
delta m = G1[f] + G2[f,f] + O(||f||^3).
```

Consequently, the second-order susceptibility used here is the quadratic
coefficient kernel. With respect to a scalar strength, that coefficient is one
half of the second derivative at zero. Every formula and stored numerical
result uses this convention.

Define paired phase contrasts

```text
odd(h, a, theta)  = [m(+h, a, theta) - m(-h, a, theta)] / 2
even(h, a, theta) = [m(+h, a, theta) + m(-h, a, theta)] / 2 - m0(theta).
```

and finite-strength phase representations

```text
l_h(a, theta) = odd(h, a, theta) / h
q_h(a, theta) = even(h, a, theta) / h^2.
```

Their Fourier coefficients give the directional frequency responses

```text
chi1_ij(omega) a_j =  2i * l_hat_i[1]
chi2_ijk(2*omega; omega, omega) a_j a_k = -4 * q_hat_i[2]
Qdc_ijk(omega) a_j a_k = q_hat_i[0].
```

The factors `2i` and `-4` follow from the Fourier coefficient of a sine input;
they are part of the tensor definition rather than plotting conventions.

The primary first-order object is the complex `3 x 3` tensor
`chi1(omega)`. The primary second-order object is the complex `3 x 3 x 3`
second-harmonic tensor `chi2(2*omega; omega, omega)`. Directional forcing
identifies only its contraction with `a_j a_k`, so the reconstructed numerical
object is its effective symmetric part in the two input indices for this
monochromatic, same-frequency experiment. This is an identifiability statement
for the present protocol, not a claim that a general second-order
susceptibility at fixed frequency arguments is independently symmetric in its
input indices.

`Qdc(omega)` is a separate real-valued rectification tensor: the coefficient
of `h^2` in the output DC component. In terms of a general ordered-frequency
susceptibility, the experiment observes the input-index symmetric part of

```text
[chi2(0; omega, -omega) + chi2(0; -omega, omega)] / 4.
```

It does not separately identify those two ordered-frequency quantities, so the
project reports `Qdc` without calling it the complete DC susceptibility.

All three reported quantities are zero-strength limits of their
finite-strength estimators. Existence and usable convergence of those limits
are empirical questions for the strength study, not assumptions built into
the estimator.

The phase-resolved directional responses are retained with the frequency-domain
tensors as underlying estimates and diagnostics. Under a stationary periodic
response to a sinusoidal input, the first-order signal is expected at the
fundamental harmonic and the second-order signal at DC and the second harmonic.
Energy elsewhere is retained as a diagnostic of finite-strength or higher-order
contamination, sampling error, incomplete transient removal, or implementation
error.

The current scope does not include the full two-frequency surface
`chi2(omega1 + omega2; omega1, omega2)`. Identifying that object would require
a later, explicitly approved extension to multi-tone forcing.

## Ensemble and replication definition

The physical/SRB measure `mu0` of the unforced Lorenz system defines the
initial-state ensemble. For a forcing protocol `(h, omega, a)`, initial-state
blocks drawn from this ensemble evolve separately after forcing is applied.
After the forced transient is discarded, the target estimand is the
phase-conditioned expectation `m_(h,omega,a)(theta)` under the asymptotic
statistical state of the forced system.

An initial-state block is the sampling replication unit. A seed is only an
identifier used to generate and reproduce a block; distinct seed labels do not
by themselves establish statistical independence. The block-generation
procedure must justify independence or preserve any dependence in the
uncertainty analysis.

Sampling uncertainty and numerical or transient bias are separate:

- sampling uncertainty is estimated from variation across blocks;
- unforced spinup, forced-transient discard, integration accuracy, and related
  numerical approximations are assessed through separate convergence checks.

## Working experimental assumptions

The following assumptions have not yet been confirmed as the complete formal
experimental design:

- Cycles within one trajectory may reduce noise but are not counted as
  independent replicates.
- The `+h`, `-h`, and unforced conditions use paired initial-state blocks and
  identical sampling grids so uncertainty can be estimated from within-block
  contrasts.
- Cycle summaries are retained within each block. Conditions are contrasted
  inside a block, and uncertainty is summarized across independent blocks.
- Multiple positive strengths are required. Effective tensors are reported by
  strength, and the approach to the zero-strength limit must be checked rather
  than assumed.
- Cross-strength analysis fits unnormalized odd/even Fourier contrasts with
  their allowed odd/even powers. This separates the low-strength asymptotic
  range from the identification window where signal also exceeds sampling
  noise.
- Forcing directions must make both the linear direction matrix and the
  effective monochromatic quadratic design matrix full rank. More than the
  minimum number of directions may be used to expose reconstruction error.

Frequency selection, formal sample sizes, strength ranges, convergence rules,
and inferential thresholds remain undefined until pilot measurements establish
the relevant time, noise, and response scales.
