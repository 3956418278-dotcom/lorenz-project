# Retained-data contract and storage scaling

This document defines what the integration/persistence layer retains for the
production experiment, what it derives, and how much storage each object
class costs. It exists so that aggregation is a later analysis operation
instead of an integration-side discard, and so the production retention
level is chosen from storage costs rather than hard-coded.

Implementation: `src/lorenz/retention.py` (policy, condition identity,
chunked layout), `src/lorenz/strength_study.py` (`DenseBlockSpectrum`,
dense integration returns), `src/lorenz/x_response_pilot.py` (chunked
generation, persistence, loaders), `src/lorenz/storage_scale.py`
(calculator). Plotting consumers: `src/lorenz/response_plots.py`.

## 1. What the previous pipeline discarded

- **Phase-grid samples** `(block, condition, cycle, phase, state)`: computed
  in `simulate_phase_samples` / `simulate_phase_and_dense` and reduced to
  per-cycle Fourier coefficients immediately (`_integrate_cycle_fourier`,
  `_integrate_phase_and_dense`). Never reached persistence.
- **Per-cycle complex Fourier coefficients** for the general population:
  `persist_x_response` kept full per-cycle arrays only for a hard-coded
  small subset (`retention.per_cycle_subset`: omega=8, 4 of 64 blocks); all
  other cells were collapsed to cycle means plus real/imag cycle variances.
- **Per-block complex physical-frequency spectra S_b(Omega)**: never
  computed at all. The dense summary stored only the real-valued Welch PSD
  of the *residual* after folded-cycle-mean removal (per block) and the
  real folded cycle mean. `abs(mean_b S_b(Omega))` after complex averaging
  was therefore impossible from the artifacts.
- **Dense physical-time trajectories**: reduced inside
  `dense_trajectory_summary` to folded mean + Welch PSD + small raw
  segments; raw segments were retained for only `raw_segment_blocks = 2`
  blocks x 1200 samples per frequency.
- **Condition identity**: omega was encoded in array-key strings, the
  condition order in manifest strings, and `load_x_response_cells`
  hard-coded `n_phase = 32`; direction, phase, discard, spinup, and solver
  settings existed only in `config_snapshot.json`, not as structured
  per-condition records aligned with the condition axis.

The derived objects (cycle means, variances, folded means, Welch PSDs,
bootstrap family, c1-c4 fits, significance labels, validity
classifications) were always allowed to exist; the problem was that they
were the only retained form of the data.

## 2. Retained objects (the raw/block-level contract)

The block is the replication unit everywhere. Conditions are indexed by a
condition axis: index 0 is the unforced reference; indices `1 + 2*s` and
`2 + 2*s` are the `+h`/`-h` pair of strength `s` (per forcing direction in
a crossed design).

### 2.1 Condition identity

File `condition_metadata.json` (JSON) plus per-frequency scalar/grid arrays
in `raw_x_response_summaries.npz`:

| object | dtype | axes |
|---|---|---|
| `observable_labels` | JSON | `("x", "y", "z")` |
| `block_ids` | uint32 | `(B,)` |
| `condition_signs` | int8 | `(n_condition,)` — +1/-1 collinear with the reference direction, 0 unforced or non-collinear |
| `condition_strengths` | float64 | `(n_condition,)` — signed strength; canonical positive for non-collinear directions |
| `condition_directions` | float64 | `(n_condition, 3)` — unit forcing vector per condition |
| `omega`, `forcing_phase`, `discard_time`, `n_cycle`, `n_phase`, `phase_offset`, `dense_dt` | float64/int64 scalars | per frequency cell |
| `phase_sample_times` | float64 | `(n_cycle, n_phase)` — the actual estimator grid, stored once per cell (identical across blocks/conditions) |
| `dense_sample_times` | float64 | `(n_dense,)` — the actual dense grid, stored once per cell |
| `lorenz`, `solver`, `initial_ensemble`, `protocol`, `retention_policy` | JSON | settings copied from the configuration snapshot |

This is sufficient to pair the same block across `+h`/`-h`, strengths,
frequencies, directions, and the unforced reference without reconstructing
identity from filenames.

### 2.2 Block-level chunk files

Directory `block_level/omega_{key}/omega_{key}_blocks_{start:06d}_{end:06d}.npz`,
`chunk_blocks` blocks per file (configurable), written chunk-by-chunk
during integration:

| object | file key | dtype | axes |
|---|---|---|---|
| per-cycle Fourier coefficients | `cycle_fourier` | complex128 | `(block, condition, cycle, state, harmonic)` |
| estimator phase-grid samples | `phase_values` | float64 | `(block, condition, cycle, phase, state)` |
| complex physical-frequency spectra | `spectrum` | complex128 | `(block, condition, state, frequency_bin)` |
| spectrum frequency grid | `spectrum_frequency_grid` | float64 | `(frequency_bin,)` — explicit, identical across chunks |
| spectrum segment counts | `spectrum_segment_counts` | int64 | `(block, condition)` |
| dense trajectories | `dense_values` | float64 | `(block, condition, state, time)` |
| dense sample times | `dense_times` | float64 | `(time,)` |
| `block_ids` | uint32 | `(chunk,)` | chunk membership |

`S_b(Omega_k) = sum_n w_n x_b(t_n) exp(-i Omega_k t_n) / sum_n w_n` — ONE
periodic Hann window over the complete observation interval, one FFT on
the common fixed-dt grid, referenced to the absolute physical-time origin
`t = 0` shared by every block and condition (no short segments). By
default the complete one-sided range `0 <= Omega <= pi/dt` is stored (or
an explicit physical `maximum_omega` cap via
`dense.spectrum_max_omega`); the stored frequency grid is authoritative.
This is the continuous **display** spectrum: an arbitrary forcing frequency
is not in general on an FFT bin, so scientific values at `n*omega` come
from the direct known-frequency phase/cycle estimator, never from the
nearest bin.
The spectrum is of the **raw** trajectory (the forcing peak is present),
is **complex**, and is never averaged across blocks before persistence.
Both `abs(S_b(Omega))` per block and `abs(mean_b S_b(Omega))` after complex
averaging are computable later. If a segmented implementation is ever
reintroduced for memory reasons, every segment must first be rotated by
`exp(-i Omega_k t_start)` to the same absolute-time reference before
complex averaging.

### 2.3 Retention policy (explicit configuration)

```json
"retention": {
  "per_cycle_fourier_blocks": "all",       // "all" | count | null (default "all")
  "phase_samples_blocks": "all",           // "all" | count | null (default "all")
  "dense_trajectory_blocks": "all",        // REQUIRED explicit decision (default null)
  "block_spectra": true,                   // default true
  "chunk_blocks": 8
}
```

Per-cycle Fourier and phase samples default to "all" because the estimator
contract requires them at minimum; dense trajectory retention must be set
explicitly after consulting the storage report, and is never silently
hard-coded to a small block count. The legacy `retention.per_cycle_subset`
key of the executed pilot configs is superseded by this surface; old
artifacts remain readable.

### 2.4 Derived summaries (reproducible, not authoritative)

`raw_x_response_summaries.npz` and `derived_diagnostics.json` keep the
derived convenience objects: cycle means
`(block, strength, state, harmonic)` complex128, cycle variances
`(block, strength, state, harmonic, [real, imaginary])` float64, folded
cycle means `(block, condition, state, theta_bin)` float64, Welch PSDs
`(block, condition, state, frequency_bin)` float64, segment counts, raw
segments, the bootstrap family, fitted c1/c2/c3/c4, significance labels,
and validity classifications.

Derivation chain from the raw contract: cycle mean = mean over the cycle
axis of `cycle_fourier`; cycle variance = var(ddof=1) over the cycle axis
of the real/imag parts; folded cycle mean = phase-conditioned mean of
`dense_values` on the theta grid; Welch PSD = one-sided Hann-windowed
segmented periodogram of the dense residual after removing the folded cycle
mean; all bootstrap/fit/label objects derive from the cycle means. The
figure-generation script asserts the first chain link on the executed
artifact (cycle means equal the means of the retained per-cycle subset).

## 3. Storage scaling

All formulas are implemented in `src/lorenz/storage_scale.py`
(`python -m lorenz.storage_scale --config <config.json>` or explicit
flags; `--compression` applies an optional factor — the executed dense
pilot stored at an observed compressed/uncompressed ratio of 0.953).

Per frequency omega, with `B` blocks, `D` directions, `S` strengths,
`C = 1 + 2*S*D` conditions, observation rule
`n_cycle = max(min_cycles, ceil(min_time * omega / 2*pi))`,
`n_dense = floor(n_cycle * 2*pi/omega / dt) + 1`, `P = n_phase`,
`H = harmonic count`, `F = spectrum bins`, `T = theta bins`:

| object | size (bytes) |
|---|---|
| dense trajectories | `B_dense * C * 3 * n_dense * 8` |
| phase samples | `B_phase * C * n_cycle * P * 3 * 8` |
| per-cycle Fourier | `B_fourier * C * n_cycle * 3 * H * 16` |
| block spectra | `B * C * 3 * F * 16` |
| derived summaries | `B*C*3*H*16 + B*C*3*H*2*8 + B*C*3*T*8 + B*C*3*F*8 + B*C*8 + raw segments` |
| identity and grids | `n_cycle*P*8 + n_dense*8 + F*8 + C*5*8` |

The dominant term at the executed pilot's parameter scale is the phase
sample grid (`B * C * n_cycle * P` grows with omega because the
minimum-physical-time floor raises the cycle count); see the calculator
output for the exact per-frequency breakdown.

Reference numbers for the executed x-direction design (B=64, D=1, 12
frequencies, 4 strengths, min 64 cycles / 400 time, dt=0.05, P=32, H=6,
F=256, uncompressed): dense trajectories 0 (not retained), phase samples
2.01 GiB, per-cycle Fourier 773 MiB, block spectra 81 MiB, derived
summaries 91 MiB, grids 2 MiB — 2.94 GiB total. With full dense retention
at dt=0.05 the trajectories add 1.39 GiB — 4.33 GiB total. The production
experiment chooses the retention level from this report; nothing is thrown
away silently.
