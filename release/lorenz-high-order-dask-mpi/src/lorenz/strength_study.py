"""Shared sampling and data ownership for crossed strength studies.

This module owns the numerical composition that is common to strength,
frequency, and variance-reduction experiments.  Experiment modules own their
artifact contracts and scientific decision families; they consume the stable
block-level Fourier summaries defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core import simulate_phase_samples, simulate_phase_and_dense
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from .parallel import map_tasks
from .response import phase_fourier


STATE_NAMES = ("x", "y", "z")
CONDITION_NAMES = ("positive", "negative", "unforced")


@dataclass(frozen=True)
class CycleFourierSampling:
    """Phase-grid sampling definition for one forcing frequency."""

    omega: float
    phase: float
    discard_time: float
    n_cycle: int
    n_phase: int
    harmonics: np.ndarray


@dataclass(frozen=True)
class StrengthStudyData:
    """Cycle Fourier summaries for a crossed strength experiment.

    Forced arrays have axes ``block, strength, state, cycle, harmonic``.
    The unforced array has axes ``block, state, cycle, harmonic`` because one
    matched unforced trajectory per block is shared by every strength.
    """

    block_ids: tuple[int, ...]
    strengths: np.ndarray
    harmonics: np.ndarray
    omega: float
    n_phase: int
    positive_cycle_fourier: np.ndarray
    negative_cycle_fourier: np.ndarray
    unforced_cycle_fourier: np.ndarray
    raw_proposals: np.ndarray
    initial_states: np.ndarray
    child_spawn_keys: tuple[tuple[int, ...], ...]
    generation_metadata: dict


@dataclass(frozen=True)
class FrequencyStudyData:
    """Crossed block summaries keyed by forcing frequency."""

    block_ids: tuple[int, ...]
    strengths: np.ndarray
    harmonics: np.ndarray
    frequencies: tuple[float, ...]
    studies: dict[float, StrengthStudyData]
    frequency_runtime_seconds: dict[float, float]
    source: dict[float, dict]

    def __post_init__(self):
        frequencies = tuple(float(value) for value in self.frequencies)
        if len(set(frequencies)) != len(frequencies):
            raise ValueError("frequency container requires unique frequencies")
        if set(self.studies) != set(frequencies):
            raise ValueError("frequency container studies must match frequencies")
        for omega in frequencies:
            study = self.studies[omega]
            if study.block_ids != self.block_ids:
                raise ValueError("all frequency studies must use the same block IDs/order")
            if not np.array_equal(study.strengths, self.strengths):
                raise ValueError("all frequency studies must use the same strengths")
            if not np.array_equal(study.harmonics, self.harmonics):
                raise ValueError("all frequency studies must use the same harmonics")


@dataclass(frozen=True)
class StrengthFourierComponents:
    """Explicit parity components before choosing a baseline subtraction."""

    odd: np.ndarray
    forced_even: np.ndarray
    unforced: np.ndarray


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_strengths(strengths, *, minimum_count: int = 2) -> np.ndarray:
    strengths = np.asarray(strengths, dtype=float)
    if (
        strengths.ndim != 1
        or len(strengths) < minimum_count
        or not np.isfinite(strengths).all()
        or np.any(strengths <= 0)
        or np.any(np.diff(strengths) <= 0)
    ):
        raise ValueError(
            f"strengths must contain at least {minimum_count} finite, positive, "
            "strictly increasing values"
        )
    return strengths


def _checked_harmonics(harmonics, n_phase: int) -> np.ndarray:
    raw = np.asarray(harmonics)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or len(np.unique(raw)) != len(raw)
        or not {0, 1, 2}.issubset(set(int(value) for value in raw))
        or len(np.unique(np.mod(raw, n_phase))) != len(raw)
    ):
        raise ValueError("harmonics must be unique, alias-free integers including 0, 1, 2")
    if 2 * int(np.max(np.abs(raw))) >= n_phase:
        raise ValueError("configured harmonics must lie strictly below Nyquist")
    return raw.astype(int)


def validate_strength_sampling_config(config: dict) -> dict:
    """Validate the shared numerical definition of one strength study."""
    strengths = validate_strengths(config["strengths"])
    protocol = config["protocol"]
    omega = float(protocol["omega"])
    phase = float(protocol.get("phase", 0.0))
    direction = np.asarray(protocol["direction"], dtype=float)
    discard_time = float(config["discard_time"])
    n_cycle = _positive_integer(config["n_cycle"], "n_cycle")
    n_phase = _positive_integer(config["n_phase"], "n_phase")
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("direction must be a finite nonzero vector with shape (3,)")
    if not np.isfinite(discard_time) or discard_time < 0:
        raise ValueError("discard_time must be finite and nonnegative")
    return {
        "strengths": strengths,
        "harmonics": _checked_harmonics(config["harmonics"], n_phase),
        "omega": omega,
        "phase": phase,
        "direction": direction,
        "discard_time": discard_time,
        "n_cycle": n_cycle,
        "n_phase": n_phase,
    }


def minimum_duration_cycle_count(
    omega: float, minimum_cycles: int, minimum_physical_time: float
) -> int:
    """Smallest cycle count satisfying both duration floors."""
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    minimum_cycles = _positive_integer(minimum_cycles, "minimum_cycles")
    if not np.isfinite(minimum_physical_time) or minimum_physical_time <= 0:
        raise ValueError("minimum_physical_time must be finite and positive")
    time_cycles = math.ceil(minimum_physical_time * omega / (2 * np.pi))
    return max(minimum_cycles, time_cycles)


def frequency_key(omega: float) -> str:
    """Return the stable artifact key used for a forcing frequency."""
    if not np.isfinite(omega):
        raise ValueError("omega must be finite")
    return format(float(omega), ".12g").replace("-", "m").replace(".", "p")


def validate_frequency_sampling_config(config: dict) -> dict:
    """Validate sampling fields shared by crossed-frequency experiments."""
    frequencies = tuple(float(value) for value in config["frequencies"])
    if (
        len(frequencies) < 2
        or any(not np.isfinite(value) or value <= 0 for value in frequencies)
        or tuple(sorted(frequencies)) != frequencies
        or len(set(frequencies)) != len(frequencies)
    ):
        raise ValueError("frequencies must be distinct, positive, and increasing")
    strengths = validate_strengths(config["strengths"], minimum_count=3)
    block_count = _positive_integer(config["block_count"], "block_count")
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    minimum_cycles = _positive_integer(
        config["observation_rule"]["minimum_cycles"], "minimum_cycles"
    )
    minimum_time = float(config["observation_rule"]["minimum_physical_time"])
    n_phase = _positive_integer(config["n_phase"], "n_phase")
    cycles = {
        omega: minimum_duration_cycle_count(omega, minimum_cycles, minimum_time)
        for omega in frequencies
    }
    return {
        "frequencies": frequencies,
        "strengths": strengths,
        "harmonics": _checked_harmonics(config["harmonics"], n_phase),
        "block_count": block_count,
        "cycles": cycles,
    }


def single_frequency_strength_config(config: dict, omega: float, n_cycle: int) -> dict:
    """Project a crossed-frequency config onto one shared strength study."""
    return {
        "strengths": config["strengths"],
        "harmonics": config["harmonics"],
        "discard_time": config["discard_time"],
        "n_cycle": n_cycle,
        "n_phase": config["n_phase"],
        "block_count": config["block_count"],
        "workers": config.get("workers", 1),
        "lorenz": config["lorenz"],
        "solver": config["solver"],
        "initial_ensemble": config["initial_ensemble"],
        "protocol": {
            "omega": omega,
            "direction": config["protocol"]["direction"],
            "phase": float(config["protocol"].get("phase", 0.0)),
        },
    }


def _integrate_cycle_fourier(arguments):
    if len(arguments) == 4:
        initial_state, forcing, sampling, numerical_config = arguments
        forcing_phases = None
    else:
        initial_state, forcing, forcing_phases, sampling, numerical_config = arguments
    phase_options = {} if forcing_phases is None else {"forcing_phases": forcing_phases}
    samples = simulate_phase_samples(
        initial_state,
        forcing,
        sampling.omega,
        sampling.phase,
        sampling.discard_time,
        sampling.n_cycle,
        sampling.n_phase,
        numerical_config,
        **phase_options,
    )
    return phase_fourier(samples.values, sampling.harmonics, samples.phase_offset)


def _integrate_cycle_fourier_block(arguments):
    """Run every condition for one block inside one execution task."""
    if len(arguments) == 4:
        initial_state, forcing_vectors, sampling, numerical_config = arguments
        forcing_phases = [None] * len(forcing_vectors)
    else:
        initial_state, forcing_vectors, forcing_phases, sampling, numerical_config = arguments
    return tuple(
        _integrate_cycle_fourier(
            (initial_state, forcing, sampling, numerical_config)
            if phases is None else
            (initial_state, forcing, phases, sampling, numerical_config)
        )
        for forcing, phases in zip(forcing_vectors, forcing_phases)
    )


def integrate_cycle_fourier_conditions(
    block_ids,
    initial_states,
    forcing_vectors,
    sampling: CycleFourierSampling,
    numerical_config: dict,
    *,
    forcing_phases=None,
    workers: int = 1,
) -> np.ndarray:
    """Integrate a block-by-condition design into cycle Fourier summaries.

    The result axes are ``block, condition, state, cycle, harmonic``.  A caller
    can therefore integrate one zero-forcing condition once per frequency and
    reuse it across every strength or forcing direction at that frequency.
    """
    block_ids = tuple(block_ids)
    initial_states = np.asarray(initial_states, dtype=float)
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
    workers = _positive_integer(workers, "workers")
    if len(block_ids) < 1 or len(set(block_ids)) != len(block_ids):
        raise ValueError("block_ids must be nonempty and unique")
    if initial_states.shape != (len(block_ids), 3) or not np.isfinite(initial_states).all():
        raise ValueError("initial_states must have finite axes block,state")
    if (
        forcing_vectors.ndim != 2
        or forcing_vectors.shape[0] < 1
        or forcing_vectors.shape[1] != 3
        or not np.isfinite(forcing_vectors).all()
    ):
        raise ValueError("forcing_vectors must have finite axes condition,state")
    if not isinstance(sampling, CycleFourierSampling):
        raise TypeError("sampling must be a CycleFourierSampling")
    if forcing_phases is None:
        condition_phases = [None] * len(forcing_vectors)
    else:
        condition_phases = np.asarray(forcing_phases, dtype=float)
        if condition_phases.shape != forcing_vectors.shape or not np.isfinite(condition_phases).all():
            raise ValueError("forcing_phases must have finite axes condition,state")
    tasks = [
        (initial_states[block_index], forcing_vectors, sampling, numerical_config)
        if forcing_phases is None else
        (initial_states[block_index], forcing_vectors, condition_phases, sampling, numerical_config)
        for block_index in range(len(block_ids))
    ]
    integrated_blocks = map_tasks(
        _integrate_cycle_fourier_block, tasks, workers=workers
    )
    integrated = [
        condition
        for block_conditions in integrated_blocks
        for condition in block_conditions
    ]
    return np.asarray(integrated).reshape(
        len(block_ids),
        len(forcing_vectors),
        3,
        sampling.n_cycle,
        len(sampling.harmonics),
    )


def generate_strength_study(config: dict) -> StrengthStudyData:
    """Generate one crossed strength study with one unforced run per block."""
    checked = validate_strength_sampling_config(config)
    block_count = _positive_integer(config["block_count"], "block_count")
    workers = _positive_integer(config.get("workers", 1), "workers")
    initial = config["initial_ensemble"]
    block_id_start = int(initial["block_id_start"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    proposal_cfg = initial["proposal"]
    proposal = SymmetricXYUniformProposal(
        x_half_width=proposal_cfg["x_half_width"],
        y_half_width=proposal_cfg["y_half_width"],
        z_bounds=tuple(proposal_cfg["z_bounds"]),
    )
    numerical_config = {"lorenz": dict(config["lorenz"]), "solver": dict(config["solver"])}
    blocks = generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        proposal,
        float(initial["spinup_time"]),
        numerical_config,
    )
    forcing_vectors = [np.zeros(3)]
    forcing_vectors.extend(
        sign * strength * checked["direction"]
        for strength in checked["strengths"]
        for sign in (1.0, -1.0)
    )
    sampling = CycleFourierSampling(
        omega=checked["omega"],
        phase=checked["phase"],
        discard_time=checked["discard_time"],
        n_cycle=checked["n_cycle"],
        n_phase=checked["n_phase"],
        harmonics=checked["harmonics"],
    )
    all_conditions = integrate_cycle_fourier_conditions(
        blocks.block_ids,
        blocks.final_states,
        forcing_vectors,
        sampling,
        numerical_config,
        workers=workers,
    )
    unforced = all_conditions[:, 0]
    paired = all_conditions[:, 1:].reshape(
        block_count,
        len(checked["strengths"]),
        2,
        3,
        checked["n_cycle"],
        len(checked["harmonics"]),
    )
    return StrengthStudyData(
        block_ids=blocks.block_ids,
        strengths=checked["strengths"],
        harmonics=checked["harmonics"],
        omega=checked["omega"],
        n_phase=checked["n_phase"],
        positive_cycle_fourier=paired[:, :, 0],
        negative_cycle_fourier=paired[:, :, 1],
        unforced_cycle_fourier=unforced,
        raw_proposals=blocks.raw_proposals,
        initial_states=blocks.final_states,
        child_spawn_keys=blocks.child_spawn_keys,
        generation_metadata={
            "root_entropy": blocks.root_entropy,
            "root_spawn_key": blocks.root_spawn_key,
            "bit_generator": blocks.bit_generator,
            "block_ids": blocks.block_ids,
            "child_spawn_keys": blocks.child_spawn_keys,
            "proposal": {
                "x_half_width": proposal.x_half_width,
                "y_half_width": proposal.y_half_width,
                "z_bounds": proposal.z_bounds,
            },
            "spinup_time": blocks.spinup_time,
        },
    )


def strength_fourier_components(data: StrengthStudyData) -> StrengthFourierComponents:
    """Return cycle-averaged odd, forced-even, and unforced components."""
    positive = np.asarray(data.positive_cycle_fourier).mean(axis=-2)
    negative = np.asarray(data.negative_cycle_fourier).mean(axis=-2)
    unforced = np.asarray(data.unforced_cycle_fourier).mean(axis=-2)
    expected = (len(data.block_ids), len(data.strengths), 3, len(data.harmonics))
    if positive.shape != expected or negative.shape != expected:
        raise ValueError("forced Fourier summaries have invalid axes")
    if unforced.shape != (len(data.block_ids), 3, len(data.harmonics)):
        raise ValueError("unforced Fourier summaries have invalid axes")
    return StrengthFourierComponents(
        odd=(positive - negative) / 2,
        forced_even=(positive + negative) / 2,
        unforced=unforced,
    )


def strength_target_contrasts(data: StrengthStudyData) -> dict[str, np.ndarray]:
    """Return existing baseline-subtracted target contrasts."""
    components = strength_fourier_components(data)
    index = {int(value): i for i, value in enumerate(data.harmonics)}
    even = components.forced_even - components.unforced[:, None]
    return {
        "odd_fundamental": components.odd[..., index[1]],
        "even_second_harmonic": even[..., index[2]],
        "even_dc": even[..., index[0]].real,
    }


def second_harmonic_estimators(data: StrengthStudyData) -> dict[str, np.ndarray]:
    """Return A2, U2, and the legacy A2-U2 second-harmonic estimators."""
    components = strength_fourier_components(data)
    index = {int(value): i for i, value in enumerate(data.harmonics)}[2]
    a2 = components.forced_even[..., index]
    u2 = components.unforced[..., index]
    return {
        "current_E2": a2 - u2[:, None],
        "alternative_A2": a2,
        "unforced_U2_diagnostic": u2,
    }


def dc_components(data: StrengthStudyData) -> dict[str, np.ndarray]:
    """Return forced-even A0, unforced U0, and required A0-U0 DC output."""
    components = strength_fourier_components(data)
    index = {int(value): i for i, value in enumerate(data.harmonics)}[0]
    a0 = components.forced_even[..., index].real
    u0 = components.unforced[..., index].real
    return {
        "forced_even_A0": a0,
        "unforced_U0": u0,
        "current_E0": a0 - u0[:, None],
    }


def harmonic_role(contrast: str, harmonic: int) -> str:
    """Classify a retained harmonic relative to parity and target roles."""
    if (contrast == "odd" and harmonic == 1) or (
        contrast == "even" and harmonic in (0, 2)
    ):
        return "target"
    if (harmonic % 2 == 1) == (contrast == "odd"):
        return "allowed_higher_harmonic"
    return "parity_forbidden_harmonic"


# --------------------------------------------------------------------------
# dense spectral summaries
# --------------------------------------------------------------------------

DENSE_AXES = ("state", "theta_bin")
PSD_AXES = ("state", "frequency_bin")


@dataclass(frozen=True)
class DenseTrajectorySummary:
    """Block-level dense spectral summaries for one trajectory.

    ``folded_cycle_mean`` has axes ``state, theta_bin``: the phase-conditioned
    mean folded onto a dense forcing-phase grid from fixed-dt samples.
    ``welch_psd`` has axes ``state, frequency_bin``: the one-sided Welch
    periodogram of the residual after removing the folded cycle mean.
    ``segment_count`` records how many Hann-windowed segments contributed.
    ``raw_values`` / ``raw_times`` are optionally retained dense samples.
    """

    folded_cycle_mean: np.ndarray
    welch_psd: np.ndarray
    segment_count: int
    raw_values: np.ndarray
    raw_times: np.ndarray
    dense_dt: float
    frequency_step: float
    nyquist: float
    maximum_frequency: float


def dense_trajectory_summary(
    dense_values,
    dense_times,
    omega: float,
    phase: float,
    n_theta_bins: int,
    welch_segment: int,
    max_psd_bins: int,
    raw_samples: int,
) -> DenseTrajectorySummary:
    """Summarize dense samples into folded mean, Welch PSD, and raw segment."""
    dense_values = np.asarray(dense_values, dtype=float)
    dense_times = np.asarray(dense_times, dtype=float)
    if dense_values.ndim != 2 or dense_values.shape[0] != 3:
        raise ValueError("dense values must have axes state,time")
    if dense_times.shape != (dense_values.shape[1],):
        raise ValueError("dense times must match the time axis")
    if len(dense_times) < 2 or np.any(np.diff(dense_times) <= 0):
        raise ValueError("dense times must be strictly increasing")
    n_theta_bins = int(n_theta_bins)
    welch_segment = int(welch_segment)
    max_psd_bins = int(max_psd_bins)
    if n_theta_bins < 32 or welch_segment < 16 or max_psd_bins < 8:
        raise ValueError("dense summary settings are too small")
    dt = float(np.median(np.diff(dense_times)))
    theta = np.mod(omega * dense_times + phase, 2 * np.pi)
    folded_mean = np.empty((3, n_theta_bins), dtype=float)
    for state in range(3):
        folded_mean[state] = np.bincount(
            np.floor(theta / (2 * np.pi) * n_theta_bins).astype(int),
            weights=dense_values[state],
            minlength=n_theta_bins,
        ) / np.maximum(
            np.bincount(
                np.floor(theta / (2 * np.pi) * n_theta_bins).astype(int),
                minlength=n_theta_bins,
            ),
            1,
        )
    residual = dense_values - folded_mean[
        :, np.floor(theta / (2 * np.pi) * n_theta_bins).astype(int)
    ]
    window = np.hanning(welch_segment)
    total_bins = min(max_psd_bins, welch_segment // 2)
    psd = np.zeros((3, total_bins), dtype=float)
    segment_count = 0
    offset = 0
    while offset + welch_segment <= residual.shape[1]:
        segment = residual[:, offset : offset + welch_segment]
        spectrum = np.fft.rfft(segment * window[None, :], axis=1)
        psd += np.abs(spectrum[:, :total_bins]) ** 2
        segment_count += 1
        offset += welch_segment
    if segment_count == 0:
        raise ValueError("dense window is shorter than one Welch segment")
    window_energy = float(np.sum(window**2))
    psd *= 2.0 / (welch_segment * dt * window_energy * segment_count)
    raw_values = np.asarray(dense_values[:, : int(raw_samples)], dtype=float)
    raw_times = np.asarray(dense_times[: int(raw_samples)], dtype=float)
    frequency_step = 2 * np.pi / (welch_segment * dt)
    return DenseTrajectorySummary(
        folded_cycle_mean=folded_mean,
        welch_psd=psd,
        segment_count=segment_count,
        raw_values=raw_values,
        raw_times=raw_times,
        dense_dt=dt,
        frequency_step=float(frequency_step),
        nyquist=float(np.pi / dt),
        maximum_frequency=float((total_bins - 1) * frequency_step),
    )


@dataclass(frozen=True)
class DenseBlockSpectrum:
    """Complex block-level physical-frequency spectrum of one trajectory.

    ``coefficients`` has axes ``state, frequency_bin``: the Hann-windowed
    complex DFT of the complete observation interval on the common
    fixed-dt grid, referenced to the absolute physical-time origin
    ``t = 0``,

        S_b(Omega_k) = sum_n w_n x(t_n) exp(-i Omega_k t_n) / sum_n w_n,

    with ``t_n = t0 + n*dt``, ``Omega_k = 2*pi*k / (N*dt)``, and one Hann
    window ``w`` over the complete retained interval.  This is the
    continuous display spectrum only: an arbitrary forcing frequency does
    not in general fall on an FFT bin, so scientific values at
    ``n*omega`` are the direct known-frequency phase/cycle coefficients,
    never the nearest bin of this grid.  The single full-interval window
    shares one common Fourier phase origin, so complex averaging across
    blocks is meaningful; ``S_b`` is stored, not ``abs(S_b)`` or
    ``abs(S_b)**2``, so later analysis can compute both
    ``abs(S_b(Omega))`` per block and ``abs(mean_b S_b(Omega))`` after
    complex averaging.  ``frequency_grid`` is the explicit angular-frequency
    grid (complete one-sided range up to ``pi/dt`` by default, or the
    requested physical ``maximum_omega``) and is identical for every
    trajectory of one frequency cell.
    ``segment_count`` is always 1 (one full-interval window); the field is
    retained for the chunk-file contract.  If a segmented implementation is
    ever reintroduced for memory reasons, every segment must first be
    rotated by ``exp(-i Omega_k t_start)`` to the same absolute-time
    reference before complex averaging.
    """

    coefficients: np.ndarray
    frequency_grid: np.ndarray
    segment_count: int
    dense_dt: float
    frequency_step: float
    nyquist: float
    maximum_frequency: float


def dense_block_spectrum(
    dense_values,
    dense_times,
    maximum_omega=None,
) -> DenseBlockSpectrum:
    """Compute the complex per-block display spectrum of the raw trajectory.

    The spectrum is the fluctuation spectrum of the trajectory: each
    state's time mean is removed before windowing (display/background
    object only; the harmonic estimator and DC response are unaffected),
    so the forcing response peak at ``omega`` and its harmonics is present
    but a nonzero time mean produces no Omega=0 peak.  ONE Hann window is
    applied over the complete observation interval and the FFT is taken on
    the common fixed-dt grid; there are no short segments.  The phase
    reference is the absolute physical-time origin shared by every block
    and condition.  No averaging across blocks happens here.

    By default the complete one-sided spectrum is returned, covering the
    physical range ``0 <= Omega <= pi/dt`` (the Nyquist range); the returned
    ``frequency_grid`` is authoritative.  An explicit PHYSICAL
    ``maximum_omega`` selects the bins of the computed grid up to that
    angular frequency, so the same physical range is requested regardless of
    observation length; no fixed bin count is used.
    """
    dense_values = np.asarray(dense_values, dtype=float)
    dense_times = np.asarray(dense_times, dtype=float)
    if dense_values.ndim != 2 or dense_values.shape[0] != 3:
        raise ValueError("dense values must have axes state,time")
    if dense_times.shape != (dense_values.shape[1],):
        raise ValueError("dense times must match the time axis")
    if len(dense_times) < 2 or np.any(np.diff(dense_times) <= 0):
        raise ValueError("dense times must be strictly increasing")
    if maximum_omega is not None:
        if not np.isfinite(maximum_omega) or float(maximum_omega) <= 0:
            raise ValueError("maximum_omega must be a finite positive frequency")
    dt = float(np.median(np.diff(dense_times)))
    sample_count = dense_values.shape[1]
    # Fluctuation spectrum (display/background object only): remove each
    # state's time mean before windowing so a nonzero time mean (e.g. z's
    # DC offset) cannot produce a large Omega=0 peak.  This does NOT touch
    # the DC response, Qdc, raw trajectories, or the harmonic estimator.
    dense_values = dense_values - dense_values.mean(axis=1, keepdims=True)
    # Periodic Hann over the complete interval: its DFT is exactly zero at
    # every bin |k| >= 2, so an on-bin sinusoid has no image-bin leakage.
    window = 0.5 - 0.5 * np.cos(
        2 * np.pi * (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    )
    windowed = dense_values * window[None, :]
    transform = np.fft.rfft(windowed, axis=1)
    normalization = float(np.sum(window))
    frequency_step = 2 * np.pi / (sample_count * dt)
    grid = frequency_step * np.arange(sample_count // 2 + 1, dtype=float)
    if maximum_omega is not None:
        keep = grid <= float(maximum_omega)
        transform = transform[:, keep]
        grid = grid[keep]
    # Rotate from the first-sample origin to the absolute origin t = 0.
    reference = np.exp(-1j * grid * float(dense_times[0]))
    coefficients = reference[None, :] * transform / normalization
    return DenseBlockSpectrum(
        coefficients=coefficients,
        frequency_grid=grid,
        segment_count=1,
        dense_dt=dt,
        frequency_step=float(frequency_step),
        nyquist=float(np.pi / dt),
        maximum_frequency=float(grid[-1]),
    )


def _integrate_phase_and_dense(arguments):
    if len(arguments) == 5:
        initial_state, forcing, sampling, dense_config, numerical_config = arguments
        forcing_phases = None
    else:
        initial_state, forcing, forcing_phases, sampling, dense_config, numerical_config = arguments
    phase_options = {} if forcing_phases is None else {"forcing_phases": forcing_phases}
    phase_samples, dense_values, dense_times = simulate_phase_and_dense(
        initial_state,
        forcing,
        sampling.omega,
        sampling.phase,
        sampling.discard_time,
        sampling.n_cycle,
        sampling.n_phase,
        dense_config["dt"],
        numerical_config,
        **phase_options,
    )
    fourier = phase_fourier(
        phase_samples.values, sampling.harmonics, phase_samples.phase_offset
    )
    summary = None
    if dense_config.get("compute_summary", True):
        summary = dense_trajectory_summary(
            dense_values,
            dense_times,
            sampling.omega,
            sampling.phase,
            dense_config["n_theta_bins"],
            dense_config["welch_segment"],
            dense_config["max_psd_bins"],
            dense_config.get("raw_segment_samples", 0),
        )
    spectrum = dense_block_spectrum(
        dense_values,
        dense_times,
        dense_config.get("spectrum_max_omega"),
    )
    if dense_config.get("retain_dense"):
        dense_retained = (dense_values, dense_times)
    else:
        dense_retained = None
    return fourier, summary, spectrum, phase_samples, dense_retained


def _integrate_phase_and_dense_block(arguments):
    """Run every dense-output condition for one block in condition order."""
    if len(arguments) == 5:
        initial_state, forcing_vectors, sampling, dense_config, numerical_config = arguments
        forcing_phases = [None] * len(forcing_vectors)
    else:
        (
            initial_state, forcing_vectors, forcing_phases, sampling,
            dense_config, numerical_config,
        ) = arguments
    return tuple(
        _integrate_phase_and_dense(
            (initial_state, forcing, sampling, dense_config, numerical_config)
            if phases is None else
            (initial_state, forcing, phases, sampling, dense_config, numerical_config)
        )
        for forcing, phases in zip(forcing_vectors, forcing_phases)
    )


def integrate_phase_and_dense_conditions(
    block_ids,
    initial_states,
    forcing_vectors,
    sampling: CycleFourierSampling,
    dense_config: dict,
    numerical_config: dict,
    *,
    forcing_phases=None,
    workers: int = 1,
    raw_segment_blocks: int = 0,
    retain_phase_samples: bool = False,
    retain_dense_blocks: int = 0,
    compute_summaries: bool = True,
):
    """Integrate once per block-by-condition cell into Fourier, dense, and
    spectral summaries.

    The Fourier result axes are ``block, condition, state, cycle, harmonic``
    exactly as :func:`integrate_cycle_fourier_conditions`.  Dense summaries
    are returned as a matching ``block, condition`` array of
    :class:`DenseTrajectorySummary`; complex block spectra as a matching
    array of :class:`DenseBlockSpectrum`.  Raw dense segments are retained
    only for the first ``raw_segment_blocks`` blocks.  When
    ``retain_phase_samples`` is true, the phase-grid samples with axes
    ``block, condition, state, cycle, phase`` are returned as well.  When
    ``retain_dense_blocks`` is positive, the full dense trajectories of the
    first that many blocks are returned with axes
    ``block, condition, state, time``.  Callers are responsible for the
    retention policy (these objects are large).  ``compute_summaries=False``
    skips dense trajectory summaries while preserving the returned object-array
    contract with ``None`` entries.
    """
    block_ids = tuple(block_ids)
    initial_states = np.asarray(initial_states, dtype=float)
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
    workers = _positive_integer(workers, "workers")
    raw_segment_blocks = int(raw_segment_blocks)
    retain_dense_blocks = int(retain_dense_blocks)
    if len(block_ids) < 1 or len(set(block_ids)) != len(block_ids):
        raise ValueError("block_ids must be nonempty and unique")
    if initial_states.shape != (len(block_ids), 3) or not np.isfinite(initial_states).all():
        raise ValueError("initial_states must have finite axes block,state")
    if (
        forcing_vectors.ndim != 2
        or forcing_vectors.shape[0] < 1
        or forcing_vectors.shape[1] != 3
        or not np.isfinite(forcing_vectors).all()
    ):
        raise ValueError("forcing_vectors must have finite axes condition,state")
    if not isinstance(sampling, CycleFourierSampling):
        raise TypeError("sampling must be a CycleFourierSampling")
    if forcing_phases is None:
        condition_phases = [None] * len(forcing_vectors)
    else:
        condition_phases = np.asarray(forcing_phases, dtype=float)
        if condition_phases.shape != forcing_vectors.shape or not np.isfinite(condition_phases).all():
            raise ValueError("forcing_phases must have finite axes condition,state")
    if raw_segment_blocks < 0 or raw_segment_blocks > len(block_ids):
        raise ValueError("raw_segment_blocks must lie within the block axis")
    if retain_dense_blocks < 0 or retain_dense_blocks > len(block_ids):
        raise ValueError("retain_dense_blocks must lie within the block axis")
    required = {"dt", "n_theta_bins", "welch_segment", "max_psd_bins"}
    if not required.issubset(dense_config):
        raise ValueError("dense config requires dt, n_theta_bins, welch_segment, max_psd_bins")
    tasks = [
        (
            initial_states[block_index],
            forcing_vectors,
            condition_phases,
            sampling,
            {
                **dense_config,
                "raw_segment_samples": (
                    int(dense_config.get("raw_segment_samples", 0))
                    if block_index < raw_segment_blocks
                    else 0
                ),
                "retain_dense": block_index < retain_dense_blocks,
                "compute_summary": compute_summaries,
            },
            numerical_config,
        )
        for block_index in range(len(block_ids))
    ]
    if forcing_phases is None:
        tasks = [
            (initial_state, vectors, sampling, dense_options, numerical_config)
            for initial_state, vectors, _, sampling, dense_options, numerical_config
            in tasks
        ]
    integrated_blocks = map_tasks(
        _integrate_phase_and_dense_block, tasks, workers=workers
    )
    integrated = [
        condition
        for block_conditions in integrated_blocks
        for condition in block_conditions
    ]
    fourier = np.asarray([item[0] for item in integrated]).reshape(
        len(block_ids),
        len(forcing_vectors),
        3,
        sampling.n_cycle,
        len(sampling.harmonics),
    )
    summary_flat = [item[1] for item in integrated]
    spectrum_flat = [item[2] for item in integrated]
    summaries = np.empty((len(block_ids), len(forcing_vectors)), dtype=object)
    spectra = np.empty((len(block_ids), len(forcing_vectors)), dtype=object)
    for block_index in range(len(block_ids)):
        for condition in range(len(forcing_vectors)):
            flat_index = block_index * len(forcing_vectors) + condition
            summaries[block_index, condition] = summary_flat[flat_index]
            spectra[block_index, condition] = spectrum_flat[flat_index]
    if retain_phase_samples:
        phase_values = np.asarray(
            [item[3].values for item in integrated]
        ).reshape(
            len(block_ids),
            len(forcing_vectors),
            3,
            sampling.n_cycle,
            sampling.n_phase,
        )
        phase_sample_times = integrated[0][3].sample_times
    else:
        phase_values = None
        phase_sample_times = None
    if retain_dense_blocks > 0:
        dense_values = np.asarray(
            [
                item[4][0]
                for item in integrated
                if item[4] is not None
            ]
        ).reshape(
            retain_dense_blocks,
            len(forcing_vectors),
            3,
            -1,
        )
        dense_times = next(item[4][1] for item in integrated if item[4] is not None)
    else:
        dense_values = None
        dense_times = None
    return fourier, summaries, spectra, phase_values, phase_sample_times, (
        (dense_values, dense_times) if dense_values is not None else None
    )
