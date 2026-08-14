"""Shared sampling and data ownership for crossed strength studies.

This module owns the numerical composition that is common to strength,
frequency, and variance-reduction experiments.  Experiment modules own their
artifact contracts and scientific decision families; they consume the stable
block-level Fourier summaries defined here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math

import numpy as np

from .core import simulate_phase_samples
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from .response import phase_fourier


STATE_NAMES = ("x", "y", "z")


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


def _checked_strengths(strengths, *, minimum_count: int = 2) -> np.ndarray:
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
    strengths = _checked_strengths(config["strengths"])
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
    strengths = _checked_strengths(config["strengths"], minimum_count=3)
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
    initial_state, forcing, sampling, numerical_config = arguments
    samples = simulate_phase_samples(
        initial_state,
        forcing,
        sampling.omega,
        sampling.phase,
        sampling.discard_time,
        sampling.n_cycle,
        sampling.n_phase,
        numerical_config,
    )
    return phase_fourier(samples.values, sampling.harmonics, samples.phase_offset)


def integrate_cycle_fourier_conditions(
    block_ids,
    initial_states,
    forcing_vectors,
    sampling: CycleFourierSampling,
    numerical_config: dict,
    *,
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
    tasks = [
        (initial_states[block_index], forcing, sampling, numerical_config)
        for block_index in range(len(block_ids))
        for forcing in forcing_vectors
    ]
    if workers == 1:
        integrated = list(map(_integrate_cycle_fourier, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            integrated = list(executor.map(_integrate_cycle_fourier, tasks))
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
