"""Uniform phase-grid sampling and direct known-harmonic Fourier projection.

The production experiment does not need a broadband spectrum.  The forcing
frequency ``omega`` and the harmonics of interest are fixed by the protocol,
so each cycle is sampled on a uniform forcing-phase grid and projected directly
onto ``exp(-1j*n*theta)`` for those harmonics.  This is a targeted DFT (often
called lock-in or phase-coherent extraction), not a different spectral
quantity from an FFT.  An FFT would compute many unused frequency bins and
would obscure the explicit forcing-phase origin used by the paired design.

The stored complex coefficient at harmonic ``n`` is ``Z_n``.  For ``n>0`` and
real ``A*cos(n*theta)+B*sin(n*theta)``, ``Z_n=A/2-1j*B/2``.  Any later
``cos``/``sin`` columns therefore mean ``Re(Z_n)`` and ``-Im(Z_n)`` unless a
function explicitly says it has already converted to a susceptibility.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from ._fast_dop853 import try_fourier_block
from .core import simulate_phase_samples
from .parallel import iter_task_results, map_tasks
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


def _unpack_block_arguments(arguments):
    if len(arguments) == 4:
        initial_state, forcing_vectors, sampling, numerical_config = arguments
        forcing_phases = None
    else:
        initial_state, forcing_vectors, raw_phases, sampling, numerical_config = arguments
        if raw_phases is None or all(item is None for item in raw_phases):
            forcing_phases = None
        elif any(item is None for item in raw_phases):
            return initial_state, forcing_vectors, raw_phases, sampling, numerical_config, False
        else:
            forcing_phases = raw_phases
    return initial_state, forcing_vectors, forcing_phases, sampling, numerical_config, True


def _integrate_cycle_fourier_block(arguments):
    """Run every condition for one block inside one execution task."""
    (
        initial_state,
        forcing_vectors,
        forcing_phases,
        sampling,
        numerical_config,
        uniform_phases,
    ) = _unpack_block_arguments(arguments)
    if uniform_phases and isinstance(sampling, CycleFourierSampling):
        fast = try_fourier_block(
            initial_state,
            forcing_vectors,
            forcing_phases,
            sampling.omega,
            sampling.phase,
            sampling.discard_time,
            sampling.n_cycle,
            sampling.n_phase,
            sampling.harmonics,
            numerical_config,
            average_cycles=False,
        )
        if fast is not None:
            return tuple(fast[index] for index in range(fast.shape[0]))
    phases_iter = [None] * len(forcing_vectors) if forcing_phases is None else forcing_phases
    return tuple(
        _integrate_cycle_fourier(
            (initial_state, forcing, sampling, numerical_config)
            if phases is None else
            (initial_state, forcing, phases, sampling, numerical_config)
        )
        for forcing, phases in zip(forcing_vectors, phases_iter)
    )


def _condition_tasks(
    block_ids,
    initial_states,
    forcing_vectors,
    sampling: CycleFourierSampling,
    numerical_config: dict,
    *,
    forcing_phases=None,
    workers: int = 1,
):
    """Validate shared block/condition axes and lazily construct tasks."""
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
    tasks = (
        (initial_states[block_index], forcing_vectors, sampling, numerical_config)
        if forcing_phases is None else
        (initial_states[block_index], forcing_vectors, condition_phases, sampling, numerical_config)
        for block_index in range(len(block_ids))
    )
    return block_ids, forcing_vectors, tasks


def integrate_cycle_fourier_conditions(
    block_ids, initial_states, forcing_vectors, sampling: CycleFourierSampling,
    numerical_config: dict, *, forcing_phases=None, workers: int = 1,
) -> np.ndarray:
    """Integrate direct ``Z_n`` with axes block,condition,state,cycle,harmonic.

    Cycles reduce within-trajectory noise; matched initial-state blocks are
    the independent statistical replicates. The unforced reference is run
    once per block and reused across every strength and direction.
    """
    block_ids, forcing_vectors, tasks = _condition_tasks(
        block_ids, initial_states, forcing_vectors, sampling, numerical_config,
        forcing_phases=forcing_phases, workers=workers,
    )
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


def _integrate_cycle_fourier_block_means(arguments):
    """Average each condition on the worker before transferring its result."""
    (
        initial_state,
        forcing_vectors,
        forcing_phases,
        sampling,
        numerical_config,
        uniform_phases,
    ) = _unpack_block_arguments(arguments)
    if uniform_phases and isinstance(sampling, CycleFourierSampling):
        fast = try_fourier_block(
            initial_state,
            forcing_vectors,
            forcing_phases,
            sampling.omega,
            sampling.phase,
            sampling.discard_time,
            sampling.n_cycle,
            sampling.n_phase,
            sampling.harmonics,
            numerical_config,
            average_cycles=True,
        )
        if fast is not None:
            return fast
    # The reduction is identical to fourier.mean(axis=3) in the cycle API.
    return np.asarray(_integrate_cycle_fourier_block(arguments)).mean(axis=2)


def iter_cycle_fourier_condition_means(
    block_ids, initial_states, forcing_vectors, sampling: CycleFourierSampling,
    numerical_config: dict, *, forcing_phases=None, workers: int = 1,
    max_in_flight: int | None = None,
):
    """Yield indexed block means independently of persistence chunk size.

    Each complete matched block still owns every condition; completion order
    is a scheduling detail and never changes the block/condition axes.
    """
    _, _, tasks = _condition_tasks(
        block_ids, initial_states, forcing_vectors, sampling, numerical_config,
        forcing_phases=forcing_phases, workers=workers,
    )
    yield from iter_task_results(
        _integrate_cycle_fourier_block_means, tasks, workers=workers,
        max_in_flight=max_in_flight,
    )
