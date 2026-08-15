"""Lorenz equations, forced integration, and phase-grid sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PhaseSamples:
    """State samples on a ``cycle x phase`` forcing grid.

    ``values`` has shape ``(state, cycle, phase)`` and ``sample_times`` has
    shape ``(cycle, phase)``. ``phase_offset`` is the forcing phase at phase
    index zero, modulo ``2*pi``; it is the offset expected by response Fourier
    extraction.
    """

    values: np.ndarray
    sample_times: np.ndarray
    phase_offset: float


def _forcing_angle(t, omega: float, phase: float):
    """Use one floating-point expression for the RHS and phase metadata."""
    return omega * t + phase


def lorenz_rhs(t, state, cfg, forcing=None, omega=0.0, phase=0.0):
    x, y, z = state
    sigma = cfg["lorenz"]["sigma"]
    rho = cfg["lorenz"]["rho"]
    beta = cfg["lorenz"]["beta"]
    fx = fy = fz = 0.0
    if forcing is not None:
        f = np.sin(_forcing_angle(t, omega, phase))
        fx, fy, fz = np.asarray(forcing, dtype=float) * f
    return [
        sigma * (y - x) + fx,
        rho * x - y - x * z + fy,
        x * y - beta * z + fz,
    ]


def check_solution(sol, expected_shape):
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    if sol.y.shape != expected_shape:
        raise RuntimeError(f"unexpected sol.y shape {sol.y.shape}, expected {expected_shape}")
    if not np.isfinite(sol.y).all():
        raise RuntimeError("non-finite values in solve_ivp output")


def _positive_count(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def simulate_phase_samples(
    initial_state,
    forcing_vector,
    omega: float,
    phase: float,
    discard_time: float,
    n_cycle: int,
    n_phase: int,
    cfg: dict,
) -> PhaseSamples:
    """Integrate from forcing onset and retain a cycle-resolved phase grid.

    The first sample is exactly at ``discard_time``. Later phase indices and
    cycles advance by uniform fractions and whole multiples of the forcing
    period, respectively. The caller owns construction of ``initial_state``;
    in particular, ensemble generation is not inferred from a seed here.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    forcing_vector = np.asarray(forcing_vector, dtype=float)
    if initial_state.shape != (3,) or not np.isfinite(initial_state).all():
        raise ValueError("initial_state must be a finite vector with shape (3,)")
    if forcing_vector.shape != (3,) or not np.isfinite(forcing_vector).all():
        raise ValueError("forcing_vector must be a finite vector with shape (3,)")
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    if not np.isfinite(discard_time) or discard_time < 0:
        raise ValueError("discard_time must be finite and nonnegative")
    n_cycle = _positive_count(n_cycle, "n_cycle")
    n_phase = _positive_count(n_phase, "n_phase")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        period = 2 * np.pi / float(omega)
    cycle = np.arange(n_cycle, dtype=float)
    phase_index = np.arange(n_phase, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        sample_times = float(discard_time) + (
            cycle[:, None] + phase_index[None, :] / n_phase
        ) * period
    flat_times = sample_times.ravel()
    if not np.isfinite(flat_times).all() or not np.all(np.diff(flat_times) > 0):
        raise ValueError(
            "sampling times must be finite and strictly increasing at float precision"
        )

    if flat_times[-1] == 0.0:
        values = initial_state[:, None]
    else:
        sol = solve_ivp(
            lambda t, state: lorenz_rhs(
                t, state, cfg, forcing_vector, omega, phase
            ),
            [0.0, float(flat_times[-1])],
            initial_state,
            t_eval=flat_times,
            **cfg["solver"],
        )
        check_solution(sol, (3, n_cycle * n_phase))
        values = sol.y

    phase_offset = np.mod(_forcing_angle(flat_times[0], omega, phase), 2 * np.pi)
    return PhaseSamples(
        values=values.reshape(3, n_cycle, n_phase),
        sample_times=sample_times,
        phase_offset=float(phase_offset),
    )


def simulate_phase_and_dense(
    initial_state,
    forcing_vector,
    omega: float,
    phase: float,
    discard_time: float,
    n_cycle: int,
    n_phase: int,
    dense_dt: float,
    cfg: dict,
):
    """Integrate once and return phase-grid samples plus dense fixed-dt samples.

    Dense samples span ``[discard_time, last phase sample time]`` at the fixed
    physical spacing ``dense_dt`` so lab-frame spectral summaries (Welch PSD,
    folded cycle means, raw segments) share the exact trajectory of the
    inferential phase grid.  DOP853 dense output makes the merged evaluation
    grid cheap relative to the integration itself.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    forcing_vector = np.asarray(forcing_vector, dtype=float)
    if initial_state.shape != (3,) or not np.isfinite(initial_state).all():
        raise ValueError("initial_state must be a finite vector with shape (3,)")
    if forcing_vector.shape != (3,) or not np.isfinite(forcing_vector).all():
        raise ValueError("forcing_vector must be a finite vector with shape (3,)")
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    if not np.isfinite(discard_time) or discard_time < 0:
        raise ValueError("discard_time must be finite and nonnegative")
    if not np.isfinite(dense_dt) or dense_dt <= 0:
        raise ValueError("dense_dt must be finite and positive")
    n_cycle = _positive_count(n_cycle, "n_cycle")
    n_phase = _positive_count(n_phase, "n_phase")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        period = 2 * np.pi / float(omega)
    cycle = np.arange(n_cycle, dtype=float)
    phase_index = np.arange(n_phase, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        sample_times = float(discard_time) + (
            cycle[:, None] + phase_index[None, :] / n_phase
        ) * period
    flat_times = sample_times.ravel()
    if not np.isfinite(flat_times).all() or not np.all(np.diff(flat_times) > 0):
        raise ValueError(
            "sampling times must be finite and strictly increasing at float precision"
        )
    dense_count = int(np.floor((flat_times[-1] - float(discard_time)) / dense_dt)) + 1
    dense_times = float(discard_time) + dense_dt * np.arange(dense_count, dtype=float)
    dense_times = dense_times[dense_times <= flat_times[-1]]
    merged = np.union1d(flat_times, dense_times)
    merged = merged[merged >= flat_times[0]]
    merged = merged[merged <= flat_times[-1]]

    if merged[-1] == 0.0:
        values = initial_state[:, None]
    else:
        sol = solve_ivp(
            lambda t, state: lorenz_rhs(
                t, state, cfg, forcing_vector, omega, phase
            ),
            [0.0, float(merged[-1])],
            initial_state,
            t_eval=merged,
            **cfg["solver"],
        )
        check_solution(sol, (3, len(merged)))
        values = sol.y
    by_time = {float(value): index for index, value in enumerate(merged)}
    phase_indices = np.asarray([by_time[float(value)] for value in flat_times])
    dense_indices = np.asarray([by_time[float(value)] for value in dense_times])

    phase_offset = np.mod(_forcing_angle(flat_times[0], omega, phase), 2 * np.pi)
    phase_samples = PhaseSamples(
        values=values[:, phase_indices].reshape(3, n_cycle, n_phase),
        sample_times=sample_times,
        phase_offset=float(phase_offset),
    )
    return phase_samples, values[:, dense_indices], dense_times


def phase_mean(
    initial_state,
    forcing_vector,
    omega: float,
    phase: float,
    discard_time: float,
    n_cycle: int,
    n_phase: int,
    cfg: dict,
) -> np.ndarray:
    """Return a cycle average for exploratory use only."""
    result = simulate_phase_samples(
        initial_state,
        forcing_vector,
        omega,
        phase,
        discard_time,
        n_cycle,
        n_phase,
        cfg,
    )
    return result.values.mean(axis=1)
