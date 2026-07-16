"""Lorenz equations, spinup, forced integration, and phase sampling."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def lorenz_rhs(t, state, cfg, forcing=None, omega=0.0, phase=0.0):
    x, y, z = state
    sigma = cfg["lorenz"]["sigma"]
    rho = cfg["lorenz"]["rho"]
    beta = cfg["lorenz"]["beta"]
    fx = fy = fz = 0.0
    if forcing is not None:
        f = np.sin(omega * t + phase)
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


def spinup_state(seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    y0 = rng.random(3)
    sol = solve_ivp(
        lambda t, s: lorenz_rhs(t, s, cfg),
        [0, cfg["T_spinup"]],
        y0,
        t_eval=[cfg["T_spinup"]],
        **cfg["solver"],
    )
    check_solution(sol, (3, 1))
    return sol.y[:, -1]


def simulate_phase_samples(seed: int, forcing_vector, omega: float, phase: float,
                           n_skip: int, n_cycle: int, n_phase: int, cfg: dict,
                           initial_state=None):
    if initial_state is None:
        initial_state = spinup_state(seed, cfg)
    period = 2 * np.pi / omega
    m = np.arange(n_cycle)
    q = np.arange(n_phase)
    times = (n_skip + m[:, None] + q[None, :] / n_phase) * period
    sol = solve_ivp(
        lambda t, s: lorenz_rhs(t, s, cfg, forcing_vector, omega, phase),
        [0, float(times[-1, -1])],
        initial_state,
        t_eval=times.ravel(),
        **cfg["solver"],
    )
    check_solution(sol, (3, n_cycle * n_phase))
    return sol.y.reshape(3, n_cycle, n_phase)


def phase_mean(seed: int, forcing_vector, omega: float, phase: float,
               n_skip: int, n_cycle: int, n_phase: int, cfg: dict,
               initial_state=None):
    samples = simulate_phase_samples(seed, forcing_vector, omega, phase, n_skip,
                                     n_cycle, n_phase, cfg, initial_state=initial_state)
    return samples.mean(axis=1)
