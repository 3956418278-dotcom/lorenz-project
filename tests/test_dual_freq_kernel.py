"""Unit and contract tests for the C99 dual-frequency DOP853 projection kernel."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lorenz._fast_dop853 import load_c_solver, try_dc_moments_block, try_dual_freq_block


def _cfg() -> dict:
    return {
        "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "solver": {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11},
    }


def _scipy_dual_reference(
    y0: np.ndarray,
    fv1: np.ndarray,
    fp1: np.ndarray,
    omega1: float,
    fv2: np.ndarray,
    fp2: np.ndarray,
    omega2: float,
    discard_time: float,
    dt: float,
    n_samples: int,
    target_omegas: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = cfg["lorenz"]["sigma"]
    rho = cfg["lorenz"]["rho"]
    beta = cfg["lorenz"]["beta"]

    def rhs(t, y):
        f = fv1 * np.sin(omega1 * t + fp1) + fv2 * np.sin(omega2 * t + fp2)
        return [
            sigma * (y[1] - y[0]) + f[0],
            rho * y[0] - y[1] - y[0] * y[2] + f[1],
            y[0] * y[1] - beta * y[2] + f[2],
        ]

    times = discard_time + np.arange(n_samples, dtype=float) * dt
    solver_kwargs = {
        "rtol": cfg["solver"]["rtol"],
        "atol": cfg["solver"]["atol"],
    }
    if cfg["solver"].get("first_step") is not None:
        solver_kwargs["first_step"] = cfg["solver"]["first_step"]
    sol = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        y0,
        method="DOP853",
        t_eval=times,
        **solver_kwargs,
    )
    assert sol.success
    vals = sol.y  # (3, n_samples)
    exp_mat = np.exp(-1j * np.outer(times, target_omegas))  # (n_samples, n_freq)
    z_out = (vals @ exp_mat) / float(n_samples)

    vx, vy, vz = vals[0], vals[1], vals[2]
    xy = vx * vy
    pos = vx > 0.0
    total_t = float(n_samples) * dt
    moments = np.array([
        np.mean(xy),
        (vz[-1] - vz[0]) / total_t,
        np.mean(pos),
        np.mean(np.where(pos, vx, 0.0)),
        np.mean(np.where(~pos, vx, 0.0)),
        np.mean(np.where(pos, vz, 0.0)),
        np.mean(np.where(~pos, vz, 0.0)),
        np.mean(np.where(pos, xy, 0.0)),
        np.mean(vx * vx),
        np.mean(vz * vz),
    ], dtype=float)
    return z_out, moments


def test_dual_freq_matches_scipy_dop853_and_moments():
    if load_c_solver() is None:
        pytest.skip("C99 compiler unavailable")
    cfg = _cfg()
    y0 = np.array([-4.2, -5.1, 21.3], dtype=float)
    T = 6.4
    n_samples = 256
    dt = T / n_samples  # 0.025
    d_om = 2.0 * math.pi / T
    omega1 = 4 * d_om
    omega2 = 7 * d_om
    target_omegas = np.array([0.0, omega1, omega2, 2 * omega1, 2 * omega2, omega1 + omega2, abs(omega1 - omega2)])

    fv1 = np.array([[0.0, 0.0, 0.0], [1.2, -0.7, 0.9], [0.0, 1.5, 0.0]], dtype=float)
    fp1 = np.array([[0.0, 0.0, 0.0], [0.3, -0.8, 0.5], [0.0, 0.25, 0.0]], dtype=float)
    fv2 = np.array([[0.0, 0.0, 0.0], [-0.5, 0.8, 1.1], [0.0, 0.0, 1.3]], dtype=float)
    fp2 = np.array([[0.0, 0.0, 0.0], [-0.4, 0.6, -1.1], [0.0, 0.0, -0.7]], dtype=float)
    discard_time = 2.0

    res = try_dual_freq_block(
        y0, fv1, fp1, omega1, fv2, fp2, omega2,
        discard_time, dt, n_samples, target_omegas, cfg, return_moments=True,
    )
    assert res is not None
    z_c, m_c = res
    assert z_c.shape == (3, 3, len(target_omegas))
    assert m_c.shape == (3, 10)

    for c in range(3):
        z_ref, m_ref = _scipy_dual_reference(
            y0, fv1[c], fp1[c], omega1, fv2[c], fp2[c], omega2,
            discard_time, dt, n_samples, target_omegas, cfg,
        )
        np.testing.assert_allclose(z_c[c], z_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(m_c[c], m_ref, rtol=1e-10, atol=1e-10)


def test_dual_freq_long_window_phase_reanchor_matches_scipy():
    """Verify exact phase-recurrence re-anchoring over 1024 samples on non-chaotic rho=10."""
    if load_c_solver() is None:
        pytest.skip("C99 compiler unavailable")
    cfg = _cfg()
    cfg["lorenz"]["rho"] = 10.0  # stable spiral fixed points: zero Lyapunov amplification
    y0 = np.array([2.0, 3.0, 9.0], dtype=float)
    T = 25.6
    n_samples = 1024  # crosses 4 re-anchor boundaries (every 256 samples)
    dt = T / n_samples
    d_om = 2.0 * math.pi / T
    omega1 = 15 * d_om
    omega2 = 23 * d_om
    target_omegas = np.array([0.0, omega1, omega2, 2 * omega1, 2 * omega2, omega1 + omega2, abs(omega1 - omega2)])
    fv1 = np.array([[0.0, 1.5, 0.0]], dtype=float)
    fp1 = np.array([[0.0, 0.4, 0.0]], dtype=float)
    fv2 = np.array([[0.0, 0.0, 1.2]], dtype=float)
    fp2 = np.array([[0.0, 0.0, -0.6]], dtype=float)

    z_c, m_c = try_dual_freq_block(
        y0, fv1, fp1, omega1, fv2, fp2, omega2,
        4.0, dt, n_samples, target_omegas, cfg, return_moments=True,
    )
    z_ref, m_ref = _scipy_dual_reference(
        y0, fv1[0], fp1[0], omega1, fv2[0], fp2[0], omega2,
        4.0, dt, n_samples, target_omegas, cfg,
    )
    np.testing.assert_allclose(z_c[0], z_ref, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(m_c[0], m_ref, rtol=1e-11, atol=1e-11)


def test_dual_freq_reduces_exactly_to_single_freq_kernel():
    if load_c_solver() is None:
        pytest.skip("C99 compiler unavailable")
    cfg = _cfg()
    y0 = np.array([3.7, 6.2, 19.8], dtype=float)
    omega = 6.0
    n_cycle = 12
    n_phase = 32
    period = 2.0 * math.pi / omega
    dt = period / n_phase
    n_samples = n_cycle * n_phase
    discard_time = 4.0
    harmonics = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    target_omegas = harmonics.astype(float) * omega

    fv1 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, -2.0], [0.0, 1.8, 1.1]], dtype=float)
    fp1 = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, -0.4], [0.0, 0.7, -0.3]], dtype=float)
    fv2 = np.zeros_like(fv1)

    z_single, m_single = try_dc_moments_block(
        y0, fv1, fp1, omega, 0.0, discard_time, n_cycle, n_phase, harmonics, cfg
    )
    z_dual, m_dual = try_dual_freq_block(
        y0, fv1, fp1, omega, fv2, None, 3.1, discard_time, dt, n_samples, target_omegas, cfg,
        return_moments=True,
    )
    np.testing.assert_allclose(z_dual, z_single, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(m_dual, m_single, rtol=1e-11, atol=1e-11)


def test_dual_freq_r_equivariance_and_phase_reanchor_long_window():
    """Verify exact R = diag(-1, -1, 1) equivariance over >512 samples (crossing re-anchor)."""
    if load_c_solver() is None:
        pytest.skip("C99 compiler unavailable")
    cfg = _cfg()
    y0 = np.array([2.5, -3.1, 25.0], dtype=float)
    R = np.array([-1.0, -1.0, 1.0], dtype=float)
    y0_R = R * y0

    T = 25.6
    n_samples = 1024  # crosses 4 re-anchor boundaries (every 256 samples)
    dt = T / n_samples
    d_om = 2.0 * math.pi / T
    omega1 = 18 * d_om
    omega2 = 29 * d_om
    target_omegas = np.array([0.0, omega1, omega2, omega1 + omega2, abs(omega1 - omega2)])

    fv1 = np.array([[0.0, 1.2, 0.0], [0.0, 0.0, 1.4]], dtype=float)
    fv2 = np.array([[0.0, 0.0, 1.1], [0.0, 0.0, 0.9]], dtype=float)

    z_orig = try_dual_freq_block(
        y0, fv1, None, omega1, fv2, None, omega2, 1.0, dt, n_samples, target_omegas, cfg
    )
    z_refl = try_dual_freq_block(
        y0_R, fv1 * R[None, :], None, omega1, fv2 * R[None, :], None, omega2, 1.0, dt, n_samples, target_omegas, cfg
    )
    assert z_orig is not None and z_refl is not None
    # Exact equivariance: m[R f] = R m[f]
    np.testing.assert_allclose(z_refl, z_orig * R[None, :, None], rtol=1e-11, atol=1e-11)


def test_dual_freq_edge_cases_and_validation():
    if load_c_solver() is None:
        pytest.skip("C99 compiler unavailable")
    cfg = _cfg()
    y0 = np.array([1.0, 2.0, 20.0], dtype=float)
    fv1 = np.array([[0.0, 0.0, 1.0]], dtype=float)
    fv2 = np.array([[0.0, 0.0, 0.5]], dtype=float)
    om = np.array([0.0, 2.0, 5.0])

    # discard_time = 0 and n_samples = 1
    z1, m1 = try_dual_freq_block(y0, fv1, None, 2.0, fv2, None, 3.0, 0.0, 0.025, 1, om, cfg, return_moments=True)
    np.testing.assert_allclose(z1[0, :, 0], y0, atol=1e-14)
    np.testing.assert_allclose(z1[0, :, 1], y0, atol=1e-14)
    assert m1[0, 0] == pytest.approx(y0[0] * y0[1])

    # discard_time = 0 and n_samples > 1
    z2 = try_dual_freq_block(y0, fv1, None, 2.0, fv2, None, 3.0, 0.0, 0.025, 20, om, cfg)
    assert z2 is not None and np.isfinite(z2).all()

    # Invalid arguments cleanly return None
    assert try_dual_freq_block(y0, fv1, None, 2.0, fv2, None, 3.0, 0.0, -0.01, 10, om, cfg) is None
    assert try_dual_freq_block(y0, fv1, None, 2.0, fv2, None, 3.0, 0.0, 0.025, 0, om, cfg) is None
    assert try_dual_freq_block(y0, fv1, None, 2.0, np.zeros((2, 3)), None, 3.0, 0.0, 0.025, 10, om, cfg) is None
    assert try_dual_freq_block(y0, fv1, None, 2.0, fv2, None, 3.0, 0.0, 0.025, 10, np.array([np.nan]), cfg) is None
    assert try_dual_freq_block(y0, np.array([[np.nan, 0.0, 0.0]]), None, 2.0, fv2, None, 3.0, 0.0, 0.025, 1, om, cfg) is None
    assert try_dual_freq_block(y0, np.array([[np.nan, 0.0, 0.0]]), None, 2.0, fv2, None, 3.0, 1.0, 0.025, 10, om, cfg) is None
    assert try_dual_freq_block(y0, fv1, None, -1.0, fv2, None, 3.0, 1.0, 0.025, 10, om, cfg) is None
    assert try_dual_freq_block(y0, fv1, None, 2.0, fv2, None, -3.0, 1.0, 0.025, 10, om, cfg) is None


def test_dual_freq_degenerate_omegas_negative_targets_and_first_step():
    """Verify omega1 == omega2 != 0, negative target_omegas, first_step > 0, and discard_time > 0 with n_samples=1."""
    if load_c_solver() is None:
        pytest.skip("C99 compiler unavailable")
    cfg = _cfg()
    y0 = np.array([2.1, -3.4, 22.5], dtype=float)
    w = 3.7
    fv1 = np.array([[0.0, 1.0, 0.0]], dtype=float)
    fp1 = np.array([[0.0, 0.3, 0.0]], dtype=float)
    fv2 = np.array([[0.0, 0.0, 0.8]], dtype=float)
    fp2 = np.array([[0.0, 0.0, -0.5]], dtype=float)
    t_om = np.array([-2.0 * w, -w, 0.0, w, 2.0 * w], dtype=float)

    # 1. omega1 == omega2 and negative target_omegas
    z_c, m_c = try_dual_freq_block(
        y0, fv1, fp1, w, fv2, fp2, w, 1.5, 0.025, 128, t_om, cfg, return_moments=True
    )
    z_ref, m_ref = _scipy_dual_reference(
        y0, fv1[0], fp1[0], w, fv2[0], fp2[0], w, 1.5, 0.025, 128, t_om, cfg
    )
    np.testing.assert_allclose(z_c[0], z_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(m_c[0], m_ref, rtol=1e-10, atol=1e-10)
    # Conjugate symmetry for real signal: Z[-w] == conj(Z[+w])
    np.testing.assert_allclose(z_c[0, :, 0], np.conj(z_c[0, :, 4]), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(z_c[0, :, 1], np.conj(z_c[0, :, 3]), rtol=1e-13, atol=1e-13)

    # 2. Explicit first_step > 0
    cfg_fs = {"lorenz": dict(cfg["lorenz"]), "solver": {**cfg["solver"], "first_step": 0.005}}
    z_fs = try_dual_freq_block(y0, fv1, fp1, w, fv2, fp2, 5.1, 1.0, 0.025, 64, t_om, cfg_fs)
    z_fs_ref, _ = _scipy_dual_reference(y0, fv1[0], fp1[0], w, fv2[0], fp2[0], 5.1, 1.0, 0.025, 64, t_om, cfg_fs)
    np.testing.assert_allclose(z_fs[0], z_fs_ref, rtol=1e-10, atol=1e-10)

    # 3. discard_time > 0 with n_samples == 1
    z_1s, m_1s = try_dual_freq_block(
        y0, fv1, fp1, w, fv2, fp2, 5.1, 1.25, 0.025, 1, t_om, cfg, return_moments=True
    )
    z_1s_ref, m_1s_ref = _scipy_dual_reference(
        y0, fv1[0], fp1[0], w, fv2[0], fp2[0], 5.1, 1.25, 0.025, 1, t_om, cfg
    )
    np.testing.assert_allclose(z_1s[0], z_1s_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(m_1s[0], m_1s_ref, rtol=1e-10, atol=1e-10)
    assert m_1s[0, 1] == 0.0
