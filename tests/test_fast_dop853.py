"""Numerical parity and fallback checks for the C99 DOP853 fast path."""
from __future__ import annotations

import numpy as np
import pytest

from lorenz import _fast_dop853
from lorenz.core import simulate_phase_samples
from lorenz.ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from lorenz.sampling import (
    CycleFourierSampling,
    _integrate_cycle_fourier_block,
    _integrate_cycle_fourier_block_means,
)

DEFAULT_CFG = {
    "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "solver": {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11},
}


def test_c_extension_loads_and_matches_spinup(monkeypatch):
    assert _fast_dop853.load_c_solver() is not None
    proposal = SymmetricXYUniformProposal(
        x_half_width=20.0, y_half_width=20.0, z_bounds=(0.0, 40.0)
    )
    c_blocks = generate_initial_state_blocks(
        range(4), 12345, proposal, 5.0, DEFAULT_CFG
    )

    monkeypatch.setenv("LORENZ_DISABLE_C_SOLVER", "1")
    py_blocks = generate_initial_state_blocks(
        range(4), 12345, proposal, 5.0, DEFAULT_CFG
    )
    np.testing.assert_allclose(c_blocks.final_states, py_blocks.final_states, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("forcing", "phase", "forcing_phases", "discard_time"),
    [
        ((0.0, 0.0, 0.0), 0.0, None, 2.0),
        ((0.5, 0.0, -0.25), -0.4, None, 0.0),
        ((0.4, -0.3, 0.2), 0.3, (0.3, 0.3 + 0.5 * np.pi, 0.3 + np.pi), 3.0),
    ],
)
def test_c_phase_samples_match_scipy_reference(
    monkeypatch, forcing, phase, forcing_phases, discard_time
):
    assert _fast_dop853.load_c_solver() is not None
    y0 = np.array([-4.5, -5.2, 22.1])
    c_samples = simulate_phase_samples(
        y0,
        forcing,
        omega=6.0,
        phase=phase,
        discard_time=discard_time,
        n_cycle=3,
        n_phase=16,
        cfg=DEFAULT_CFG,
        forcing_phases=forcing_phases,
    )

    monkeypatch.setenv("LORENZ_DISABLE_C_SOLVER", "1")
    py_samples = simulate_phase_samples(
        y0,
        forcing,
        omega=6.0,
        phase=phase,
        discard_time=discard_time,
        n_cycle=3,
        n_phase=16,
        cfg=DEFAULT_CFG,
        forcing_phases=forcing_phases,
    )
    np.testing.assert_allclose(c_samples.values, py_samples.values, rtol=0.0, atol=1e-10)
    assert c_samples.phase_offset == py_samples.phase_offset


def test_c_fourier_block_and_means_match_scipy_reference(monkeypatch):
    assert _fast_dop853.load_c_solver() is not None
    y0 = np.array([6.1, 8.2, 24.7])
    forcing_vectors = np.array([
        [0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        [0.3, 0.3, 0.0],
        [-0.3, 0.3, -0.3],
    ])
    forcing_phases = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2],
        [0.2, 0.2 + 0.5 * np.pi, 0.2],
        [0.2, 0.2, 0.2 + np.pi],
    ])
    sampling = CycleFourierSampling(
        omega=6.0,
        phase=-0.25,
        discard_time=2.5,
        n_cycle=4,
        n_phase=32,
        harmonics=np.array([0, 1, 2, 3, 4, 5]),
    )

    c_cycles = np.asarray(
        _integrate_cycle_fourier_block((y0, forcing_vectors, sampling, DEFAULT_CFG))
    )
    c_means = _integrate_cycle_fourier_block_means(
        (y0, forcing_vectors, sampling, DEFAULT_CFG)
    )
    c_mixed_means = _integrate_cycle_fourier_block_means(
        (y0, forcing_vectors, forcing_phases, sampling, DEFAULT_CFG)
    )

    monkeypatch.setenv("LORENZ_DISABLE_C_SOLVER", "1")
    py_cycles = np.asarray(
        _integrate_cycle_fourier_block((y0, forcing_vectors, sampling, DEFAULT_CFG))
    )
    py_means = _integrate_cycle_fourier_block_means(
        (y0, forcing_vectors, sampling, DEFAULT_CFG)
    )
    py_mixed_means = _integrate_cycle_fourier_block_means(
        (y0, forcing_vectors, forcing_phases, sampling, DEFAULT_CFG)
    )

    np.testing.assert_allclose(c_cycles, py_cycles, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(c_means, py_means, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(c_mixed_means, py_mixed_means, rtol=0.0, atol=1e-11)


def test_non_dop853_solver_falls_back_cleanly_to_scipy():
    rk45_cfg = {
        "lorenz": DEFAULT_CFG["lorenz"],
        "solver": {"method": "RK45", "rtol": 1e-6, "atol": 1e-8},
    }
    samples = simulate_phase_samples(
        np.array([1.0, 2.0, 15.0]),
        (0.2, 0.0, 0.0),
        omega=6.0,
        phase=0.0,
        discard_time=0.5,
        n_cycle=1,
        n_phase=8,
        cfg=rk45_cfg,
    )
    assert samples.values.shape == (3, 1, 8)
    assert np.isfinite(samples.values).all()
