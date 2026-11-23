"""Verify compiled C99 DC phase-space moment kernel against SciPy reference."""
import numpy as np

from lorenz._fast_dop853 import try_dc_moments_block, try_fourier_block
from lorenz.core import simulate_phase_samples


def test_dc_moments_kernel_matches_fourier_and_scipy() -> None:
    cfg = {
        "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "solver": {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11},
    }
    y0 = np.array([1.2, -2.5, 22.0], dtype=float)
    vectors = np.array([[0.0, 0.0, 0.0], [3.0, 1.5, 0.0], [0.0, 0.0, -2.0]], dtype=float)
    phases = np.array([[0.0, 0.0, 0.0], [0.2, -0.3, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    omega = 6.0
    discard_time = 2.0
    n_cycle = 8
    n_phase = 32
    harmonics = np.array([0, 1, 2, 3, 4, 5], dtype=int)

    fourier_only = try_fourier_block(
        y0, vectors, phases, omega, 0.0, discard_time, n_cycle, n_phase, harmonics, cfg,
        average_cycles=True,
    )
    res = try_dc_moments_block(
        y0, vectors, phases, omega, 0.0, discard_time, n_cycle, n_phase, harmonics, cfg,
    )
    assert res is not None and fourier_only is not None
    means, moments = res
    assert np.array_equal(means, fourier_only)
    assert moments.shape == (len(vectors), 10)

    total_t = n_cycle * (2.0 * np.pi / omega)
    for idx, vec in enumerate(vectors):
        samp = simulate_phase_samples(
            y0, vec, omega, phase=0.0, discard_time=discard_time,
            n_cycle=n_cycle, n_phase=n_phase, cfg=cfg,
            forcing_phases=phases[idx],
        )
        x = samp.values[0].ravel()
        y = samp.values[1].ravel()
        z = samp.values[2].ravel()
        pos = x > 0.0
        ref = np.array([
            np.mean(x * y),
            (z[-1] - z[0]) / total_t,
            np.mean(pos),
            np.mean(np.where(pos, x, 0.0)),
            np.mean(np.where(pos, 0.0, x)),
            np.mean(np.where(pos, z, 0.0)),
            np.mean(np.where(pos, 0.0, z)),
            np.mean(np.where(pos, x * y, 0.0)),
            np.mean(x * x),
            np.mean(z * z),
        ])
        assert np.allclose(moments[idx], ref, rtol=1e-11, atol=1e-11)
