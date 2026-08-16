import numpy as np
import pytest

import lorenz.strength_study as strength_study
from lorenz.strength_study import (
    CycleFourierSampling,
    FrequencyStudyData,
    StrengthStudyData,
    dc_components,
    frequency_key,
    integrate_cycle_fourier_conditions,
    second_harmonic_estimators,
    strength_fourier_components,
)


def _study(block_ids=(4, 9)):
    strengths = np.array([0.5, 1.0])
    harmonics = np.array([0, 1, 2])
    shape = (len(block_ids), len(strengths), 3, 2, len(harmonics))
    positive = np.full(shape, 7 + 0j)
    negative = np.full(shape, 3 + 0j)
    unforced = np.full((len(block_ids), 3, 2, len(harmonics)), 1 + 0j)
    return StrengthStudyData(
        block_ids=block_ids,
        strengths=strengths,
        harmonics=harmonics,
        omega=2.0,
        n_phase=8,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=np.zeros((len(block_ids), 3)),
        initial_states=np.ones((len(block_ids), 3)),
        child_spawn_keys=tuple((value,) for value in block_ids),
        generation_metadata={},
    )


def test_condition_integration_preserves_block_and_condition_axes(monkeypatch):
    sampling = CycleFourierSampling(
        omega=2.0,
        phase=0.0,
        discard_time=4.0,
        n_cycle=2,
        n_phase=8,
        harmonics=np.array([0, 1, 2]),
    )
    block_ids = (11, 27)
    initial_states = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    forcing_vectors = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    def fake_integrate(arguments):
        initial_state, forcing, local_sampling, _ = arguments
        value = initial_state + forcing
        return np.broadcast_to(
            value[:, None, None],
            (3, local_sampling.n_cycle, len(local_sampling.harmonics)),
        )

    monkeypatch.setattr(strength_study, "_integrate_cycle_fourier", fake_integrate)
    result = integrate_cycle_fourier_conditions(
        block_ids,
        initial_states,
        forcing_vectors,
        sampling,
        {"lorenz": {}, "solver": {}},
    )

    assert result.shape == (2, 2, 3, 2, 3)
    np.testing.assert_allclose(result[0, 0, :, 0, 0], initial_states[0])
    np.testing.assert_allclose(
        result[1, 1, :, 0, 0], initial_states[1] + forcing_vectors[1]
    )


def test_explicit_components_keep_a2_u2_and_dc_semantics_separate():
    study = _study()
    components = strength_fourier_components(study)
    np.testing.assert_allclose(components.odd, 2)
    np.testing.assert_allclose(components.forced_even, 5)
    np.testing.assert_allclose(components.unforced, 1)

    second = second_harmonic_estimators(study)
    np.testing.assert_allclose(second["alternative_A2"], 5)
    np.testing.assert_allclose(second["unforced_U2_diagnostic"], 1)
    np.testing.assert_allclose(second["current_E2"], 4)

    dc = dc_components(study)
    np.testing.assert_allclose(dc["forced_even_A0"], 5)
    np.testing.assert_allclose(dc["unforced_U0"], 1)
    np.testing.assert_allclose(dc["current_E0"], 4)


def test_dense_block_spectrum_hann_full_window_peak_and_phase():
    """The full-window Hann spectrum must report the correct peak frequency
    and the complex phase referenced to the absolute time origin t=0.

    For a sinusoid exactly on a bin k >= 2 of the full-interval grid, the
    Hann window has zero response at the 2*k image bin, so
    S_b(Omega_k) = (A/2) exp(i*phase) exactly (up to roundoff), with the
    phase measured at t = 0.
    """
    from lorenz.strength_study import dense_block_spectrum

    dt = 0.1
    count = 1024
    discard = 160.0  # nonzero absolute-time origin exercises the rotation
    times = discard + dt * np.arange(count, dtype=float)
    full_step = 2 * np.pi / (count * dt)
    peak_bin = 10
    omega0 = peak_bin * full_step
    amplitude = 3.0
    phase = 0.7
    values = np.stack(
        [
            amplitude * np.cos(omega0 * times + phase),
            amplitude * np.sin(omega0 * times + phase),
            np.zeros(count),
        ]
    )
    spectrum = dense_block_spectrum(values, times)
    # Correct peak frequency: the largest coefficient sits at omega0.
    assert int(np.argmax(np.abs(spectrum.coefficients[0]))) == peak_bin
    # Correct complex phase and amplitude: S_b(omega0) = (A/2) exp(i*phase).
    # Exact for the periodic Hann (zero image-bin leakage); tolerance is
    # float64 roundoff of the 1024-point FFT and the t0 = 160 rotation.
    np.testing.assert_allclose(
        spectrum.coefficients[0, peak_bin],
        amplitude / 2 * np.exp(1j * phase),
        rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(
        spectrum.coefficients[1, peak_bin],
        amplitude / 2 * np.exp(1j * (phase - np.pi / 2)),
        rtol=1e-10, atol=1e-10,
    )
    assert spectrum.segment_count == 1
    # Complete one-sided spectrum by default: the grid reaches the Nyquist
    # frequency and is authoritative.
    np.testing.assert_allclose(
        spectrum.frequency_grid[-1], np.pi / dt, rtol=1e-12
    )
    np.testing.assert_allclose(
        spectrum.frequency_grid, full_step * np.arange(count // 2 + 1)
    )


def test_dense_block_spectrum_physical_range_follows_dt_and_length():
    """The spectrum range is a physical frequency, not a fixed bin count.

    The default covers the complete one-sided range up to pi/dt regardless
    of observation length; an explicit maximum_omega selects the bins of the
    computed grid up to that physical frequency.
    """
    from lorenz.strength_study import dense_block_spectrum

    dt = 0.1
    rng = np.random.default_rng(3)
    results = {}
    for count in (512, 1024):
        times = 10.0 + dt * np.arange(count, dtype=float)
        values = rng.normal(size=(3, count))
        spectrum = dense_block_spectrum(values, times)
        results[count] = spectrum
        # Full range by default, at the Nyquist frequency pi/dt.
        np.testing.assert_allclose(
            spectrum.frequency_grid[-1], np.pi / dt, rtol=1e-12
        )
        assert spectrum.coefficients.shape[1] == count // 2 + 1
    # Same physical range for both lengths, different bin counts.
    assert (
        results[512].frequency_grid[-1] == results[1024].frequency_grid[-1]
    )
    assert results[512].frequency_grid.shape[0] < results[1024].frequency_grid.shape[0]
    # An explicit PHYSICAL maximum selects bins on the computed grid.
    capped = dense_block_spectrum(values, times, maximum_omega=2.5)
    assert capped.frequency_grid[-1] <= 2.5
    assert capped.frequency_grid.shape[0] == int(2.5 / (2 * np.pi / (1024 * dt))) + 1
    assert np.array_equal(capped.frequency_grid, results[1024].frequency_grid[: len(capped.frequency_grid)])
    with pytest.raises(ValueError):
        dense_block_spectrum(values, times, maximum_omega=-1.0)


def test_frequency_container_preserves_crossed_blocks_and_stable_keys():
    first = _study()
    second = StrengthStudyData(**{**first.__dict__, "omega": 4.0})
    data = FrequencyStudyData(
        block_ids=first.block_ids,
        strengths=first.strengths,
        harmonics=first.harmonics,
        frequencies=(2.0, 4.0),
        studies={2.0: first, 4.0: second},
        frequency_runtime_seconds={2.0: 1.0, 4.0: 2.0},
        source={2.0: {}, 4.0: {}},
    )

    assert data.studies[4.0].block_ids == data.block_ids
    assert frequency_key(0.5) == "0p5"
    assert frequency_key(2.0) == "2"
