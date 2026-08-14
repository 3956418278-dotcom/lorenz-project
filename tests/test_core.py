import numpy as np
import pytest

from lorenz.core import _forcing_angle, phase_mean, simulate_phase_samples


@pytest.fixture
def cfg():
    return {
        "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "solver": {"rtol": 1e-8, "atol": 1e-10},
    }


def test_phase_samples_retain_cycles_and_report_timing_and_phase(cfg):
    omega = 2.5
    forcing_phase = 0.4
    discard_time = 0.7
    n_cycle = 2
    n_phase = 4

    result = simulate_phase_samples(
        np.array([1.0, 1.0, 1.0]),
        np.array([0.1, -0.2, 0.3]),
        omega,
        forcing_phase,
        discard_time,
        n_cycle,
        n_phase,
        cfg,
    )

    period = 2 * np.pi / omega
    expected_times = discard_time + (
        np.arange(n_cycle)[:, None] + np.arange(n_phase)[None, :] / n_phase
    ) * period
    assert result.values.shape == (3, n_cycle, n_phase)
    np.testing.assert_allclose(result.sample_times, expected_times)
    np.testing.assert_allclose(
        result.phase_offset, (omega * discard_time + forcing_phase) % (2 * np.pi)
    )
    expected_phase = (
        result.phase_offset + 2 * np.pi * np.arange(n_phase) / n_phase
    ) % (2 * np.pi)
    actual_phase = (omega * result.sample_times[0] + forcing_phase) % (2 * np.pi)
    np.testing.assert_allclose(actual_phase, expected_phase)
    np.testing.assert_allclose(
        (omega * result.sample_times[1] + forcing_phase) % (2 * np.pi),
        expected_phase,
    )


def test_phase_metadata_uses_the_same_large_time_expression_as_rhs():
    omega = 12345.6789
    sample_time = 9.87654321e8
    phase = 0.314

    expected = np.mod(_forcing_angle(sample_time, omega, phase), 2 * np.pi)
    nested_mod = np.mod(
        omega * np.mod(sample_time, 2 * np.pi / omega) + phase, 2 * np.pi
    )

    assert expected != nested_mod
    assert expected == np.mod(_forcing_angle(sample_time, omega, phase), 2 * np.pi)


def test_zero_time_single_sample_returns_initial_state(cfg):
    initial = np.array([1.0, 2.0, 3.0])

    result = simulate_phase_samples(
        initial, np.zeros(3), 1.0, -0.5, 0.0, 1, 1, cfg
    )

    np.testing.assert_array_equal(result.values[:, 0, 0], initial)
    np.testing.assert_array_equal(result.sample_times, [[0.0]])
    np.testing.assert_allclose(result.phase_offset, (-0.5) % (2 * np.pi))


def test_phase_mean_is_only_the_cycle_average_convenience(cfg):
    args = (
        np.array([1.0, 1.0, 1.0]),
        np.zeros(3),
        3.0,
        0.2,
        0.1,
        2,
        3,
        cfg,
    )

    full = simulate_phase_samples(*args)
    averaged = phase_mean(*args)

    np.testing.assert_allclose(averaged, full.values.mean(axis=1))
    assert averaged.shape == (3, 3)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("omega", 0.0, "omega"),
        ("omega", np.inf, "omega"),
        ("phase", np.nan, "phase"),
        ("discard_time", -0.1, "discard_time"),
        ("discard_time", np.inf, "discard_time"),
        ("n_cycle", 0, "n_cycle"),
        ("n_cycle", 1.5, "n_cycle"),
        ("n_phase", True, "n_phase"),
    ],
)
def test_phase_sampling_rejects_invalid_scalar_inputs(
    cfg, argument, value, message
):
    kwargs = {
        "initial_state": np.ones(3),
        "forcing_vector": np.zeros(3),
        "omega": 1.0,
        "phase": 0.0,
        "discard_time": 0.0,
        "n_cycle": 1,
        "n_phase": 1,
        "cfg": cfg,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=message):
        simulate_phase_samples(**kwargs)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("initial_state", np.ones((3, 1)), "initial_state"),
        ("initial_state", np.array([1.0, 2.0, np.nan]), "initial_state"),
        ("forcing_vector", np.ones(2), "forcing_vector"),
        ("forcing_vector", np.array([1.0, np.inf, 2.0]), "forcing_vector"),
    ],
)
def test_phase_sampling_rejects_invalid_vectors(cfg, argument, value, message):
    kwargs = {
        "initial_state": np.ones(3),
        "forcing_vector": np.zeros(3),
        "omega": 1.0,
        "phase": 0.0,
        "discard_time": 0.0,
        "n_cycle": 1,
        "n_phase": 1,
        "cfg": cfg,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=message):
        simulate_phase_samples(**kwargs)


def test_phase_sampling_rejects_times_that_overflow_or_collapse(cfg):
    with pytest.raises(ValueError, match="sampling times"):
        simulate_phase_samples(
            np.ones(3), np.zeros(3), 1e-320, 0.0, 0.0, 1, 2, cfg
        )
    with pytest.raises(ValueError, match="sampling times"):
        simulate_phase_samples(
            np.ones(3), np.zeros(3), 1e300, 0.0, 1e300, 1, 2, cfg
        )
