import json

import numpy as np
import pytest

from lorenz.strength_identifiability import (
    _real_imag_summary,
    _sample_size_projection,
    analyze_strength_series,
    analyze_strength_study,
    persist_strength_study,
)
from lorenz.strength_series import TARGET_ORDERS, fit_block_power_series
from lorenz.strength_study import StrengthStudyData, validate_strength_sampling_config


def test_block_power_fit_recovers_complex_physical_coefficients():
    strengths = np.array([0.2, 0.5, 1.0, 2.0])
    low = np.array([[1 + 2j, -3 + 0.5j], [2 - 1j, 4 + 3j]])
    high = np.array([[0.1 - 0.2j, 0.5j], [-0.3 + 0.2j, 0.4 - 0.1j]])
    values = (
        strengths[None, :, None] * low[:, None]
        + strengths[None, :, None] ** 3 * high[:, None]
    )

    fit = fit_block_power_series(values, strengths, (1, 3))

    np.testing.assert_allclose(fit.coefficients[:, 0], low)
    np.testing.assert_allclose(fit.coefficients[:, 1], high)
    np.testing.assert_allclose(fit.fitted, values)
    np.testing.assert_allclose(fit.residuals, 0, atol=1e-13)


def test_target_order_definition_preserves_decision_family_order():
    assert tuple(TARGET_ORDERS.items()) == (
        ("odd_fundamental", (1, 3)),
        ("even_second_harmonic", (2, 4)),
        ("even_dc", (2, 4)),
    )


def test_power_fit_requires_explicit_block_and_strength_axes():
    with pytest.raises(ValueError, match="block, strength"):
        fit_block_power_series(np.ones(4), [0.2, 0.5, 1.0, 2.0], (1, 3))
    with pytest.raises(ValueError, match="at least two finite blocks"):
        fit_block_power_series(np.ones((1, 4)), [0.2, 0.5, 1.0, 2.0], (1, 3))
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_block_power_series(np.ones((2, 3)), [0.5, 0.2, 1.0], (1, 3))


def test_zero_signal_with_zero_variance_is_not_infinitely_resolved():
    real = _real_imag_summary(np.zeros((4, 2)))
    complex_values = _real_imag_summary(np.zeros((4, 2), dtype=complex))
    np.testing.assert_array_equal(real["magnitude_over_standard_error"], 0)
    np.testing.assert_array_equal(
        complex_values["magnitude_over_radial_standard_error"], 0
    )

    nonzero = _real_imag_summary(np.ones((4, 1)))
    assert np.isinf(nonzero["magnitude_over_standard_error"][0])
    projection = _sample_size_projection(
        real["magnitude_over_standard_error"], n_block=4
    )
    assert np.isinf(projection["ratio_3"]).all()


def test_complex_summary_retains_joint_real_imag_covariance():
    values = np.array(
        [[1 + 2j, 3 + 4j], [2 + 1j, 5 + 8j], [4 + 3j, 6 + 7j]],
        dtype=complex,
    )
    summary = _real_imag_summary(values)
    realified = np.concatenate((values.real, values.imag), axis=1)
    expected = np.cov(realified, rowvar=False, ddof=1)

    np.testing.assert_allclose(summary["sample_covariance_realified"], expected)
    np.testing.assert_allclose(
        summary["covariance_of_mean_realified"], expected / len(values)
    )
    assert summary["realified_feature_order"].startswith("all flattened real")


def test_config_rejects_nyquist_harmonic():
    config = {
        "strengths": [0.5, 1.0],
        "harmonics": [0, 1, 2],
        "discard_time": 1.0,
        "n_cycle": 2,
        "n_phase": 4,
        "protocol": {"omega": 2.0, "direction": [1.0, 0.0, 0.0]},
    }
    with pytest.raises(ValueError, match="strictly below Nyquist"):
        validate_strength_sampling_config(config)


def _synthetic_contrasts():
    strengths = np.array([0.25, 0.5, 1.0, 2.0])
    harmonics = np.arange(6)
    n_block = 8
    odd = np.zeros((n_block, 4, 3, 6), dtype=complex)
    even = np.zeros_like(odd)
    block_shift = np.linspace(-0.1, 0.1, n_block)[:, None]
    beta1 = np.array([1 + 2j, 2 - 0.5j, -1 + 0.2j])
    beta3 = np.array([0.05j, -0.1 + 0.02j, 0.04])
    gamma2 = np.array([0.5 - 0.2j, 0.1 + 0.3j, -0.4j])
    gamma4 = np.array([0.02, -0.03j, 0.01 + 0.01j])
    delta2 = np.array([0.4, -0.2, 0.8])
    delta4 = np.array([0.01, 0.02, -0.03])
    for index, strength in enumerate(strengths):
        odd[:, index, :, 1] = (
            strength * beta1 + strength**3 * beta3 + strength * block_shift
        )
        even[:, index, :, 2] = (
            strength**2 * gamma2 + strength**4 * gamma4 + strength**2 * block_shift
        )
        even[:, index, :, 0] = (
            strength**2 * delta2 + strength**4 * delta4 + strength**2 * block_shift
        )
        odd[:, index, :, 3] = strength**3 * (0.01 + 0.02j)
    return strengths, harmonics, odd, even


def test_analysis_separates_targets_ranges_and_harmonic_roles():
    strengths, harmonics, odd, even = _synthetic_contrasts()
    result = analyze_strength_series(
        range(100, 108), strengths, harmonics, odd, even
    )

    assert result["block_count"] == 8
    assert "asymptotic_range" in result["range_semantics"]
    targets = result["target_series"]
    assert targets["odd_fundamental"]["raw_expansion"].startswith("h * beta1")
    prefix = targets["odd_fundamental"]["nested_prefix_power_fits"]["2.0"]
    np.testing.assert_allclose(
        prefix["coefficient_summaries"][0]["mean_real"], [1.0, 2.0, -1.0]
    )
    adjacent = targets["odd_fundamental"][
        "adjacent_strength_normalized_changes"
    ]["1.0_to_2.0"]
    np.testing.assert_allclose(
        adjacent["mean_real"], 3 * np.array([0.0, -0.1, 0.04]), atol=1e-14
    )
    assert result["harmonic_diagnostics"]["odd"]["3"]["role"] == (
        "allowed_higher_harmonic"
    )
    assert result["harmonic_diagnostics"]["odd"]["2"]["role"] == (
        "parity_forbidden_harmonic"
    )
    assert result["harmonic_diagnostics"]["even"]["2"]["role"] == "target"
    assert result["response_normalization"]["gamma2"].endswith("-4 * gamma2")


def _synthetic_study_data():
    strengths, harmonics, odd, even = _synthetic_contrasts()
    n_block, n_strength, n_state, n_harmonic = odd.shape
    n_cycle = 2
    unforced = np.ones((n_block, n_state, n_cycle, n_harmonic), dtype=complex)
    positive_mean = unforced[:, None, :, 0, :] + odd + even
    negative_mean = unforced[:, None, :, 0, :] - odd + even
    positive = np.repeat(positive_mean[:, :, :, None, :], n_cycle, axis=3)
    negative = np.repeat(negative_mean[:, :, :, None, :], n_cycle, axis=3)
    return StrengthStudyData(
        block_ids=tuple(range(10, 18)),
        strengths=strengths,
        harmonics=harmonics,
        omega=2.0,
        n_phase=16,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=np.zeros((n_block, 3)),
        initial_states=np.ones((n_block, 3)),
        child_spawn_keys=tuple((value,) for value in range(10, 18)),
        generation_metadata={"root_entropy": 1},
    )


def test_study_analysis_preserves_crossed_contrasts():
    result = analyze_strength_study(_synthetic_study_data())
    assert result["block_ids"] == tuple(range(10, 18))
    odd_raw = result["target_series"]["odd_fundamental"]["raw_by_strength"][-1]
    np.testing.assert_allclose(odd_raw["strength"], 2.0)


def test_persistence_records_cycle_axes_and_hashes(tmp_path):
    data = _synthetic_study_data()
    derived = analyze_strength_study(data)
    config = {
        "study_id": "test",
        "protocol": {"omega": 2.0, "direction": [1, 0, 0]},
        "strengths": data.strengths.tolist(),
        "lorenz": {"sigma": 10, "rho": 28, "beta": 8 / 3},
        "solver": {"method": "DOP853"},
    }
    manifest = persist_strength_study(
        tmp_path / "run", data, derived, config, {"config_identifier": "test"}
    )
    with np.load(tmp_path / "run/raw_fourier_summaries.npz") as stored:
        np.testing.assert_array_equal(
            stored["positive_cycle_fourier"], data.positive_cycle_fourier
        )
        np.testing.assert_array_equal(stored["block_ids"], data.block_ids)
    loaded = json.loads((tmp_path / "run/manifest.json").read_text())
    assert loaded["classification"] == "exploratory"
    assert loaded["protocol_status"] == "provisional_method_validation_only"
    assert loaded["response_normalization"] == derived["response_normalization"]
    assert loaded["files"] == manifest["files"]
    assert all(value.startswith("sha256:") for value in manifest["files"].values())
