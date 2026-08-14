import numpy as np
import pytest

from lorenz.strength_bootstrap import (
    _bootstrap_max_statistic,
    analyze_block_bootstrap,
    build_decision_family,
)


STRENGTHS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
NULLS = {
    "odd_fundamental": np.array([False, False, True]),
    "even_second_harmonic": np.array([True, True, False]),
    "even_dc": np.array([True, True, False]),
}


def _synthetic_targets(n_block=64, high_scale=0.005, zero_low=False):
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(n_block, len(STRENGTHS), 3))
    complex_noise = noise + 1j * rng.normal(size=noise.shape)
    low = 0.0 if zero_low else 1.0
    odd = (
        STRENGTHS[None, :, None] * np.array([low, 0.7, 0.0])
        + STRENGTHS[None, :, None] ** 3
        * np.array([high_scale, high_scale, 0.0])
        + 0.002 * complex_noise
    )
    second = (
        STRENGTHS[None, :, None] ** 2 * np.array([0.0, 0.0, low])
        + STRENGTHS[None, :, None] ** 4 * np.array([0.0, 0.0, high_scale])
        + 0.002 * complex_noise
    )
    dc = (
        STRENGTHS[None, :, None] ** 2 * np.array([0.0, 0.0, low])
        + STRENGTHS[None, :, None] ** 4 * np.array([0.0, 0.0, high_scale])
        + 0.002 * noise
    )
    return {
        "odd_fundamental": odd,
        "even_second_harmonic": second,
        "even_dc": dc,
    }


def test_decision_family_declares_all_210_scalar_coordinates():
    matrix, features, groups, masks = build_decision_family(
        _synthetic_targets(), STRENGTHS, NULLS
    )

    assert matrix.shape == (64, 210)
    assert len(features) == 210
    assert len(groups) == 126
    assert sum(feature.section == "identification" for feature in features) == 90
    assert sum(feature.section == "adequacy" for feature in features) == 120
    np.testing.assert_array_equal(masks["even_dc"], [True, True, False])


def test_bootstrap_uses_one_index_draw_for_joint_columns():
    values = np.arange(20, dtype=float)
    samples = np.column_stack((values, 10 * values))
    mean, standard_error, statistics, metadata = _bootstrap_max_statistic(
        samples, resamples=199, root_entropy=123, batch_size=17
    )

    np.testing.assert_allclose(mean, [9.5, 95.0])
    np.testing.assert_allclose(standard_error[1], 10 * standard_error[0])
    assert np.isfinite(statistics).all()
    assert metadata["bit_generator"] == "PCG64DXSM"


def test_simultaneous_analysis_separates_identification_and_adequacy():
    result = analyze_block_bootstrap(
        range(64),
        STRENGTHS,
        _synthetic_targets(high_scale=0.005),
        NULLS,
        confidence=0.95,
        resamples=499,
        root_entropy=1234,
        batch_size=64,
        higher_order_fraction_limit=0.2,
    )

    assert result["coverage"]["confidence"] == 0.95
    assert result["family"]["scalar_coordinate_count"] == 210
    assert result["identification"]["odd_fundamental"]["4.0"]["identified"]
    assert result["adequacy"]["odd_fundamental"]["4.0"]["adequate"]
    assert result["identification_window"]["odd_fundamental"]["4.0"] == {
        "identified": True,
        "adequate": True,
        "in_identification_window": True,
    }
    assert not result["null_control_exclusions"]


def test_unresolved_allowed_denominator_never_forms_ratio():
    targets = _synthetic_targets(zero_low=True, high_scale=0.0)
    # Make all allowed components exact zero while retaining noisy null controls.
    targets["odd_fundamental"][..., :2] = 0
    targets["even_second_harmonic"][..., 2] = 0
    targets["even_dc"][..., 2] = 0
    result = analyze_block_bootstrap(
        range(64),
        STRENGTHS,
        targets,
        NULLS,
        confidence=0.95,
        resamples=99,
        root_entropy=7,
        batch_size=25,
        higher_order_fraction_limit=0.2,
    )

    decision = result["adequacy"]["even_second_harmonic"]["4.0"]
    assert decision["denominator_status"] == "unresolved"
    assert decision["higher_to_dominant_low_upper"] is None
    assert decision["adequate"] is False


def test_null_masks_must_leave_an_allowed_output():
    invalid = dict(NULLS)
    invalid["even_dc"] = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match="retain at least one"):
        build_decision_family(_synthetic_targets(), STRENGTHS, invalid)
