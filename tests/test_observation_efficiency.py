import numpy as np

from lorenz.observation_efficiency import (
    _cycle_family,
    _target_cycles,
    _window_statistics,
    analyze_observation_efficiency,
)
from lorenz.strength_bootstrap import build_decision_family


STRENGTHS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
NULLS = {
    "odd_fundamental": [False, False, True],
    "even_second_harmonic": [True, True, False],
    "even_dc": [True, True, False],
}


def _synthetic_raw(n_block=32, n_cycle=64):
    rng = np.random.default_rng(17)
    shape = (n_block, len(STRENGTHS), 3, n_cycle)
    odd = (
        STRENGTHS[None, :, None, None]
        * np.array([1.0, 0.7, 0.0])[None, None, :, None]
        + STRENGTHS[None, :, None, None] ** 3
        * np.array([0.005, 0.004, 0.0])[None, None, :, None]
        + 0.1 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    )
    second = (
        STRENGTHS[None, :, None, None] ** 2
        * np.array([0.0, 0.0, 0.5])[None, None, :, None]
        + 0.1 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    )
    dc = (
        STRENGTHS[None, :, None, None] ** 2
        * np.array([0.0, 0.0, 0.4])[None, None, :, None]
        + 0.1 * rng.normal(size=shape)
    )
    positive = np.zeros(shape + (3,), dtype=complex)
    negative = np.zeros_like(positive)
    positive[..., 1] = odd
    negative[..., 1] = -odd
    positive[..., 2] = second
    negative[..., 2] = second
    positive[..., 0] = dc
    negative[..., 0] = dc
    return {
        "block_ids": np.arange(n_block),
        "strengths": STRENGTHS,
        "harmonics": np.array([0, 1, 2]),
        "positive_cycle_fourier": positive,
        "negative_cycle_fourier": negative,
        "unforced_cycle_fourier": np.zeros((n_block, 3, n_cycle, 3), complex),
    }


def _config():
    return {
        "parent_artifact": "test",
        "cycle_lengths": [1, 2, 4, 8, 16, 32, 64],
        "bootstrap_cycle_lengths": [8, 64],
        "projection_cycle_options": [128, 256],
        "projection_candidates": [
            {"target": "odd_fundamental", "prefix_upper_strength": 4.0}
        ],
        "discard_time": 160.0,
        "n_phase": 32,
        "protocol": {"omega": 2.0},
        "bootstrap": {
            "confidence": 0.95,
            "resamples": 99,
            "root_entropy": 4,
            "batch_size": 25,
            "higher_order_fraction_limit": 0.2,
            "structural_null_outputs": NULLS,
        },
    }


def test_cycle_family_average_equals_family_of_cycle_average():
    raw = _synthetic_raw()
    strengths, cycles = _target_cycles(raw)
    matrix, features, groups = _cycle_family(cycles, strengths, NULLS)
    direct, direct_features, direct_groups, _ = build_decision_family(
        {name: values.mean(axis=-1) for name, values in cycles.items()},
        strengths,
        NULLS,
    )

    np.testing.assert_allclose(matrix.mean(axis=1), direct)
    assert features == direct_features
    assert groups == direct_groups


def test_windows_remain_inside_blocks():
    raw = _synthetic_raw()
    strengths, cycles = _target_cycles(raw)
    matrix, _, groups = _cycle_family(cycles, strengths, NULLS)
    summary = _window_statistics(matrix, 8, 3.5, groups)

    assert matrix.shape == (32, 64, 210)
    assert summary["window_count"] == 8
    assert summary["adjacent_window_drift"]["pair_count"] == 7
    assert summary["median_cross_block_variance_realified"].shape == (210,)


def test_analysis_estimates_variance_scaling_without_cycle_replications():
    result = analyze_observation_efficiency(
        _synthetic_raw(),
        _config(),
        {
            "confirmed_block_bootstrap": {"coverage": {"critical_value": 3.5}},
            "provenance": {"runtime_seconds_before_persistence": 100.0},
        },
    )

    assert result["family"]["block_count"] == 32
    assert result["family"]["cycle_count"] == 64
    median_alpha = result["variance_scaling"]["alpha_quantiles_all"][3]
    assert 0.7 < median_alpha < 1.3
    assert set(result["nested_prefix_bootstrap"]) == {"8", "64"}
    projection = result["adequacy_resource_projection"]["candidates"][
        "odd_fundamental"
    ]["4.0"]
    assert set(projection["projected_upper_ratio_by_cycles"]) == {"128", "256"}
    assert result["resource_options"]["by_cycle_count"]["256"][
        "projected_wall_seconds_from_parent_runtime"
    ] > 100
