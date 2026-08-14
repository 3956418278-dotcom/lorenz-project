import numpy as np

from lorenz.strength_study import (
    FrequencyStudyData,
    StrengthStudyData,
    dc_components,
    second_harmonic_estimators,
)
from lorenz.variance_reduction import (
    _bootstrap_indices,
    analyze_dc_variance,
    analyze_second_identification,
    analyze_second_variance_ratios,
    analyze_unforced_second_diagnostic,
)


STRENGTHS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
HARMONICS = np.arange(6)


def _study(n_block=48):
    rng = np.random.default_rng(4)
    n_cycle = 2
    shape = (n_block, len(STRENGTHS), 3, n_cycle, len(HARMONICS))
    positive = np.zeros(shape, dtype=complex)
    negative = np.zeros(shape, dtype=complex)
    unforced = np.zeros((n_block, 3, n_cycle, len(HARMONICS)), dtype=complex)
    forced_noise = rng.normal(size=(n_block, len(STRENGTHS), 3, 1)) + 1j * rng.normal(
        size=(n_block, len(STRENGTHS), 3, 1)
    )
    baseline_noise = rng.normal(size=(n_block, 3, 1)) + 1j * rng.normal(
        size=(n_block, 3, 1)
    )
    for strength_index, strength in enumerate(STRENGTHS):
        signal = strength**2 * np.array([0, 0, 0.3 + 0.2j])
        positive[:, strength_index, :, :, 2] = (
            signal[None, :, None] + 0.2 * forced_noise[:, strength_index]
        )
        negative[:, strength_index, :, :, 2] = (
            signal[None, :, None] + 0.2 * forced_noise[:, strength_index]
        )
    unforced[:, :, :, 2] = 0.5 * baseline_noise
    dc_forced_noise = 0.2 * rng.normal(size=(n_block, len(STRENGTHS), 3, 1))
    dc_baseline_noise = 0.5 * rng.normal(size=(n_block, 3, 1))
    positive[..., 0] = 3.0 + dc_forced_noise
    negative[..., 0] = 3.0 + dc_forced_noise
    unforced[..., 0] = 2.0 + dc_baseline_noise
    return StrengthStudyData(
        block_ids=tuple(range(n_block)),
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        omega=2.0,
        n_phase=32,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=np.zeros((n_block, 3)),
        initial_states=np.ones((n_block, 3)),
        child_spawn_keys=tuple((value,) for value in range(n_block)),
        generation_metadata={},
    )


def test_second_estimators_differ_only_by_retained_unforced_diagnostic():
    values = second_harmonic_estimators(_study())
    np.testing.assert_allclose(
        values["current_E2"],
        values["alternative_A2"] - values["unforced_U2_diagnostic"][:, None],
    )


def test_dc_subtraction_cannot_omit_nonzero_unforced_mean():
    values = dc_components(_study())
    np.testing.assert_allclose(
        values["current_E0"],
        values["forced_even_A0"] - values["unforced_U0"][:, None],
    )
    assert np.allclose(values["forced_even_A0"].mean(), 3.0, atol=0.1)
    assert np.allclose(values["unforced_U0"].mean(), 2.0, atol=0.1)


def test_heldout_variance_family_confirms_fixed_lower_variance_alternative():
    study = _study()
    data = FrequencyStudyData(
        block_ids=study.block_ids,
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        frequencies=(2.0,),
        studies={2.0: study},
        frequency_runtime_seconds={2.0: 1.0},
        source={},
    )
    config = {
        "confidence": 0.95,
        "resamples": 499,
        "root_entropy": 10,
        "batch_size": 31,
    }
    indices, _ = _bootstrap_indices(len(study.block_ids), config)
    result = analyze_second_variance_ratios(data, config, indices)
    allowed = [entry for entry in result["entries"] if not entry["structural_null"]]
    assert all(entry["alternative_to_current_variance_ratio"] < 0.5 for entry in allowed)
    assert result["allowed_component_summary"]["all_upper_bounds_below_one"]


def test_identification_and_dc_families_keep_shared_block_as_unit():
    study = _study()
    data = FrequencyStudyData(
        block_ids=study.block_ids,
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        frequencies=(2.0,),
        studies={2.0: study},
        frequency_runtime_seconds={2.0: 1.0},
        source={},
    )
    bootstrap = {
        "confidence": 0.95,
        "resamples": 199,
        "root_entropy": 10,
        "batch_size": 31,
        "higher_order_fraction_limit": 0.2,
    }
    identification = analyze_second_identification(data, bootstrap)
    assert identification["coverage"]["scalar_coordinate_count"] == 168
    assert identification["coverage"]["resampling_unit"].startswith("whole held-out")
    indices, _ = _bootstrap_indices(len(study.block_ids), bootstrap)
    dc = analyze_dc_variance(
        data,
        {
            "confidence": 0.95,
            "batch_size": 31,
            "independent_baseline_variance_fraction": 0.1,
        },
        indices,
    )
    assert dc["coverage"]["coordinate_count"] == 54
    assert len(dc["entries"]) == 18
    unforced = analyze_unforced_second_diagnostic(data, bootstrap)
    assert unforced["coverage"]["scalar_coordinate_count"] == 6
    assert len(unforced["entries"]) == 3
