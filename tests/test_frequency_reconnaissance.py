import numpy as np
import pytest

from lorenz.frequency_reconnaissance import (
    FrequencyReconData,
    _multi_frequency_bootstrap,
    _normalized_target,
    observation_cycle_count,
)
from lorenz.strength_identifiability import StrengthStudyData


STRENGTHS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
HARMONICS = np.arange(6)
NULLS = {
    "odd_fundamental": [False, False, True],
    "even_second_harmonic": [True, True, False],
    "even_dc": [True, True, False],
}


def _study(omega, n_block=16):
    rng = np.random.default_rng(int(omega * 100))
    n_cycle = 2
    shape = (n_block, len(STRENGTHS), 3, n_cycle, len(HARMONICS))
    unforced = np.zeros((n_block, 3, n_cycle, len(HARMONICS)), dtype=complex)
    positive = np.zeros(shape, dtype=complex)
    negative = np.zeros(shape, dtype=complex)
    noise = 0.001 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    positive += noise
    negative -= noise
    for index, strength in enumerate(STRENGTHS):
        odd = strength * np.array([1 + 0.5j, 0.7 - 0.2j, 0])
        second = strength**2 * np.array([0, 0, 0.8 + 0.1j])
        dc = strength**2 * np.array([0, 0, 0.6])
        positive[:, index, :, :, 1] += odd[None, :, None]
        negative[:, index, :, :, 1] -= odd[None, :, None]
        positive[:, index, :, :, 2] += second[None, :, None]
        negative[:, index, :, :, 2] += second[None, :, None]
        positive[:, index, :, :, 0] += dc[None, :, None]
        negative[:, index, :, :, 0] += dc[None, :, None]
    return StrengthStudyData(
        block_ids=tuple(range(n_block)),
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        omega=omega,
        n_phase=32,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=np.zeros((n_block, 3)),
        initial_states=np.ones((n_block, 3)),
        child_spawn_keys=tuple((value,) for value in range(n_block)),
        generation_metadata={},
    )


def test_observation_rule_uses_both_time_and_cycle_floors():
    assert observation_cycle_count(0.5, 64, 200) == 64
    assert observation_cycle_count(2.0, 64, 200) == 64
    assert observation_cycle_count(4.0, 64, 200) == 128
    assert observation_cycle_count(8.0, 64, 200) == 255
    with pytest.raises(ValueError, match="positive"):
        observation_cycle_count(0, 64, 200)


def test_confirmed_fourier_normalizations_are_applied():
    values = np.array([1 + 2j])
    np.testing.assert_allclose(_normalized_target("odd_fundamental", values, 2), -2 + 1j)
    np.testing.assert_allclose(_normalized_target("even_second_harmonic", values, 2), -1 - 2j)
    np.testing.assert_allclose(_normalized_target("even_dc", np.array([4.0]), 2), 1)


def test_frequency_family_is_one_explicit_concatenated_max_family():
    studies = {0.5: _study(0.5), 1.0: _study(1.0)}
    data = FrequencyReconData(
        block_ids=tuple(range(16)),
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        frequencies=(0.5, 1.0),
        studies=studies,
        frequency_runtime_seconds={0.5: 1.0, 1.0: 1.0},
        source={
            0.5: {"new_runtime_seconds": 1.0, "equivalent_runtime_seconds": 1.0},
            1.0: {"new_runtime_seconds": 1.0, "equivalent_runtime_seconds": 1.0},
        },
    )
    result = _multi_frequency_bootstrap(
        data,
        {
            "bootstrap": {
                "confidence": 0.95,
                "resamples": 99,
                "root_entropy": 12,
                "batch_size": 17,
                "higher_order_fraction_limit": 0.2,
                "structural_null_outputs": NULLS,
            }
        },
    )

    assert result["family"]["scalar_coordinate_count"] == 2 * 210
    assert len(result["family"]["members"]) == 2 * 210
    assert result["coverage"]["resampling_unit"].startswith("whole initial-state block")
    assert set(result["by_frequency"]) == {"0.5", "1.0"}


def test_frequency_family_rejects_misaligned_block_rows():
    first = _study(0.5)
    second = _study(1.0)
    second = StrengthStudyData(
        **{**second.__dict__, "block_ids": tuple(reversed(second.block_ids))}
    )
    data = FrequencyReconData(
        block_ids=first.block_ids,
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        frequencies=(0.5, 1.0),
        studies={0.5: first, 1.0: second},
        frequency_runtime_seconds={0.5: 1.0, 1.0: 1.0},
        source={},
    )
    with pytest.raises(ValueError, match="block IDs/order"):
        _multi_frequency_bootstrap(
            data,
            {
                "bootstrap": {
                    "confidence": 0.95,
                    "resamples": 9,
                    "root_entropy": 12,
                    "batch_size": 5,
                    "higher_order_fraction_limit": 0.2,
                    "structural_null_outputs": NULLS,
                }
            },
        )
