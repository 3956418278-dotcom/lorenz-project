import numpy as np
import pytest

from lorenz.response import reconstruct_linear_tensor, reconstruct_monochromatic_quadratic_tensor
from lorenz.statistics import (
    BlockFourierContrasts,
    BlockFrequencyResponses,
    finite_strength_block_responses,
    paired_block_fourier_contrasts,
    realify_block_values,
    reconstruct_block_tensors,
    restore_block_values,
    summarize_block_responses,
)


def test_raw_contrasts_become_finite_strength_responses_per_block():
    odd = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    even_second = np.array([[2 - 1j, 4 - 3j], [6 - 5j, 8 - 7j]])
    even_dc = np.array([[1.0, 2.0], [3.0, 4.0]])
    contrasts = BlockFourierContrasts(
        block_ids=(10, 11),
        odd_fundamental=odd,
        even_second_harmonic=even_second,
        even_dc=even_dc,
    )

    responses = finite_strength_block_responses(contrasts, strength=0.5)

    np.testing.assert_allclose(responses.first_order, 4j * odd)
    np.testing.assert_allclose(responses.second_harmonic, -16 * even_second)
    np.testing.assert_allclose(responses.rectification, 4 * even_dc)
    assert responses.first_order.shape == (2, 2)
    assert responses.block_ids == (10, 11)


def test_realification_is_bijective_with_documented_ordering():
    values = np.array(
        [
            [[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]],
            [[9 + 13j, 10 + 14j], [11 + 15j, 12 + 16j]],
        ]
    )

    realified = realify_block_values(values)

    np.testing.assert_array_equal(
        realified[0], [1, 2, 3, 4, 5, 6, 7, 8]
    )
    np.testing.assert_array_equal(
        restore_block_values(realified, (2, 2), complex_valued=True), values
    )


def test_joint_summary_returns_sample_and_mean_covariance():
    responses = BlockFrequencyResponses(
        block_ids=(3, 4, 5),
        first_order=np.array([1 + 4j, 2 + 5j, 4 + 9j]),
        second_harmonic=np.array([10 - 2j, 13 - 1j, 15 + 3j]),
        rectification=np.array([2.0, 8.0, 5.0]),
    )
    samples = np.array(
        [
            [1, 4, 10, -2, 2],
            [2, 5, 13, -1, 8],
            [4, 9, 15, 3, 5],
        ],
        dtype=float,
    )

    summary = summarize_block_responses(responses)

    expected_sample_covariance = np.cov(samples, rowvar=False, ddof=1)
    np.testing.assert_allclose(summary.realified_mean, samples.mean(axis=0))
    np.testing.assert_allclose(
        summary.sample_covariance, expected_sample_covariance
    )
    np.testing.assert_allclose(
        summary.covariance_of_mean, expected_sample_covariance / 3
    )
    np.testing.assert_allclose(summary.mean.first_order, (7 + 18j) / 3)
    assert summary.block_ids == (3, 4, 5)
    assert summary.features[0].real == slice(0, 1)
    assert summary.features[0].imaginary == slice(1, 2)
    assert summary.features[2].imaginary is None


def test_tensor_reconstruction_is_per_block_and_precedes_summary():
    directions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    first = np.arange(24, dtype=float).reshape(3, 4, 2)
    first = first + 1j * (first + 0.5)
    second = np.arange(24, 48, dtype=float).reshape(3, 4, 2)
    second = second - 1j * (second / 3)
    dc = np.arange(48, 72, dtype=float).reshape(3, 4, 2)
    directional = BlockFrequencyResponses((20, 21, 22), first, second, dc)

    tensors = reconstruct_block_tensors(directions, directional)

    expected_first = np.stack(
        [reconstruct_linear_tensor(directions, block) for block in first]
    )
    expected_second = np.stack(
        [
            reconstruct_monochromatic_quadratic_tensor(directions, block)
            for block in second
        ]
    )
    expected_dc = np.stack(
        [
            reconstruct_monochromatic_quadratic_tensor(directions, block)
            for block in dc
        ]
    )
    np.testing.assert_allclose(tensors.first_order, expected_first)
    np.testing.assert_allclose(tensors.second_harmonic, expected_second)
    np.testing.assert_allclose(tensors.rectification, expected_dc)

    summary = summarize_block_responses(tensors)
    assert summary.mean.first_order.shape == (2, 2)
    assert summary.mean.second_harmonic.shape == (2, 2, 2)
    assert summary.mean.rectification.shape == (2, 2, 2)
    assert tensors.block_ids == directional.block_ids


def test_covariance_requires_two_blocks():
    responses = BlockFrequencyResponses(
        block_ids=(0,),
        first_order=np.array([1 + 2j]),
        second_harmonic=np.array([3 + 4j]),
        rectification=np.array([5.0]),
    )

    with pytest.raises(ValueError, match="at least two blocks"):
        summarize_block_responses(responses)


@pytest.mark.parametrize(
    "function",
    [
        lambda: finite_strength_block_responses(
            BlockFourierContrasts(
                block_ids=(0, 1),
                odd_fundamental=np.ones(2),
                even_second_harmonic=np.ones(2),
                even_dc=np.array([1 + 0j, 2 + 0.1j]),
            ),
            1.0,
        ),
        lambda: summarize_block_responses(
            BlockFrequencyResponses(
                block_ids=(0, 1),
                first_order=np.ones(2),
                second_harmonic=np.ones(2),
                rectification=np.array([1 + 0j, 2 + 0.1j]),
            )
        ),
    ],
)
def test_rectification_rejects_nonzero_imaginary_part(function):
    with pytest.raises(ValueError, match="real-valued"):
        function()


def test_block_adapter_rejects_one_unlabeled_state_cycle_phase_array():
    one_block = np.zeros((3, 4, 8))

    with pytest.raises(ValueError, match="block, .*cycle, phase"):
        paired_block_fourier_contrasts(
            (0,), one_block, one_block, one_block
        )


def test_block_adapter_matches_explicit_cycle_means_and_preserves_axes():
    rng = np.random.default_rng(4)
    positive = rng.normal(size=(2, 3, 4, 16))
    negative = rng.normal(size=(2, 3, 4, 16))
    unforced = rng.normal(size=(2, 3, 4, 16))

    actual = paired_block_fourier_contrasts(
        (7, 9), positive, negative, unforced, phase_offset=0.2
    )
    from lorenz.response import directional_fourier_contrasts

    expected = directional_fourier_contrasts(
        positive.mean(axis=-2),
        negative.mean(axis=-2),
        unforced.mean(axis=-2),
        phase_offset=0.2,
    )

    assert actual.block_ids == (7, 9)
    assert actual.odd_fundamental.shape == (2, 3)
    np.testing.assert_allclose(actual.odd_fundamental, expected.odd_fundamental)
    np.testing.assert_allclose(
        actual.even_second_harmonic, expected.even_second_harmonic
    )
    np.testing.assert_allclose(actual.even_dc, expected.even_dc)
