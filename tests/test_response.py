import numpy as np
import pytest

from lorenz.response import (
    directional_fourier_contrasts,
    directional_frequency_response,
    effective_phase_responses,
    monochromatic_quadratic_design,
    paired_order_contrasts,
    phase_fourier,
    reconstruct_linear_tensor,
    reconstruct_monochromatic_quadratic_tensor,
)


def test_phase_fourier_recovers_dc_fundamental_and_second_harmonic():
    n_phase = 32
    theta = 2 * np.pi * np.arange(n_phase) / n_phase
    values = 1.5 + 2.0 * np.cos(theta) - 3.0 * np.sin(2 * theta)

    coefficients = phase_fourier(values, [0, 1, 2, 3])

    np.testing.assert_allclose(coefficients[0], 1.5, atol=1e-14)
    np.testing.assert_allclose(coefficients[1], 1.0, atol=1e-14)
    np.testing.assert_allclose(coefficients[2], 1.5j, atol=1e-14)
    np.testing.assert_allclose(coefficients[3], 0.0, atol=1e-14)


def test_paired_contrasts_isolate_linear_and_quadratic_terms():
    baseline = np.array([10.0, -4.0])
    linear = np.array([2.0, 3.0])
    quadratic = np.array([-1.0, 5.0])
    strength = 0.2
    positive = baseline + strength * linear + strength**2 * quadratic
    negative = baseline - strength * linear + strength**2 * quadratic

    odd, even = paired_order_contrasts(positive, negative, baseline)
    first, second = effective_phase_responses(odd, even, strength)

    np.testing.assert_allclose(first, linear)
    np.testing.assert_allclose(second, quadratic)


def test_extracts_susceptibilities_with_sine_and_no_factorial_convention():
    n_phase = 32
    theta = 2 * np.pi * np.arange(n_phase) / n_phase
    strength = 0.2
    baseline = np.array([3.0, -2.0])[:, None]
    chi1 = np.array([1.5 - 0.75j, -2.0 + 3.0j])
    chi2 = np.array([-4.0 + 2.0j, 1.0 - 0.5j])
    rectification = np.array([0.25, -1.25])

    first_phase = 2 * np.real(
        (chi1 / (2j))[:, None] * np.exp(1j * theta)[None, :]
    )
    second_phase = rectification[:, None] + 2 * np.real(
        (-chi2 / 4)[:, None] * np.exp(2j * theta)[None, :]
    )
    positive = baseline + strength * first_phase + strength**2 * second_phase
    negative = baseline - strength * first_phase + strength**2 * second_phase
    unforced = np.broadcast_to(baseline, positive.shape)

    response = directional_frequency_response(
        positive, negative, unforced, strength
    )

    np.testing.assert_allclose(response.first_order, chi1, atol=1e-13)
    np.testing.assert_allclose(response.second_harmonic, chi2, atol=1e-13)
    np.testing.assert_allclose(response.rectification, rectification, atol=1e-13)
    np.testing.assert_allclose(response.first_phase, first_phase, atol=1e-13)
    np.testing.assert_allclose(response.second_phase, second_phase, atol=1e-13)


def test_nonzero_forcing_phase_uses_actual_phase_for_susceptibility():
    n_phase = 32
    phase_offset = 0.7
    theta = phase_offset + 2 * np.pi * np.arange(n_phase) / n_phase
    strength = 0.3
    chi1 = np.array([2.0 - 1.0j])
    chi2 = np.array([-0.5 + 3.0j])
    qdc = np.array([1.25])
    baseline = np.array([[4.0]])
    first_phase = 2 * np.real(
        (chi1 / (2j))[:, None] * np.exp(1j * theta)[None, :]
    )
    second_phase = qdc[:, None] + 2 * np.real(
        (-chi2 / 4)[:, None] * np.exp(2j * theta)[None, :]
    )
    positive = baseline + strength * first_phase + strength**2 * second_phase
    negative = baseline - strength * first_phase + strength**2 * second_phase
    unforced = np.broadcast_to(baseline, positive.shape)

    response = directional_frequency_response(
        positive, negative, unforced, strength, phase_offset
    )

    np.testing.assert_allclose(response.first_order, chi1, atol=1e-13)
    np.testing.assert_allclose(response.second_harmonic, chi2, atol=1e-13)
    np.testing.assert_allclose(response.rectification, qdc, atol=1e-13)


def test_raw_fourier_contrasts_preserve_strength_scaling_and_leading_axes():
    n_phase = 16
    theta = 2 * np.pi * np.arange(n_phase) / n_phase
    strengths = np.array([0.1, 0.2])[:, None, None]
    first_phase = np.sin(theta)[None, None, :]
    second_phase = (2.0 + np.cos(2 * theta))[None, None, :]
    baseline = np.full((2, 1, n_phase), 5.0)
    positive = baseline + strengths * first_phase + strengths**2 * second_phase
    negative = baseline - strengths * first_phase + strengths**2 * second_phase

    contrasts = directional_fourier_contrasts(positive, negative, baseline)

    np.testing.assert_allclose(
        contrasts.odd_fundamental[:, 0], strengths[:, 0, 0] / (2j), atol=1e-14
    )
    np.testing.assert_allclose(
        contrasts.even_second_harmonic[:, 0], strengths[:, 0, 0] ** 2 / 2,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        contrasts.even_dc[:, 0], 2 * strengths[:, 0, 0] ** 2, atol=1e-14
    )


def test_reconstructs_complex_linear_and_monochromatic_quadratic_tensors():
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ]
    )
    linear_base = np.arange(18, dtype=float).reshape(2, 3, 3) / 7
    linear = linear_base + 1j * np.flip(linear_base, axis=-1)
    quadratic_base = np.arange(54, dtype=float).reshape(2, 3, 3, 3) / 11
    quadratic = quadratic_base + 1j * np.flip(quadratic_base, axis=-1)
    quadratic = (quadratic + np.swapaxes(quadratic, -1, -2)) / 2

    linear_directional = np.einsum("do,xyo->dxy", directions, linear)
    quadratic_directional = np.einsum(
        "dj,xyjk,dk->dxy", directions, quadratic, directions
    )

    recovered_linear = reconstruct_linear_tensor(directions, linear_directional)
    recovered_quadratic = reconstruct_monochromatic_quadratic_tensor(
        directions, quadratic_directional
    )

    np.testing.assert_allclose(recovered_linear, linear, atol=1e-13)
    np.testing.assert_allclose(recovered_quadratic, quadratic, atol=1e-13)


def test_rank_deficient_direction_sets_are_rejected():
    directions = np.eye(3)
    responses = np.zeros((3, 2))

    with pytest.raises(ValueError, match="identifiable monochromatic input pairs"):
        reconstruct_monochromatic_quadratic_tensor(directions, responses)

    np.testing.assert_equal(
        monochromatic_quadratic_design(directions).shape, (3, 6)
    )


def test_noninteger_harmonic_is_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="contain integers"):
        phase_fourier(np.ones(8), [1.5])


def test_nonfinite_phase_offset_is_rejected():
    with pytest.raises(ValueError, match="phase_offset must be finite"):
        phase_fourier(np.ones(8), [1], np.nan)


@pytest.mark.parametrize("strength", [0.0, -1.0, np.inf, np.nan])
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="finite and positive"):
        effective_phase_responses(np.ones(2), np.ones(2), strength)
