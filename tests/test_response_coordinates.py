"""Independent trigonometric/DFT identities catch convention and phase errors."""
import numpy as np
from analysis.response_coordinates import output_coordinates, phase_relative_to_sine, circular_interval
from lorenz.response import phase_fourier


def test_sine_driven_time_signal_recovers_both_susceptibilities_with_phase_origin():
    h, l, k, d = 2.3, .18 - .07j, -.03 + .04j, .006
    theta = .713 + np.arange(128) * 2 * np.pi / 128
    # Construct independently in amplitude/phase form rather than the helper formula.
    linear = h * abs(l) * np.sin(theta + np.angle(l))
    quadratic = .5 * h ** 2 * abs(k) * np.sin(2 * theta + np.angle(k) - np.pi / 2)
    signal = linear + quadratic + h ** 2 * d
    z = phase_fourier(signal[None, None, :], [0, 1, 2], .713)[0, 0]
    np.testing.assert_allclose([2j * z[1] / h, -4 * z[2] / h ** 2, z[0] / h ** 2], [l, k, d], atol=2e-16)
    c1, s1, _ = output_coordinates(l, h, "L1")
    c2, s2, dc = output_coordinates(k, h, "K2")
    _, _, dc = output_coordinates(d, h, "D0")
    np.testing.assert_allclose(c1 * np.cos(theta) + s1 * np.sin(theta) + c2 * np.cos(2 * theta) + s2 * np.sin(2 * theta) + dc, signal, atol=4e-16)


def test_cosine_input_rotates_outputs_by_harmonic_order_not_one_common_angle():
    h, l, k = 1.7, .12 + .09j, .03 - .02j
    theta = .37 + np.arange(128) * 2 * np.pi / 128
    output = h * (l.real * np.cos(theta) - l.imag * np.sin(theta)) + .5 * h ** 2 * (k.real * np.cos(2 * theta) - k.imag * np.sin(2 * theta))
    z = phase_fourier(output[None, None, :], [0, 1, 2], .37)[0, 0]
    np.testing.assert_allclose([2 * z[1] / h, 4 * z[2] / h ** 2], [l, k], atol=2e-16)
    assert abs(np.angle(np.exp(1j * (phase_relative_to_sine(k, "K2") - np.angle(k) + np.pi / 2)))) < 1e-15


def test_phase_interval_crossing_branch_cut_is_local_not_a_360_degree_span():
    samples = np.deg2rad(np.array([178., 179., -180., -179., -178.]))
    interval = circular_interval(np.pi, samples)
    assert np.rad2deg(interval[1] - interval[0]) < 5
