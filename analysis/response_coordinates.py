"""Translate stored Fourier/response coordinates to physical sinusoidal output.

No integration or estimation happens here. The sine-input convention and
factor 1/2 at the second harmonic follow standard Fourier projection conventions.
"""
from __future__ import annotations
import numpy as np


def output_coordinates(coefficient, strength, quantity):
    """Return (cos amplitude, sin amplitude, signed DC) for one parity channel."""
    z = np.asarray(coefficient)
    h = np.asarray(strength)
    if quantity == "L1":
        return h * z.imag, h * z.real, np.zeros_like(z.real)
    if quantity == "K2":
        return -.5 * h ** 2 * z.real, .5 * h ** 2 * z.imag, np.zeros_like(z.real)
    if quantity == "D0":
        return np.zeros_like(z.real), np.zeros_like(z.real), h ** 2 * z.real
    raise ValueError("Expected L1, K2 or D0")


def phase_relative_to_sine(coefficient, quantity):
    """Angle in radians in A*sin(n*theta+delta); undefined at zero amplitude."""
    z = np.asarray(coefficient)
    if quantity == "D0":
        raise ValueError("DC has no sinusoidal phase")
    shift = 0 if quantity == "L1" else -np.pi / 2
    angle = np.angle(z) + shift
    return np.angle(np.exp(1j * angle))


def circular_interval(point_angle, sample_angles, coverage=.95):
    """Local circular percentile interval about a chosen angle, not a null test."""
    delta = np.angle(np.exp(1j * (sample_angles - point_angle)))
    tail = (1 - coverage) / 2
    return point_angle + np.quantile(delta, [tail, 1 - tail])


def coordinate_summary(coefficient, bootstrap, strength, quantity):
    """Preserve signed Cartesian quantities; flag phases near the origin."""
    z, b = complex(coefficient), np.asarray(bootstrap)
    h = float(strength)
    c, s, dc = output_coordinates(z, h, quantity)
    bc, bs, bdc = output_coordinates(b, h, quantity)
    row = {"coefficient_re": z.real, "coefficient_im": z.imag,
        "output_cos": float(c), "output_sin": float(s), "output_dc": float(dc),
        "output_cos_ci95_low": float(np.quantile(bc, .025)), "output_cos_ci95_high": float(np.quantile(bc, .975)),
        "output_sin_ci95_low": float(np.quantile(bs, .025)), "output_sin_ci95_high": float(np.quantile(bs, .975)),
        "output_dc_ci95_low": float(np.quantile(bdc, .025)), "output_dc_ci95_high": float(np.quantile(bdc, .975))}
    if quantity == "D0":
        return {**row, "fourier_re": float(dc), "fourier_minus_im": 0., "peak_amplitude": None,
            "gain_per_h_power": z.real, "gain_ci95_low": float(np.quantile(b.real, .025)), "gain_ci95_high": float(np.quantile(b.real, .975)),
            "peak_ci95_low": None, "peak_ci95_high": None,
            "coefficient_angle_deg": None, "phase_relative_sine_deg": None, "phase_ci95_low_deg": None, "phase_ci95_high_deg": None,
            "phase_status": "DC: no phase", "phase_gate": "not applicable"}
    factor = 1 if quantity == "L1" else .5
    power = 1 if quantity == "L1" else 2
    amplitude = h ** power * factor * abs(z)
    amplitudes = h ** power * factor * abs(b)
    gain = factor * abs(z)
    # A Bonferroni coordinate rectangle is only a phase-reporting guard;
    # it is not simultaneous inference over channels or a magnitude null test.
    re = np.quantile(b.real, [.0125, .9875])
    im = np.quantile(b.imag, [.0125, .9875])
    near_origin = re[0] <= 0 <= re[1] and im[0] <= 0 <= im[1]
    angle = float(phase_relative_to_sine(z, quantity))
    phase_samples = phase_relative_to_sine(b, quantity)
    interval = circular_interval(angle, phase_samples)
    return {**row, "fourier_re": float(c) / 2, "fourier_minus_im": float(s) / 2,
        "peak_amplitude": amplitude, "gain_per_h_power": gain,
        "gain_ci95_low": float(np.quantile(factor * abs(b), .025)), "gain_ci95_high": float(np.quantile(factor * abs(b), .975)),
        "peak_ci95_low": float(np.quantile(amplitudes, .025)), "peak_ci95_high": float(np.quantile(amplitudes, .975)),
        "coefficient_angle_deg": float(np.angle(z, deg=True)), "phase_relative_sine_deg": np.rad2deg(angle),
        "phase_ci95_low_deg": None if near_origin else float(np.rad2deg(interval[0])),
        "phase_ci95_high_deg": None if near_origin else float(np.rad2deg(interval[1])),
        "phase_status": "unresolved near origin; do not interpret point angle" if near_origin else "locally resolved under pointwise coordinate guard",
        "phase_gate": "origin outside nominal95% two-coordinate bootstrap rectangle; no across-channel coverage; magnitude intervals are not detection tests"}
