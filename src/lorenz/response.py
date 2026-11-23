"""Numerical definitions and reconstruction of sinusoidal response tensors.

Scientific language used by the code
------------------------------------
The project starts from a real phase-dependent response but stores positive-
frequency Fourier coefficients.  For phase ``theta`` and harmonic ``n``,

    Z_n = mean_theta[x(theta) * exp(-1j*n*theta)].

For a real signal and ``n > 0`` write the time-domain harmonic as

    x_n(theta) = A_n*cos(n*theta) + B_n*sin(n*theta).

Then

    Z_n = A_n/2 - 1j*B_n/2.

Many analysis tables use the two real coordinates

    c_n = Re(Z_n),     s_n = -Im(Z_n),

and label them ``cos`` and ``sin``.  These labels are *coordinates of the
stored complex Fourier coefficient*, not the full time-domain amplitudes:
``A_n = 2*c_n`` and ``B_n = 2*s_n`` for ``n > 0``.  At ``n = 0`` there is no
factor of two: ``Z_0`` is the DC value itself.

The susceptibility language adds one more conversion because the forcing is
``h*a*sin(theta)`` rather than a cosine.  With the repository convention

    f_hat[1] = h*a/(2j),

we have, for one forcing direction ``a``,

    L_hat[1] = chi1(omega)@a / (2j),
    Q_hat[2] = -chi2(2*omega;omega,omega)[a,a] / 4,
    Q_hat[0] = Qdc(omega)[a,a].

Thus ``chi1@a = 2j*L_hat[1]`` and ``chi2[a,a] = -4*Q_hat[2]``.  If
``L_hat[1] = c1 - 1j*s1``, then ``Re(chi1@a)=2*s1`` and
``Im(chi1@a)=2*c1``: the sine-input convention rotates the first-order
cos/sin coordinates when translated to the usual complex susceptibility.
If ``Q_hat[2] = c2 - 1j*s2``, then ``Re(chi2[a,a])=-4*c2`` and
``Im(chi2[a,a])=4*s2``.

Finally, harmonic order and perturbation order are different notions.  The
integer ``n`` labels Fourier frequency ``n*omega``.  Odd/even powers of forcing
strength select perturbative parity.  At leading order the linear response is
at ``n=1`` while the quadratic response is at ``n=0`` and ``n=2``; higher
finite-strength orders can populate the same harmonics and additional ones.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionalFrequencyResponse:
    """Finite-strength responses for one frequency and forcing direction.

    ``first_order`` is chi1(omega) contracted with the forcing direction.
    ``second_harmonic`` is chi2(2*omega; omega, omega) contracted twice with
    that direction. ``rectification`` is the separately defined DC coefficient
    Qdc contracted twice with the direction.
    """

    first_order: np.ndarray
    second_harmonic: np.ndarray
    rectification: np.ndarray
    first_phase: np.ndarray
    second_phase: np.ndarray


@dataclass(frozen=True)
class DirectionalFourierContrasts:
    """Unscaled paired contrasts retained for cross-strength inference.

    ``odd_fundamental`` and ``even_second_harmonic`` are complex Fourier
    coefficients ``Z_n``.  They are not yet susceptibilities: strength
    normalization and the sine-input factors ``2j``/``-4`` are applied later.
    ``even_dc`` is already a real DC Fourier coefficient.
    """

    odd_fundamental: np.ndarray
    even_second_harmonic: np.ndarray
    even_dc: np.ndarray
    odd_phase: np.ndarray
    even_phase: np.ndarray


def phase_fourier(values, harmonics: Iterable[int], phase_offset: float = 0.0):
    """Project a uniform phase grid directly onto selected Fourier harmonics.

    This is the targeted discrete Fourier estimator
    ``Z_n = mean(values * exp(-1j*n*theta))``.  It is mathematically the same
    coefficient that an exactly aligned DFT/FFT would return, but the
    experiment knows the forcing frequency and only needs a few harmonics, so
    computing those projections directly keeps the phase origin explicit and
    avoids estimating an unused broadband spectrum.

    For ``n > 0`` and a real signal ``A*cos(n*theta)+B*sin(n*theta)``, the
    returned coefficient is ``A/2 - 1j*B/2``.  Code and analysis often report
    ``Re(Z_n)`` and ``-Im(Z_n)`` as signed ``cos``/``sin`` coordinates.
    """
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError("values must have a phase axis")
    n_phase = values.shape[-1]
    if n_phase < 1:
        raise ValueError("phase axis must not be empty")

    harmonic_values = tuple(harmonics)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in harmonic_values
    ):
        raise ValueError("harmonics must contain integers")
    harmonics = np.asarray(harmonic_values, dtype=int)
    if harmonics.ndim != 1:
        raise ValueError("harmonics must be one-dimensional")
    if not np.isfinite(phase_offset):
        raise ValueError("phase_offset must be finite")

    theta = phase_offset + 2 * np.pi * np.arange(n_phase) / n_phase
    basis = np.exp(-1j * harmonics[:, None] * theta[None, :])
    return np.einsum("...q,nq->...n", values, basis) / n_phase


def paired_order_contrasts(positive, negative, unforced):
    """Separate strength parity before assigning a perturbative order.

    ``odd=(m(+h)-m(-h))/2`` contains odd powers ``h, h^3, ...`` and
    ``even=(m(+h)+m(-h))/2-m0`` contains even powers ``h^2, h^4, ...``.
    This parity split is about the forcing-strength expansion, not Fourier
    harmonic number.  Harmonics are selected only after the contrast is formed.
    """
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    unforced = np.asarray(unforced)
    if positive.shape != negative.shape or positive.shape != unforced.shape:
        raise ValueError("positive, negative, and unforced must have equal shapes")
    odd = (positive - negative) / 2
    even = (positive + negative) / 2 - unforced
    return odd, even


def harmonic_order_contrast(odd, even, harmonic: int):
    """Use sine-forcing parity to select the contrast for harmonic ``n``.

    For the monochromatic protocol, shifting ``h -> -h`` is a phase shift by
    ``pi``.  Harmonic ``n`` therefore changes sign as ``(-1)^n``: odd
    harmonics live in the odd-in-strength contrast and even harmonics in the
    even-in-strength contrast.  This does *not* mean harmonic ``n`` is the
    ``n``-th perturbative response order.
    """
    odd = np.asarray(odd)
    even = np.asarray(even)
    if odd.shape != even.shape:
        raise ValueError("odd and even contrasts must have equal shapes")
    if isinstance(harmonic, (bool, np.bool_)) or not isinstance(
        harmonic, (int, np.integer)
    ):
        raise ValueError("harmonic must be an integer")
    return odd if int(harmonic) % 2 else even


def effective_phase_responses(odd, even, strength: float):
    """Scale paired contrasts into finite-strength phase response coefficients."""
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    odd = np.asarray(odd)
    even = np.asarray(even)
    if odd.shape != even.shape:
        raise ValueError("odd and even contrasts must have equal shapes")
    return odd / strength, even / strength**2


def directional_fourier_contrasts(
    positive, negative, unforced, phase_offset: float = 0.0
):
    """Extract raw odd/even Fourier contrasts without strength normalization."""
    odd, even = paired_order_contrasts(positive, negative, unforced)
    odd_fundamental = phase_fourier(odd, [1], phase_offset)[..., 0]
    even_coefficients = phase_fourier(even, [0, 2], phase_offset)
    return DirectionalFourierContrasts(
        odd_fundamental=odd_fundamental,
        even_second_harmonic=even_coefficients[..., 1],
        even_dc=even_coefficients[..., 0].real,
        odd_phase=odd,
        even_phase=even,
    )


def directional_frequency_response(
    positive, negative, unforced, strength: float, phase_offset: float = 0.0
):
    """Translate phase/Fourier contrasts into the project's susceptibilities.

    Let ``L_hat`` be the fundamental Fourier coefficient of ``odd/h`` and
    ``Q_hat`` the Fourier coefficients of ``even/h**2``.  Under
    ``h*a*sin(theta)`` and ``Z_n=mean(x*exp(-1j*n*theta))``,

        chi1(omega)@a = 2j * L_hat[1]
        chi2(2*omega;omega,omega)[a,a] = -4 * Q_hat[2]
        Qdc(omega)[a,a] = Q_hat[0].

    If an analysis table writes ``L_hat[1]=c1-1j*s1``, its ``cos``/``sin``
    columns are ``c1=Re(L_hat)`` and ``s1=-Im(L_hat)``.  In the original
    complex-susceptibility language this becomes
    ``Re(chi1)=2*s1, Im(chi1)=2*c1``.  For ``Q_hat[2]=c2-1j*s2``,
    ``Re(chi2)=-4*c2, Im(chi2)=4*s2``.

    The Volterra convention has no ``1/2!`` prefactor.  ``phase_offset`` must
    equal the forcing phase at the first sample on the stored phase axis.
    """
    contrasts = directional_fourier_contrasts(
        positive, negative, unforced, phase_offset
    )
    first_phase, second_phase = effective_phase_responses(
        contrasts.odd_phase, contrasts.even_phase, strength
    )
    return DirectionalFrequencyResponse(
        first_order=2j * contrasts.odd_fundamental / strength,
        second_harmonic=-4 * contrasts.even_second_harmonic / strength**2,
        rectification=contrasts.even_dc / strength**2,
        first_phase=first_phase,
        second_phase=second_phase,
    )


def mirrored_phase_pair_second_order(
    z_plus, z_minus, z_unforced, first_amplitude: float, second_amplitude: float
) -> dict[str, np.ndarray]:
    """Separate a mirrored phase pair into mixed and diagonal channels.

    ``z_plus`` uses component phases ``(+pi/4, -pi/4)`` and ``z_minus`` uses
    ``(-pi/4, +pi/4)``.  Mirror averaging cancels the diagonal quadratic
    contributions at harmonic ``2*omega`` and leaves the mixed contraction
    ``Z_cross_raw``.

    Here ``Z=mean(x*exp(-2j*theta))`` is still a Fourier coefficient, not yet
    ``chi2``.  With sine forcing and the repository's no-factorial Volterra
    convention,

        chi2_jk = -2 * Z_cross_raw / (a_j*a_k).

    If ``Z_cross_raw=c-1j*s``, multiplying by the real factor ``-2/(a_j*a_k)``
    preserves the code's signed coordinate convention: a table row labelled
    ``component=cos`` is ``Re(chi2_jk)``, while ``component=sin`` is
    ``-Im(chi2_jk)``.  Recover the usual complex number as
    ``chi2 = chi2_cos - 1j*chi2_sin``.

    The alternative Taylor/Hessian convention
    ``M=M0+L*a+1/2*H:a*a`` has ``H_jk=2*chi2_jk``.
    """
    z_plus = np.asarray(z_plus)
    z_minus = np.asarray(z_minus)
    z_unforced = np.asarray(z_unforced)
    if z_plus.shape != z_minus.shape or z_plus.shape != z_unforced.shape:
        raise ValueError("mirrored phase-pair and unforced arrays must have equal shapes")
    amplitudes = np.asarray((first_amplitude, second_amplitude), dtype=float)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("mixed component amplitudes must be finite and positive")
    cross_raw = (z_plus + z_minus) / 2 - z_unforced
    diagonal_difference = (z_plus - z_minus) / 2
    normalized = cross_raw / float(np.prod(amplitudes))
    return {
        "cross_raw": cross_raw,
        "diagonal_difference": diagonal_difference,
        "cross_normalized": normalized,
        "second_order": -2 * normalized,
        "taylor_hessian": -4 * normalized,
    }


def _checked_direction_problem(directions, responses):
    directions = np.asarray(directions, dtype=float)
    responses = np.asarray(responses)
    if directions.ndim != 2:
        raise ValueError("directions must have shape (n_direction, n_input)")
    if responses.ndim == 0 or responses.shape[0] != directions.shape[0]:
        raise ValueError("responses must have the same leading direction axis")
    if not np.isfinite(directions).all() or not np.isfinite(responses).all():
        raise ValueError("directions and responses must be finite")
    return directions, responses


def reconstruct_linear_tensor(directions, responses):
    """Recover ``L`` from directional values ``response(a) = L @ a``.

    The leading response axis indexes forcing directions. All remaining axes
    are retained, and the reconstructed forcing-input axis is appended.
    """
    directions, responses = _checked_direction_problem(directions, responses)
    n_input = directions.shape[1]
    if np.linalg.matrix_rank(directions) < n_input:
        raise ValueError("directions do not span the forcing-input space")

    flat = responses.reshape(responses.shape[0], -1)
    coefficients, _, _, _ = np.linalg.lstsq(directions, flat, rcond=None)
    return coefficients.T.reshape(responses.shape[1:] + (n_input,))


def monochromatic_quadratic_design(directions):
    """Build the identifiable same-direction quadratic design.

    The design recovers the effective input-index symmetric part visible in
    contractions with ``a_j*a_k``. It does not assert a general susceptibility
    symmetry.
    """
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2:
        raise ValueError("directions must have shape (n_direction, n_input)")

    n_direction, n_input = directions.shape
    columns = [directions[:, j] ** 2 for j in range(n_input)]
    columns.extend(
        2 * directions[:, j] * directions[:, k]
        for j in range(n_input)
        for k in range(j + 1, n_input)
    )
    return np.column_stack(columns).reshape(n_direction, -1)


def reconstruct_monochromatic_quadratic_tensor(directions, responses):
    """Recover the effective tensor identified by same-direction contractions."""
    directions, responses = _checked_direction_problem(directions, responses)
    design = monochromatic_quadratic_design(directions)
    n_input = directions.shape[1]
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "directions do not span the identifiable monochromatic input pairs"
        )

    flat = responses.reshape(responses.shape[0], -1)
    coefficients, _, _, _ = np.linalg.lstsq(design, flat, rcond=None)
    coefficients = coefficients.T

    tensor = np.zeros(
        (coefficients.shape[0], n_input, n_input), dtype=coefficients.dtype
    )
    tensor[:, np.arange(n_input), np.arange(n_input)] = coefficients[:, :n_input]
    offset = n_input
    for j in range(n_input):
        for k in range(j + 1, n_input):
            tensor[:, j, k] = coefficients[:, offset]
            tensor[:, k, j] = coefficients[:, offset]
            offset += 1
    return tensor.reshape(responses.shape[1:] + (n_input, n_input))
