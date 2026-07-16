"""Phase Fourier projections and harmonic diagnostics."""

from __future__ import annotations

import numpy as np


def fourier_coefficients(values, kmax: int):
    """Return coefficients [..., k, (cos, sin)] for k=0..kmax.

    The sine coefficient for k=0 is zero.  For k>0 this is the least-squares
    coefficient in c_k cos(k theta) + s_k sin(k theta).
    """
    values = np.asarray(values)
    n_phase = values.shape[-1]
    theta = 2 * np.pi * np.arange(n_phase) / n_phase
    cols = [np.ones_like(theta)]
    names = [(0, 0)]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
        names += [(k, 0), (k, 1)]
    basis = np.vstack(cols).T
    coef = np.linalg.lstsq(basis, values.reshape(-1, n_phase).T, rcond=None)[0].T
    out = np.zeros(values.shape[:-1] + (kmax + 1, 2))
    out[..., 0, 0] = coef[:, 0].reshape(values.shape[:-1])
    j = 1
    for k in range(1, kmax + 1):
        out[..., k, 0] = coef[:, j].reshape(values.shape[:-1])
        out[..., k, 1] = coef[:, j + 1].reshape(values.shape[:-1])
        j += 2
    return out


def amplitude_phase(coef):
    amp = np.sqrt(coef[..., 0] ** 2 + coef[..., 1] ** 2)
    phase = np.arctan2(-coef[..., 1], coef[..., 0])
    return amp, phase
