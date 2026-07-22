"""Shared finite-seed uncertainty, norms, and power-law diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats


def t_critical(confidence_level: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(stats.t.ppf((1.0 + float(confidence_level)) / 2.0, df=n - 1))


def mean_se_ci(values, confidence_level=0.95, axis=0):
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    if n > 1:
        se = values.std(axis=axis, ddof=1) / np.sqrt(n)
        half = t_critical(confidence_level, n) * se
    else:
        se = np.zeros_like(mean)
        half = np.zeros_like(mean)
    return mean, se, mean - half, mean + half


def phase_l2(values, axis=-1):
    return np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2, axis=axis))


def vector_phase_l2(values, axes=(-2, -1)):
    return np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2, axis=axes))


def safe_ratio(numerator, denominator, floor=1e-300):
    return np.asarray(numerator) / np.maximum(np.asarray(denominator), floor)


def local_power_slopes(signal_by_amplitude, amplitudes):
    """Return adjacent log-log slopes; amplitude is the leading axis."""
    signal = np.maximum(np.asarray(signal_by_amplitude, dtype=float), 1e-300)
    amplitudes = np.asarray(amplitudes, dtype=float)
    log_signal = np.log(signal)
    delta_log_a = np.diff(np.log(amplitudes))
    reshape = (delta_log_a.size,) + (1,) * (signal.ndim - 1)
    return np.diff(log_signal, axis=0) / delta_log_a.reshape(reshape)


def bootstrap_ci(values, statistic, samples, confidence_level, random_seed):
    values = np.asarray(values)
    n = values.shape[0]
    rng = np.random.default_rng(int(random_seed))
    estimates = []
    for _ in range(int(samples)):
        indices = rng.integers(0, n, n)
        estimates.append(statistic(values[indices]))
    estimates = np.asarray(estimates)
    alpha = 100.0 * (1.0 - float(confidence_level)) / 2.0
    return np.percentile(estimates, alpha, axis=0), np.percentile(
        estimates, 100.0 - alpha, axis=0)
