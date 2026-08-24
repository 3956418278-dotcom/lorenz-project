"""Block-preserving estimators and sampling covariance for responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .response import (
    directional_fourier_contrasts,
    reconstruct_linear_tensor,
    reconstruct_monochromatic_quadratic_tensor,
)


@dataclass(frozen=True)
class CosSinStatistics:
    """Per-component cosine/sine statistics of a block-level response.

    The formal statistics are the signed components; the magnitude and
    phase are derived descriptors only and carry no significance statement
    of their own.  ``hotelling_p_value`` is the joint (cos, sin)
    significance test against the origin; the pseudoinverse flag records
    the degenerate-covariance fallback.  The DC harmonic is handled as a
    scalar elsewhere, never through this two-dimensional test.
    """

    mean_cos: float
    se_cos: float
    ci_cos: tuple[float, float]
    mean_sin: float
    se_sin: float
    ci_sin: tuple[float, float]
    hotelling_t2: float
    hotelling_f: float
    hotelling_p_value: float
    used_pseudoinverse: bool
    magnitude: float
    phase: float


def cos_sin_components(values):
    """Decompose complex block coefficients into signed components.

    Under the project Fourier convention ``mean(x(theta) exp(-i n theta))``
    (verified against known pure signals: a pure cosine maps to a real
    coefficient, a pure sine to a negative-imaginary coefficient), the real
    part IS the cosine component and the imaginary part is the NEGATIVE
    sine component: ``coefficient = cos - i * sin``.
    """
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        raise ValueError("cos/sin decomposition requires complex coefficients")
    return values.real, -values.imag


def cos_sin_statistics(values, confidence: float = 0.95) -> CosSinStatistics:
    """Signed-component statistics of per-block complex coefficients.

    ``values`` has axes ``(block, ...)``; statistics are computed over the
    leading block axis.  Returns per-component block means, standard errors
    of the mean, and t-intervals at ``confidence``, plus the Hotelling T^2
    joint test of the two-vector ``(cos_b, sin_b)`` against the origin
    (T^2 = B * mean^T S^{-1} mean with the sample covariance S; the F-form
    uses (B-2) / (2 (B-1)) * T^2 with 2 and B-2 degrees of freedom).
    ``magnitude = sqrt(mean_cos^2 + mean_sin^2)`` and the phase are derived
    descriptors.  Never use ``mean(sqrt(cos_b^2 + sin_b^2))`` as the
    response strength.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.shape[0] < 3:
        raise ValueError(
            "cos_sin_statistics takes one complex coefficient per block"
        )
    cos, sin = cos_sin_components(values)
    blocks = cos.shape[0]
    mean_cos = float(cos.mean())
    mean_sin = float(sin.mean())
    se_cos = float(cos.std(ddof=1) / np.sqrt(blocks))
    se_sin = float(sin.std(ddof=1) / np.sqrt(blocks))
    critical = float(stats.t.ppf((1 + confidence) / 2, df=blocks - 1))
    ci_cos = (mean_cos - critical * se_cos, mean_cos + critical * se_cos)
    ci_sin = (mean_sin - critical * se_sin, mean_sin + critical * se_sin)
    vectors = np.column_stack((cos, sin))
    mean = vectors.mean(axis=0)
    covariance = np.cov(vectors, rowvar=False, ddof=1)
    used_pseudoinverse = False
    try:
        score = float(mean @ np.linalg.solve(covariance, mean))
    except np.linalg.LinAlgError:
        used_pseudoinverse = True
        score = float(mean @ np.linalg.pinv(covariance) @ mean)
    t2 = blocks * max(score, 0.0)
    f_stat = (blocks - 2) / (2 * (blocks - 1)) * t2
    p_value = float(stats.f.sf(f_stat, 2, blocks - 2))
    magnitude = float(np.hypot(mean_cos, mean_sin))
    phase = float(np.arctan2(mean_sin, mean_cos))
    return CosSinStatistics(
        mean_cos=mean_cos,
        se_cos=se_cos,
        ci_cos=ci_cos,
        mean_sin=mean_sin,
        se_sin=se_sin,
        ci_sin=ci_sin,
        hotelling_t2=t2,
        hotelling_f=f_stat,
        hotelling_p_value=p_value,
        used_pseudoinverse=used_pseudoinverse,
        magnitude=magnitude,
        phase=phase,
    )


def benjamini_hochberg(p_values) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    if (
        p_values.ndim != 1
        or len(p_values) < 1
        or not np.isfinite(p_values).all()
        or np.any((p_values < 0) | (p_values > 1))
    ):
        raise ValueError("p_values must be a nonempty finite vector in [0, 1]")
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values


@dataclass(frozen=True)
class FrequencyResponses:
    """One collection of response values without a replication axis."""

    first_order: np.ndarray
    second_harmonic: np.ndarray
    rectification: np.ndarray


@dataclass(frozen=True)
class BlockFourierContrasts:
    """Raw Fourier contrasts with explicit block identity on axis zero."""

    block_ids: tuple[int, ...]
    odd_fundamental: np.ndarray
    even_second_harmonic: np.ndarray
    even_dc: np.ndarray


@dataclass(frozen=True)
class BlockFrequencyResponses:
    """Finite-strength responses with explicit block identity on axis zero."""

    block_ids: tuple[int, ...]
    first_order: np.ndarray
    second_harmonic: np.ndarray
    rectification: np.ndarray


@dataclass(frozen=True)
class ResponseFeature:
    """Location of one response quantity in a joint real feature vector."""

    name: str
    value_shape: tuple[int, ...]
    real: slice
    imaginary: slice | None


@dataclass(frozen=True)
class BlockResponseSummary:
    """Cross-block mean and sampling covariance of that mean.

    ``covariance_of_mean`` is in the real feature ordering described by
    ``features``. Numerical and transient biases are intentionally absent.
    """

    mean: FrequencyResponses
    realified_mean: np.ndarray
    sample_covariance: np.ndarray
    covariance_of_mean: np.ndarray
    features: tuple[ResponseFeature, ...]
    block_ids: tuple[int, ...]
    n_block: int


def _checked_strength(strength: float) -> float:
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    return float(strength)


def _real_rectification(values, message: str):
    values = np.asarray(values)
    if np.iscomplexobj(values):
        if np.any(values.imag != 0):
            raise ValueError(message)
        values = values.real
    return values


def _checked_block_ids(block_ids, n_block: int) -> tuple[int, ...]:
    block_ids = tuple(block_ids)
    if len(block_ids) != n_block:
        raise ValueError("block_ids must match the leading block axis")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("block_ids must be unique")
    return block_ids


def _checked_response_arrays(responses: BlockFrequencyResponses):
    arrays = tuple(
        np.asarray(value)
        for value in (
            responses.first_order,
            responses.second_harmonic,
            _real_rectification(
                responses.rectification, "rectification must be real-valued"
            ),
        )
    )
    if any(value.ndim == 0 for value in arrays):
        raise ValueError("response arrays must have a leading block axis")
    if len({value.shape[0] for value in arrays}) != 1:
        raise ValueError("response arrays must have the same number of blocks")
    block_ids = _checked_block_ids(responses.block_ids, arrays[0].shape[0])
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("response arrays must be finite")
    return arrays, block_ids


def paired_block_fourier_contrasts(
    block_ids, positive, negative, unforced, phase_offset: float = 0.0
) -> BlockFourierContrasts:
    """Average cycles within explicit blocks, then form target contrasts.

    Condition arrays must have shape ``(block, ..., cycle, phase)``. Requiring
    at least four axes prevents a single ``(state, cycle, phase)`` sample from
    being silently interpreted as three replication blocks.
    """
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    unforced = np.asarray(unforced)
    if positive.shape != negative.shape or positive.shape != unforced.shape:
        raise ValueError("condition arrays must have equal shapes")
    if positive.ndim < 4:
        raise ValueError(
            "condition arrays must have shape (block, ..., cycle, phase)"
        )
    block_ids = _checked_block_ids(block_ids, positive.shape[0])
    if positive.shape[-2] < 1 or positive.shape[-1] < 1:
        raise ValueError("cycle and phase axes must not be empty")

    contrasts = directional_fourier_contrasts(
        positive.mean(axis=-2),
        negative.mean(axis=-2),
        unforced.mean(axis=-2),
        phase_offset,
    )
    return BlockFourierContrasts(
        block_ids=block_ids,
        odd_fundamental=contrasts.odd_fundamental,
        even_second_harmonic=contrasts.even_second_harmonic,
        even_dc=contrasts.even_dc,
    )


def finite_strength_block_responses(
    contrasts: BlockFourierContrasts, strength: float
) -> BlockFrequencyResponses:
    """Normalize raw Fourier contrasts without averaging over blocks."""
    strength = _checked_strength(strength)
    odd = np.asarray(contrasts.odd_fundamental)
    even_second = np.asarray(contrasts.even_second_harmonic)
    even_dc = _real_rectification(
        contrasts.even_dc, "the DC Fourier contrast must be real-valued"
    )
    if odd.ndim == 0 or even_second.ndim == 0 or even_dc.ndim == 0:
        raise ValueError("Fourier contrasts must have a leading block axis")
    if odd.shape != even_second.shape or odd.shape != even_dc.shape:
        raise ValueError("target Fourier contrasts must have equal shapes")
    block_ids = _checked_block_ids(contrasts.block_ids, odd.shape[0])
    if not (
        np.isfinite(odd).all()
        and np.isfinite(even_second).all()
        and np.isfinite(even_dc).all()
    ):
        raise ValueError("Fourier contrasts must be finite")
    return BlockFrequencyResponses(
        block_ids=block_ids,
        first_order=2j * odd / strength,
        second_harmonic=-4 * even_second / strength**2,
        rectification=even_dc / strength**2,
    )


def reconstruct_block_tensors(
    directions, responses: BlockFrequencyResponses
) -> BlockFrequencyResponses:
    """Apply fixed direction designs independently within every block.

    Input response arrays have shape ``(block, direction, ...)``. The output
    forcing index or indices are appended by the reconstruction routines.
    """
    (first, second, dc), block_ids = _checked_response_arrays(responses)
    if any(value.ndim < 2 for value in (first, second, dc)):
        raise ValueError("response arrays must have block and direction axes")
    direction_array = np.asarray(directions)
    if direction_array.ndim != 2:
        raise ValueError("directions must have shape (n_direction, n_input)")
    n_direction = direction_array.shape[0]
    if any(value.shape[1] != n_direction for value in (first, second, dc)):
        raise ValueError("response direction axes must match directions")

    return BlockFrequencyResponses(
        block_ids=block_ids,
        first_order=reconstruct_linear_tensor(
            directions, np.moveaxis(first, 1, 0)
        ),
        second_harmonic=reconstruct_monochromatic_quadratic_tensor(
            directions, np.moveaxis(second, 1, 0)
        ),
        rectification=reconstruct_monochromatic_quadratic_tensor(
            directions, np.moveaxis(dc, 1, 0)
        ),
    )


def realify_block_values(values) -> np.ndarray:
    """Flatten each block, placing all real parts before all imaginary parts."""
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError("values must have a leading block axis")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    flat = values.reshape(values.shape[0], -1)
    if np.iscomplexobj(values):
        return np.concatenate((flat.real, flat.imag), axis=1)
    return np.asarray(flat, dtype=float)


def restore_block_values(
    realified, value_shape: tuple[int, ...], *, complex_valued: bool
) -> np.ndarray:
    """Invert :func:`realify_block_values` for a declared component shape."""
    realified = np.asarray(realified, dtype=float)
    if realified.ndim != 2:
        raise ValueError("realified values must have shape (block, feature)")
    component_count = int(np.prod(value_shape, dtype=int))
    expected = component_count * (2 if complex_valued else 1)
    if realified.shape[1] != expected:
        raise ValueError("realified feature count does not match value_shape")
    if complex_valued:
        flat = (
            realified[:, :component_count]
            + 1j * realified[:, component_count:]
        )
    else:
        flat = realified
    return flat.reshape((realified.shape[0],) + tuple(value_shape))


def summarize_block_responses(
    responses: BlockFrequencyResponses,
) -> BlockResponseSummary:
    """Estimate a joint response mean and unbiased covariance of its mean.

    Blocks are the only replications used here. For ``B`` blocks, the returned
    covariance is the usual unbiased sample covariance divided by ``B``.
    """
    (first, second, dc), block_ids = _checked_response_arrays(responses)
    n_block = first.shape[0]
    if n_block < 2:
        raise ValueError("at least two blocks are required for covariance")

    names = ("first_order", "second_harmonic", "rectification")
    arrays = (first, second, dc)
    matrices = []
    features = []
    offset = 0
    for name, values in zip(names, arrays):
        matrix = realify_block_values(values)
        matrices.append(matrix)
        component_count = int(np.prod(values.shape[1:], dtype=int))
        real_slice = slice(offset, offset + component_count)
        offset += component_count
        imaginary_slice = None
        if np.iscomplexobj(values):
            imaginary_slice = slice(offset, offset + component_count)
            offset += component_count
        features.append(
            ResponseFeature(
                name=name,
                value_shape=values.shape[1:],
                real=real_slice,
                imaginary=imaginary_slice,
            )
        )

    samples = np.concatenate(matrices, axis=1)
    realified_mean = samples.mean(axis=0)
    centered = samples - realified_mean
    sample_covariance = centered.T @ centered / (n_block - 1)
    covariance_of_mean = sample_covariance / n_block
    mean = FrequencyResponses(
        first_order=first.mean(axis=0),
        second_harmonic=second.mean(axis=0),
        rectification=dc.mean(axis=0),
    )
    return BlockResponseSummary(
        mean=mean,
        realified_mean=realified_mean,
        sample_covariance=sample_covariance,
        covariance_of_mean=covariance_of_mean,
        features=tuple(features),
        block_ids=block_ids,
        n_block=len(block_ids),
    )
