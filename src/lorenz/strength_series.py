"""Blockwise power-series fits for finite-strength response contrasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TARGET_ORDERS = {
    "odd_fundamental": (1, 3),
    "even_second_harmonic": (2, 4),
    "even_dc": (2, 4),
}


@dataclass(frozen=True)
class BlockPowerSeriesFit:
    """Power-series coefficients fitted independently inside every block."""

    strengths: np.ndarray
    orders: tuple[int, ...]
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray


def _checked_strengths(strengths) -> np.ndarray:
    strengths = np.asarray(strengths, dtype=float)
    if (
        strengths.ndim != 1
        or len(strengths) < 2
        or not np.isfinite(strengths).all()
        or np.any(strengths <= 0)
        or np.any(np.diff(strengths) <= 0)
    ):
        raise ValueError("strengths must be finite, positive, and strictly increasing")
    return strengths


def fit_block_power_series(values, strengths, orders) -> BlockPowerSeriesFit:
    """Fit declared powers to raw values separately for each crossed block.

    ``values`` has axes ``block, strength, ...``.  No strength normalization
    is performed before fitting.  Returned coefficients therefore use the
    physical powers of the configured strength even though a scaled design is
    used internally for numerical conditioning.
    """
    values = np.asarray(values)
    strengths = _checked_strengths(strengths)
    orders = tuple(orders)
    if values.ndim < 2 or values.shape[1] != len(strengths):
        raise ValueError("values must have axes block, strength, ...")
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("values require at least two finite blocks")
    if (
        not orders
        or len(set(orders)) != len(orders)
        or any(
            isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer))
            or order <= 0
            for order in orders
        )
    ):
        raise ValueError("orders must be distinct positive integers")
    if len(strengths) < len(orders):
        raise ValueError("strength count must be at least the coefficient count")

    scale = float(strengths[-1])
    scaled_design = np.column_stack(
        [(strengths / scale) ** int(order) for order in orders]
    )
    if np.linalg.matrix_rank(scaled_design) < len(orders):
        raise ValueError("strength design is rank deficient")
    flattened = np.moveaxis(values, 1, 0).reshape(len(strengths), -1)
    scaled_coefficients, _, _, _ = np.linalg.lstsq(
        scaled_design, flattened, rcond=None
    )
    physical_coefficients = scaled_coefficients / np.asarray(
        [scale**int(order) for order in orders]
    )[:, None]
    coefficient_shape = (len(orders), values.shape[0], *values.shape[2:])
    coefficients = np.moveaxis(
        physical_coefficients.reshape(coefficient_shape), 0, 1
    )
    physical_design = np.column_stack(
        [strengths ** int(order) for order in orders]
    )
    fitted = np.einsum("sp,bp...->bs...", physical_design, coefficients)
    return BlockPowerSeriesFit(
        strengths=strengths.copy(),
        orders=tuple(int(order) for order in orders),
        coefficients=coefficients,
        fitted=fitted,
        residuals=values - fitted,
    )
