"""Crossed-block GLS and joint model comparisons for coefficient analysis.

These routines are deliberately real-valued.  A complex Fourier response is
represented upstream by two signed coordinates ``(Re Z, -Im Z)`` and each
coordinate is fitted with the same strength design.  The two fits are not two
physical response mechanisms; together they are the two real degrees of
freedom of one complex coefficient.  Joint model comparisons may stack both
coordinates (and several tensor entries) as features while preserving their
cross-scale block covariance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve


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


def power_series_operator(strengths, orders) -> tuple[np.ndarray, np.ndarray]:
    """Return the physical design and stable least-squares coefficient map."""
    strengths = _checked_strengths(strengths)
    orders = tuple(orders)
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
    physical_design = np.column_stack(
        [strengths ** int(order) for order in orders]
    )
    coefficient_map = (
        np.diag([scale ** -int(order) for order in orders])
        @ np.linalg.pinv(scaled_design)
    )
    return physical_design, coefficient_map


def crossed_block_gls_fit(block_values, strengths, orders) -> dict:
    """Fit one real coordinate using its crossed-scale block covariance.

    For Fourier-response work this real coordinate is normally either
    ``Re(Z)`` or ``-Im(Z)``.  Calling them ``cos`` and ``sin`` elsewhere is a
    coordinate convention; this routine itself has no phase or complex-number
    semantics.
    """
    block_values = np.asarray(block_values, dtype=float)
    strengths = _checked_strengths(strengths)
    orders = tuple(int(order) for order in orders)
    if block_values.ndim != 2 or block_values.shape[1] != len(strengths):
        raise ValueError("block_values must have axes block,strength")
    if block_values.shape[0] <= len(strengths) or not np.isfinite(block_values).all():
        raise ValueError("GLS requires more finite blocks than strengths")
    physical_design, _ = power_series_operator(strengths, orders)
    scale = float(np.max(strengths))
    physical_scales = np.asarray([scale**order for order in orders])
    scaled_design = physical_design / physical_scales[None, :]
    covariance = np.atleast_2d(np.cov(block_values, rowvar=False, ddof=1))
    factor = cho_factor(covariance, lower=True, check_finite=False)
    inverse_design = cho_solve(factor, scaled_design, check_finite=False)
    normal = scaled_design.T @ inverse_design
    scaled_operator = np.linalg.solve(
        normal,
        scaled_design.T @ cho_solve(
            factor, np.eye(len(strengths)), check_finite=False
        ),
    )
    coefficient_operator = scaled_operator / physical_scales[:, None]
    block_coefficients = block_values @ coefficient_operator.T
    mean_coefficients = block_coefficients.mean(axis=0)
    fitted_mean = physical_design @ mean_coefficients
    return {
        "block_coefficients": block_coefficients,
        "mean_coefficients": mean_coefficients,
        "fitted_mean": fitted_mean,
        "residuals": block_values.mean(axis=0) - fitted_mean,
        "coefficient_operator": coefficient_operator,
        "intercept_weights_on_raw_contrast": coefficient_operator[0],
        "covariance": covariance,
    }


def joint_crossed_block_model_comparison(block_features, strengths, order_sets):
    """Joint Hotelling-F lack-of-fit and nested tests for crossed features."""
    block_features = np.asarray(block_features, dtype=float)
    strengths = _checked_strengths(strengths)
    if (
        block_features.ndim != 3
        or block_features.shape[1] != len(strengths)
        or not np.isfinite(block_features).all()
    ):
        raise ValueError("block_features must have finite axes block,strength,feature")
    block_count, strength_count, feature_count = block_features.shape
    if block_count <= strength_count * feature_count:
        raise ValueError("joint comparison needs more blocks than scalar features")
    mean = block_features.mean(axis=0).reshape(-1)
    covariance_of_mean = np.cov(
        block_features.reshape(block_count, -1), rowvar=False, ddof=1
    ) / block_count
    factor = cho_factor(covariance_of_mean, lower=True, check_finite=False)
    inverse_mean = cho_solve(factor, mean, check_finite=False)
    comparisons = []
    for orders in order_sets:
        orders = tuple(int(order) for order in orders)
        if strength_count <= len(orders):
            continue
        design, _ = power_series_operator(strengths, orders)
        joint_design = np.kron(design, np.eye(feature_count))
        inverse_design = cho_solve(factor, joint_design, check_finite=False)
        coefficients = np.linalg.solve(
            joint_design.T @ inverse_design,
            joint_design.T @ inverse_mean,
        )
        residual = mean - joint_design @ coefficients
        statistic = float(residual @ cho_solve(factor, residual, check_finite=False))
        degrees_of_freedom = (strength_count - len(orders)) * feature_count
        hotelling_f = (
            (block_count - degrees_of_freedom)
            / (degrees_of_freedom * (block_count - 1))
            * statistic
        )
        comparisons.append({
            "orders": orders,
            "statistic": statistic,
            "degrees_of_freedom": degrees_of_freedom,
            "hotelling_f": float(hotelling_f),
            "f_denominator_degrees_of_freedom": block_count - degrees_of_freedom,
            "lack_of_fit_p": float(stats.f.sf(
                hotelling_f, degrees_of_freedom, block_count - degrees_of_freedom
            )),
            "aic": statistic + 2 * len(orders) * feature_count,
        })
    for index in range(1, len(comparisons)):
        simpler = comparisons[index - 1]
        current = comparisons[index]
        added = (len(current["orders"]) - len(simpler["orders"])) * feature_count
        improvement = max(0.0, simpler["statistic"] - current["statistic"])
        remaining = current["degrees_of_freedom"]
        # Eliminate the simple model by fixed contrasts. Regress its added
        # mean directions (d) on the full model's remaining restrictions (q).
        # The estimated nuisance regression changes both the denominator and
        # its degrees of freedom; the added-only Hotelling F is valid at q=0.
        improvement_f = (
            (block_count - added - remaining) * improvement
            / (added * (block_count - 1 + current["statistic"]))
        )
        current["nested_improvement_f"] = float(improvement_f)
        current["nested_denominator_df"] = block_count - added - remaining
        current["nested_calibration"] = "Gaussian contrast/regression F including full-model residual dimension"
        current["nested_improvement_p"] = float(
            stats.f.sf(improvement_f, added, block_count - added - remaining)
        )
    return comparisons
