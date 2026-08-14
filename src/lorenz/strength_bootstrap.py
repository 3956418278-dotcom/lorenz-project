"""Whole-block simultaneous bootstrap for strength identifiability decisions.

One bootstrap draw resamples an initial-state block and therefore carries all
strengths, target harmonics, output components, and fitted coefficients for
that block together.  The resulting max-statistic family is deliberately
explicit and persisted with its feature labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strength_identifiability import fit_block_power_series


TARGET_ORDERS = {
    "odd_fundamental": (1, 3),
    "even_second_harmonic": (2, 4),
    "even_dc": (2, 4),
}


@dataclass(frozen=True)
class ScalarFeature:
    """One scalar coordinate in the joint realified decision family."""

    section: str
    target: str
    observable: int
    part: str
    strength: float | None = None
    prefix_upper_strength: float | None = None
    contribution: str | None = None
    structural_null: bool = False


@dataclass(frozen=True)
class ComplexFeatureGroup:
    """Indices needed to recover one real or complex physical component."""

    section: str
    target: str
    observable: int
    real_index: int
    imaginary_index: int | None
    strength: float | None = None
    prefix_upper_strength: float | None = None
    contribution: str | None = None
    structural_null: bool = False


def _checked_targets(targets, strengths, n_block: int):
    if set(targets) != set(TARGET_ORDERS):
        raise ValueError(f"targets must contain exactly {tuple(TARGET_ORDERS)}")
    checked = {}
    for name in TARGET_ORDERS:
        values = np.asarray(targets[name])
        if (
            values.ndim != 3
            or values.shape[:2] != (n_block, len(strengths))
            or values.shape[2] < 1
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"{name} must have finite axes block, strength, observable"
            )
        if name == "even_dc" and np.iscomplexobj(values):
            if np.any(values.imag != 0):
                raise ValueError("even_dc must be real-valued")
            values = values.real
        checked[name] = values
    observable_count = {value.shape[2] for value in checked.values()}
    if len(observable_count) != 1:
        raise ValueError("all targets must have the same observable count")
    return checked, observable_count.pop()


def _checked_null_masks(masks, observable_count: int):
    if set(masks) != set(TARGET_ORDERS):
        raise ValueError("structural_null_outputs must name every target")
    result = {}
    for target in TARGET_ORDERS:
        mask = np.asarray(masks[target])
        if mask.shape != (observable_count,) or mask.dtype != np.bool_:
            raise ValueError(
                "each structural-null mask must be a boolean observable vector"
            )
        if np.all(mask):
            raise ValueError(f"{target} must retain at least one non-null output")
        result[target] = mask
    return result


def _append_groups(
    columns,
    scalar_features,
    groups,
    values,
    *,
    section,
    target,
    structural_nulls,
    strength=None,
    prefix_upper_strength=None,
    contribution=None,
):
    """Append all real coordinates, then all imaginary coordinates."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("feature values must have axes block, observable")
    start = len(columns)
    for observable in range(values.shape[1]):
        columns.append(np.asarray(values[:, observable].real, dtype=float))
        scalar_features.append(
            ScalarFeature(
                section=section,
                target=target,
                observable=observable,
                part="real",
                strength=strength,
                prefix_upper_strength=prefix_upper_strength,
                contribution=contribution,
                structural_null=bool(structural_nulls[observable]),
            )
        )
    imaginary_start = None
    if np.iscomplexobj(values):
        imaginary_start = len(columns)
        for observable in range(values.shape[1]):
            columns.append(np.asarray(values[:, observable].imag, dtype=float))
            scalar_features.append(
                ScalarFeature(
                    section=section,
                    target=target,
                    observable=observable,
                    part="imaginary",
                    strength=strength,
                    prefix_upper_strength=prefix_upper_strength,
                    contribution=contribution,
                    structural_null=bool(structural_nulls[observable]),
                )
            )
    for observable in range(values.shape[1]):
        groups.append(
            ComplexFeatureGroup(
                section=section,
                target=target,
                observable=observable,
                real_index=start + observable,
                imaginary_index=(
                    imaginary_start + observable
                    if imaginary_start is not None
                    else None
                ),
                strength=strength,
                prefix_upper_strength=prefix_upper_strength,
                contribution=contribution,
                structural_null=bool(structural_nulls[observable]),
            )
        )


def build_decision_family(targets, strengths, structural_null_outputs):
    """Build the scalar family and physical-component group mapping.

    The family contains raw target contrasts at every strength plus low- and
    first-higher-order contributions at the upper strength of every nested
    prefix containing at least three strength levels.
    """
    strengths = np.asarray(strengths, dtype=float)
    if (
        strengths.ndim != 1
        or len(strengths) < 3
        or not np.isfinite(strengths).all()
        or np.any(strengths <= 0)
        or np.any(np.diff(strengths) <= 0)
    ):
        raise ValueError("strengths must be increasing positive values")
    first = np.asarray(next(iter(targets.values())))
    if first.ndim < 1 or first.shape[0] < 2:
        raise ValueError("targets need at least two blocks")
    checked, observable_count = _checked_targets(
        targets, strengths, first.shape[0]
    )
    nulls = _checked_null_masks(structural_null_outputs, observable_count)
    columns = []
    scalar_features = []
    groups = []

    for target in TARGET_ORDERS:
        for strength_index, strength in enumerate(strengths):
            _append_groups(
                columns,
                scalar_features,
                groups,
                checked[target][:, strength_index],
                section="identification",
                target=target,
                structural_nulls=nulls[target],
                strength=float(strength),
            )

    for target, orders in TARGET_ORDERS.items():
        for stop in range(3, len(strengths) + 1):
            prefix = strengths[:stop]
            fit = fit_block_power_series(checked[target][:, :stop], prefix, orders)
            upper = float(prefix[-1])
            low = fit.coefficients[:, 0] * upper ** orders[0]
            higher = fit.coefficients[:, 1] * upper ** orders[1]
            for contribution, values in (("low", low), ("higher", higher)):
                _append_groups(
                    columns,
                    scalar_features,
                    groups,
                    values,
                    section="adequacy",
                    target=target,
                    structural_nulls=nulls[target],
                    prefix_upper_strength=upper,
                    contribution=contribution,
                )
    return (
        np.column_stack(columns),
        tuple(scalar_features),
        tuple(groups),
        nulls,
    )


def _bootstrap_max_statistic(
    samples,
    *,
    resamples: int,
    root_entropy,
    batch_size: int,
):
    samples = np.asarray(samples, dtype=float)
    n_block, n_feature = samples.shape
    mean = samples.mean(axis=0)
    standard_error = samples.std(axis=0, ddof=1) / np.sqrt(n_block)
    active = standard_error > 0
    if not np.any(active):
        return mean, standard_error, np.zeros(resamples), {
            "root_entropy": root_entropy,
            "bit_generator": "PCG64DXSM",
        }
    seed_sequence = np.random.SeedSequence(root_entropy)
    rng = np.random.Generator(np.random.PCG64DXSM(seed_sequence))
    statistics = np.empty(resamples, dtype=float)
    completed = 0
    while completed < resamples:
        count = min(batch_size, resamples - completed)
        indices = rng.integers(0, n_block, size=(count, n_block))
        bootstrap_means = samples[indices].mean(axis=1)
        standardized = np.abs(
            (bootstrap_means[:, active] - mean[active]) / standard_error[active]
        )
        statistics[completed : completed + count] = standardized.max(axis=1)
        completed += count
    entropy = seed_sequence.entropy
    if isinstance(entropy, (list, tuple, np.ndarray)):
        entropy = tuple(int(item) for item in entropy)
    else:
        entropy = int(entropy)
    return mean, standard_error, statistics, {
        "root_entropy": entropy,
        "root_spawn_key": tuple(seed_sequence.spawn_key),
        "bit_generator": "PCG64DXSM",
    }


def _magnitude_interval(group, lower, upper):
    def coordinate_bounds(index):
        lo = float(lower[index])
        hi = float(upper[index])
        distance = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
        extent = max(abs(lo), abs(hi))
        return distance, extent

    real_lower, real_upper = coordinate_bounds(group.real_index)
    if group.imaginary_index is None:
        return real_lower, real_upper
    imag_lower, imag_upper = coordinate_bounds(group.imaginary_index)
    return (
        float(np.hypot(real_lower, imag_lower)),
        float(np.hypot(real_upper, imag_upper)),
    )


def _feature_dict(feature: ScalarFeature) -> dict:
    return {
        "section": feature.section,
        "target": feature.target,
        "observable": feature.observable,
        "part": feature.part,
        "strength": feature.strength,
        "prefix_upper_strength": feature.prefix_upper_strength,
        "contribution": feature.contribution,
        "structural_null": feature.structural_null,
    }


def analyze_block_bootstrap(
    block_ids,
    strengths,
    targets,
    structural_null_outputs,
    *,
    confidence: float,
    resamples: int,
    root_entropy,
    batch_size: int,
    higher_order_fraction_limit: float,
) -> dict:
    """Apply one simultaneous whole-block bootstrap decision family."""
    block_ids = tuple(block_ids)
    if len(block_ids) < 2 or len(set(block_ids)) != len(block_ids):
        raise ValueError("block_ids must contain at least two unique blocks")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if isinstance(resamples, (bool, np.bool_)) or int(resamples) != resamples or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(batch_size, (bool, np.bool_)) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (
        not np.isfinite(higher_order_fraction_limit)
        or not 0 < higher_order_fraction_limit < 1
    ):
        raise ValueError("higher_order_fraction_limit must lie between zero and one")
    matrix, scalar_features, groups, nulls = build_decision_family(
        targets, strengths, structural_null_outputs
    )
    if matrix.shape[0] != len(block_ids):
        raise ValueError("target block axes must match block_ids")
    mean, standard_error, statistics, bootstrap_metadata = _bootstrap_max_statistic(
        matrix,
        resamples=int(resamples),
        root_entropy=root_entropy,
        batch_size=int(batch_size),
    )
    critical = float(np.quantile(statistics, confidence, method="higher"))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error

    identification = {target: {} for target in TARGET_ORDERS}
    adequacy_groups = {}
    null_control_exclusions = []
    for group in groups:
        magnitude_lower, magnitude_upper = _magnitude_interval(group, lower, upper)
        record = {
            "observable": group.observable,
            "structural_null": group.structural_null,
            "magnitude_lower": magnitude_lower,
            "magnitude_upper": magnitude_upper,
            "simultaneous_interval_excludes_origin": magnitude_lower > 0,
        }
        if group.section == "identification":
            strength_key = str(float(group.strength))
            identification[group.target].setdefault(strength_key, []).append(record)
            if group.structural_null and magnitude_lower > 0:
                null_control_exclusions.append(
                    {
                        "target": group.target,
                        "strength": group.strength,
                        "observable": group.observable,
                    }
                )
        else:
            key = (group.target, float(group.prefix_upper_strength))
            adequacy_groups.setdefault(key, {"low": [], "higher": []})[
                group.contribution
            ].append(record)

    identification_results = {}
    for target, by_strength in identification.items():
        identification_results[target] = {}
        for strength, records in by_strength.items():
            allowed = [item for item in records if not item["structural_null"]]
            identification_results[target][strength] = {
                "components": records,
                "identified": any(
                    item["simultaneous_interval_excludes_origin"] for item in allowed
                ),
                "rule": (
                    "at least one non-structural-null output component has a "
                    "simultaneous complex rectangle excluding the origin"
                ),
            }

    adequacy_results = {target: {} for target in TARGET_ORDERS}
    for (target, upper_strength), contributions in adequacy_groups.items():
        allowed_low = [
            item for item in contributions["low"] if not item["structural_null"]
        ]
        allowed_higher = [
            item for item in contributions["higher"] if not item["structural_null"]
        ]
        dominant_low_lower = max(item["magnitude_lower"] for item in allowed_low)
        maximum_higher_upper = max(
            item["magnitude_upper"] for item in allowed_higher
        )
        denominator_resolved = dominant_low_lower > 0
        ratio_upper = (
            maximum_higher_upper / dominant_low_lower
            if denominator_resolved
            else None
        )
        adequate = (
            bool(ratio_upper <= higher_order_fraction_limit)
            if denominator_resolved
            else False
        )
        adequacy_results[target][str(upper_strength)] = {
            "low_order_components": contributions["low"],
            "higher_order_components": contributions["higher"],
            "dominant_low_order_magnitude_lower": dominant_low_lower,
            "maximum_higher_order_magnitude_upper": maximum_higher_upper,
            "higher_to_dominant_low_upper": ratio_upper,
            "denominator_status": (
                "simultaneously_resolved" if denominator_resolved else "unresolved"
            ),
            "adequate": adequate,
            "fraction_limit": float(higher_order_fraction_limit),
            "rule": (
                "the simultaneous upper bound on every allowed first-higher-"
                "order contribution at the prefix upper strength is at most "
                "fraction_limit times the simultaneous lower bound on the "
                "dominant allowed low-order contribution"
            ),
        }

    intersection = {target: {} for target in TARGET_ORDERS}
    for target in TARGET_ORDERS:
        for upper_strength, adequate in adequacy_results[target].items():
            identified = identification_results[target][upper_strength]["identified"]
            intersection[target][upper_strength] = {
                "identified": identified,
                "adequate": adequate["adequate"],
                "in_identification_window": identified and adequate["adequate"],
            }

    sample_covariance = np.cov(matrix, rowvar=False, ddof=1)
    return {
        "interpretation": (
            "Confirmed whole-initial-state-block bootstrap for the provisional "
            "protocol family. One index draw resamples every strength, target, "
            "component, and fitted contribution of a block together. The one "
            "max-studentized-error critical value gives simultaneous coordinate "
            "coverage for the complete persisted decision family."
        ),
        "coverage": {
            "confidence": float(confidence),
            "resamples": int(resamples),
            "statistic": (
                "max over all nonzero-SE scalar family coordinates of "
                "abs(bootstrap_mean - observed_mean) / observed_standard_error"
            ),
            "critical_value": critical,
            "quantile_method": "higher",
            "bootstrap": bootstrap_metadata,
            "resampling_unit": "whole initial-state block",
        },
        "family": {
            "scalar_coordinate_count": matrix.shape[1],
            "physical_component_count": len(groups),
            "realification": (
                "within each physical group all observable real coordinates "
                "are appended before all observable imaginary coordinates"
            ),
            "members": [_feature_dict(feature) for feature in scalar_features],
            "mean_realified": mean,
            "standard_error_realified": standard_error,
            "simultaneous_lower_realified": lower,
            "simultaneous_upper_realified": upper,
            "sample_covariance_realified": sample_covariance,
            "covariance_of_mean_realified": sample_covariance / len(block_ids),
        },
        "structural_null_outputs": {
            target: mask for target, mask in nulls.items()
        },
        "null_control_exclusions": null_control_exclusions,
        "identification": identification_results,
        "adequacy": adequacy_results,
        "identification_window": intersection,
        "denominator_policy": (
            "structural-null outputs never enter adequacy ratios. A target/prefix "
            "ratio is not formed unless at least one allowed low-order component "
            "has a positive simultaneous magnitude lower bound; unresolved "
            "denominators fail adequacy rather than produce an unstable ratio."
        ),
        "model_scope": (
            "Adequacy bounds the first omitted parity-allowed power in the "
            "configured two-power fit. It does not by itself prove that all "
            "still-higher powers are negligible; nested-prefix stability and "
            "lack-of-fit remain reported diagnostics."
        ),
    }
