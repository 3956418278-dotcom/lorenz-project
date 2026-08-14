"""Within-block observation-length diagnostics from retained cycle summaries.

Cycle windows are repeated measurements inside an initial-state block, never
sampling replications.  Every variance and bootstrap calculation continues to
use the leading block axis as the independent unit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .artifacts import (
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    write_json_atomic,
)
from .strength_bootstrap import (
    analyze_block_bootstrap,
    build_decision_family,
)
from .strength_series import TARGET_ORDERS


def _target_cycles(raw) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    block_ids = np.asarray(raw["block_ids"])
    strengths = np.asarray(raw["strengths"], dtype=float)
    harmonics = np.asarray(raw["harmonics"])
    positive = np.asarray(raw["positive_cycle_fourier"])
    negative = np.asarray(raw["negative_cycle_fourier"])
    unforced = np.asarray(raw["unforced_cycle_fourier"])
    if (
        positive.shape != negative.shape
        or positive.ndim != 5
        or positive.shape[:2] != (len(block_ids), len(strengths))
        or positive.shape[2] != 3
        or positive.shape[-1] != len(harmonics)
        or unforced.shape
        != (
            len(block_ids),
            3,
            positive.shape[-2],
            len(harmonics),
        )
    ):
        raise ValueError("raw cycle Fourier arrays have incompatible axes")
    index = {int(value): idx for idx, value in enumerate(harmonics)}
    if not {0, 1, 2}.issubset(index):
        raise ValueError("harmonics must include 0, 1, and 2")
    odd = (positive - negative) / 2
    even = (positive + negative) / 2 - unforced[:, None]
    return strengths, {
        "odd_fundamental": odd[..., index[1]],
        "even_second_harmonic": even[..., index[2]],
        "even_dc": even[..., index[0]].real,
    }


def _cycle_family(target_cycles, strengths, nulls):
    n_cycle = next(iter(target_cycles.values())).shape[-1]
    matrices = []
    scalar_features = groups = None
    for cycle in range(n_cycle):
        targets = {name: values[..., cycle] for name, values in target_cycles.items()}
        matrix, current_features, current_groups, _ = build_decision_family(
            targets, strengths, nulls
        )
        if scalar_features is None:
            scalar_features = current_features
            groups = current_groups
        elif current_features != scalar_features or current_groups != groups:
            raise RuntimeError("decision-family mapping changed across cycles")
        matrices.append(matrix)
    return np.stack(matrices, axis=1), scalar_features, groups


def _feature_dict(feature) -> dict:
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


def _point_magnitude(matrix_mean, group) -> float:
    real = matrix_mean[group.real_index]
    imaginary = (
        matrix_mean[group.imaginary_index]
        if group.imaginary_index is not None
        else 0.0
    )
    return float(np.hypot(real, imaginary))


def _point_adequacy_ratios(matrix_mean, groups) -> dict:
    keys = sorted(
        {
            (group.target, float(group.prefix_upper_strength))
            for group in groups
            if group.section == "adequacy"
        }
    )
    result = {}
    for target, upper in keys:
        selected = [
            group
            for group in groups
            if group.section == "adequacy"
            and group.target == target
            and group.prefix_upper_strength == upper
            and not group.structural_null
        ]
        low = max(
            _point_magnitude(matrix_mean, group)
            for group in selected
            if group.contribution == "low"
        )
        higher = max(
            _point_magnitude(matrix_mean, group)
            for group in selected
            if group.contribution == "higher"
        )
        result.setdefault(target, {})[str(upper)] = (
            higher / low if low > 0 else None
        )
    return result


def _window_statistics(matrix, length: int, reference_critical: float, groups):
    n_block, n_cycle, _ = matrix.shape
    if n_cycle % length:
        raise ValueError("window lengths must divide the retained cycle count")
    windows = np.stack(
        [matrix[:, start : start + length].mean(axis=1) for start in range(0, n_cycle, length)],
        axis=1,
    )
    variance = np.var(windows, axis=0, ddof=1)
    median_variance = np.median(variance, axis=0)
    standard_error = np.sqrt(variance / n_block)
    window_means = windows.mean(axis=0)
    if windows.shape[1] > 1:
        mean_position_variance = np.var(window_means, axis=0, ddof=1)
        adjacent_ratios = []
        for index in range(windows.shape[1] - 1):
            difference = windows[:, index + 1] - windows[:, index]
            difference_se = difference.std(axis=0, ddof=1) / np.sqrt(n_block)
            adjacent_ratios.append(
                np.divide(
                    np.abs(difference.mean(axis=0)),
                    difference_se,
                    out=np.zeros_like(difference_se),
                    where=difference_se > 0,
                )
            )
        adjacent_ratios = np.stack(adjacent_ratios)
        adjacent_summary = {
            "pair_count": adjacent_ratios.shape[0],
            "standardized_absolute_drift_quantiles": np.quantile(
                adjacent_ratios, [0.5, 0.9, 0.95, 0.99]
            ),
            "maximum_standardized_absolute_drift": float(adjacent_ratios.max()),
            "count_above_reference_critical": int(
                np.sum(adjacent_ratios > reference_critical)
            ),
            "comparison_count": int(adjacent_ratios.size),
        }
    else:
        mean_position_variance = np.zeros(matrix.shape[-1])
        adjacent_summary = None
    return {
        "window_count": windows.shape[1],
        "median_cross_block_variance_realified": median_variance,
        "median_cross_block_standard_error_realified": np.median(
            standard_error, axis=0
        ),
        "median_standard_error_rms": float(
            np.sqrt(np.mean(median_variance / n_block))
        ),
        "window_mean_position_variance_realified": mean_position_variance,
        "window_mean_position_spread_rms": float(
            np.sqrt(np.mean(mean_position_variance))
        ),
        "position_spread_over_median_standard_error_rms": (
            float(
                np.sqrt(np.mean(mean_position_variance))
                / np.sqrt(np.mean(median_variance / n_block))
            )
            if np.any(median_variance > 0)
            else None
        ),
        "adjacent_window_drift": adjacent_summary,
        "point_adequacy_ratios_by_window": [
            _point_adequacy_ratios(window_means[index], groups)
            for index in range(windows.shape[1])
        ],
    }


def _compact_bootstrap(result) -> dict:
    return {
        "coverage": result["coverage"],
        "null_control_exclusions": result["null_control_exclusions"],
        "identification": {
            target: {
                strength: values["identified"]
                for strength, values in by_strength.items()
            }
            for target, by_strength in result["identification"].items()
        },
        "adequacy": {
            target: {
                strength: {
                    "dominant_low_order_magnitude_lower": values[
                        "dominant_low_order_magnitude_lower"
                    ],
                    "maximum_higher_order_magnitude_upper": values[
                        "maximum_higher_order_magnitude_upper"
                    ],
                    "higher_to_dominant_low_upper": values[
                        "higher_to_dominant_low_upper"
                    ],
                    "denominator_status": values["denominator_status"],
                    "adequate": values["adequate"],
                }
                for strength, values in by_strength.items()
            }
            for target, by_strength in result["adequacy"].items()
        },
        "identification_window": result["identification_window"],
    }


def _coordinate_bounds(
    mean,
    standard_error,
    alpha,
    critical,
    cycles,
    reference_observation_cycles,
    group,
):
    bounds = []
    indices = [group.real_index]
    if group.imaginary_index is not None:
        indices.append(group.imaginary_index)
    for index in indices:
        half_width = (
            critical
            * standard_error[index]
            * (cycles / reference_observation_cycles) ** (-alpha[index] / 2)
        )
        lower = mean[index] - half_width
        upper = mean[index] + half_width
        distance = 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper))
        extent = max(abs(lower), abs(upper))
        bounds.append((distance, extent))
    return (
        math.sqrt(sum(value[0] ** 2 for value in bounds)),
        math.sqrt(sum(value[1] ** 2 for value in bounds)),
    )


def _coordinate_bounds_scale(mean, standard_error, critical, scale, group):
    bounds = []
    indices = [group.real_index]
    if group.imaginary_index is not None:
        indices.append(group.imaginary_index)
    for index in indices:
        half_width = critical * standard_error[index] * scale
        lower = mean[index] - half_width
        upper = mean[index] + half_width
        distance = 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper))
        extent = max(abs(lower), abs(upper))
        bounds.append((distance, extent))
    return (
        math.sqrt(sum(value[0] ** 2 for value in bounds)),
        math.sqrt(sum(value[1] ** 2 for value in bounds)),
    )


def _projected_ratio(
    mean,
    standard_error,
    alpha,
    critical,
    groups,
    target,
    upper,
    cycles,
    reference_observation_cycles,
):
    selected = [
        group
        for group in groups
        if group.section == "adequacy"
        and group.target == target
        and group.prefix_upper_strength == upper
        and not group.structural_null
    ]
    low = max(
        _coordinate_bounds(
            mean,
            standard_error,
            alpha,
            critical,
            cycles,
            reference_observation_cycles,
            group,
        )[0]
        for group in selected
        if group.contribution == "low"
    )
    higher = max(
        _coordinate_bounds(
            mean,
            standard_error,
            alpha,
            critical,
            cycles,
            reference_observation_cycles,
            group,
        )[1]
        for group in selected
        if group.contribution == "higher"
    )
    return higher / low if low > 0 else math.inf


def _required_cycles(
    mean,
    standard_error,
    alpha,
    critical,
    groups,
    target,
    upper,
    limit,
    reference_observation_cycles,
):
    point_ratio = _projected_ratio(
        mean,
        standard_error,
        alpha,
        critical,
        groups,
        target,
        upper,
        1e15,
        reference_observation_cycles,
    )
    if point_ratio >= limit:
        return point_ratio, None
    lower = float(reference_observation_cycles)
    upper_cycles = lower
    while (
        _projected_ratio(
            mean,
            standard_error,
            alpha,
            critical,
            groups,
            target,
            upper,
            upper_cycles,
            reference_observation_cycles,
        )
        > limit
        and upper_cycles < 1e9
    ):
        upper_cycles *= 2
    for _ in range(80):
        middle = math.sqrt(lower * upper_cycles)
        ratio = _projected_ratio(
            mean,
            standard_error,
            alpha,
            critical,
            groups,
            target,
            upper,
            middle,
            reference_observation_cycles,
        )
        if ratio <= limit:
            upper_cycles = middle
        else:
            lower = middle
    return point_ratio, int(math.ceil(upper_cycles))


def _projected_ratio_blocks(
    mean,
    standard_error,
    critical,
    groups,
    target,
    upper,
    block_count,
    current_block_count,
):
    scale = math.sqrt(current_block_count / block_count)
    selected = [
        group
        for group in groups
        if group.section == "adequacy"
        and group.target == target
        and group.prefix_upper_strength == upper
        and not group.structural_null
    ]
    low = max(
        _coordinate_bounds_scale(mean, standard_error, critical, scale, group)[0]
        for group in selected
        if group.contribution == "low"
    )
    higher = max(
        _coordinate_bounds_scale(mean, standard_error, critical, scale, group)[1]
        for group in selected
        if group.contribution == "higher"
    )
    return higher / low if low > 0 else math.inf


def _required_blocks(
    mean,
    standard_error,
    critical,
    groups,
    target,
    upper,
    limit,
    current_block_count,
):
    point_ratio = _projected_ratio_blocks(
        mean,
        standard_error,
        critical,
        groups,
        target,
        upper,
        1e15,
        current_block_count,
    )
    if point_ratio >= limit:
        return None
    lower = float(current_block_count)
    upper_blocks = lower
    while (
        _projected_ratio_blocks(
            mean,
            standard_error,
            critical,
            groups,
            target,
            upper,
            upper_blocks,
            current_block_count,
        )
        > limit
        and upper_blocks < 1e9
    ):
        upper_blocks *= 2
    for _ in range(80):
        middle = math.sqrt(lower * upper_blocks)
        ratio = _projected_ratio_blocks(
            mean,
            standard_error,
            critical,
            groups,
            target,
            upper,
            middle,
            current_block_count,
        )
        if ratio <= limit:
            upper_blocks = middle
        else:
            lower = middle
    return int(math.ceil(upper_blocks))


def analyze_observation_efficiency(raw, config, parent_manifest) -> dict:
    """Analyze nested and non-overlapping cycle windows without new trajectories."""
    strengths, target_cycles = _target_cycles(raw)
    block_ids = tuple(int(value) for value in np.asarray(raw["block_ids"]))
    bootstrap_config = config["bootstrap"]
    nulls = bootstrap_config["structural_null_outputs"]
    matrix, scalar_features, groups = _cycle_family(target_cycles, strengths, nulls)
    n_block, n_cycle, n_feature = matrix.shape
    n_harmonic = len(np.asarray(raw["harmonics"]))

    def raw_summary_bytes(block_count, cycle_count):
        complex_count = (
            2
            * block_count
            * len(strengths)
            * 3
            * cycle_count
            * n_harmonic
            + block_count * 3 * cycle_count * n_harmonic
        )
        return int(complex_count * np.dtype(np.complex128).itemsize)
    lengths = np.asarray(config["cycle_lengths"], dtype=int)
    if (
        lengths.ndim != 1
        or len(lengths) < 2
        or np.any(lengths <= 0)
        or np.any(np.diff(lengths) <= 0)
        or lengths[-1] != n_cycle
        or any(n_cycle % int(length) for length in lengths)
    ):
        raise ValueError("cycle_lengths must be increasing divisors ending at n_cycle")

    reference_critical = float(
        parent_manifest["confirmed_block_bootstrap"]["coverage"]["critical_value"]
    )
    windows = {
        str(int(length)): _window_statistics(
            matrix, int(length), reference_critical, groups
        )
        for length in lengths
    }
    median_variances = np.stack(
        [windows[str(int(length))]["median_cross_block_variance_realified"] for length in lengths]
    )
    valid = np.all(median_variances > 0, axis=0)
    alpha = np.full(n_feature, np.nan)
    alpha[valid] = -np.polyfit(
        np.log(lengths.astype(float)), np.log(median_variances[:, valid]), 1
    )[0]
    single_variance = median_variances[0]
    variance_inflation = np.divide(
        lengths[:, None] * median_variances,
        single_variance[None],
        out=np.full_like(median_variances, np.nan),
        where=single_variance[None] > 0,
    )

    nested = {}
    for length in config["bootstrap_cycle_lengths"]:
        length = int(length)
        targets = {
            name: values[..., :length].mean(axis=-1)
            for name, values in target_cycles.items()
        }
        result = analyze_block_bootstrap(
            block_ids,
            strengths,
            targets,
            nulls,
            confidence=float(bootstrap_config["confidence"]),
            resamples=int(bootstrap_config["resamples"]),
            root_entropy=bootstrap_config["root_entropy"],
            batch_size=int(bootstrap_config["batch_size"]),
            higher_order_fraction_limit=float(
                bootstrap_config["higher_order_fraction_limit"]
            ),
        )
        nested[str(length)] = _compact_bootstrap(result)

    full_matrix = matrix.mean(axis=1)
    mean = full_matrix.mean(axis=0)
    standard_error = full_matrix.std(axis=0, ddof=1) / np.sqrt(n_block)
    limit = float(bootstrap_config["higher_order_fraction_limit"])
    projections = {}
    candidate_feature_scaling = {}
    for candidate in config["projection_candidates"]:
        target = candidate["target"]
        upper = float(candidate["prefix_upper_strength"])
        group_summaries = []
        for group in groups:
            if (
                group.section != "adequacy"
                or group.target != target
                or group.prefix_upper_strength != upper
                or group.structural_null
            ):
                continue
            indices = [group.real_index]
            parts = ["real"]
            if group.imaginary_index is not None:
                indices.append(group.imaginary_index)
                parts.append("imaginary")
            order_index = 0 if group.contribution == "low" else 1
            order = TARGET_ORDERS[target][order_index]
            group_summaries.append(
                {
                    "observable": group.observable,
                    "contribution": group.contribution,
                    "strength_power_order": order,
                    "point_magnitude_at_prefix_upper": _point_magnitude(mean, group),
                    "coordinate_parts": parts,
                    "contribution_coordinate_mean": mean[indices],
                    "contribution_coordinate_standard_error": standard_error[indices],
                    "coefficient_coordinate_standard_error": (
                        standard_error[indices] / upper**order
                    ),
                    "variance_exponent_alpha": alpha[indices],
                    "correlation_inflation_at_reference_cycles": variance_inflation[
                        -1, indices
                    ],
                }
            )
        candidate_feature_scaling.setdefault(target, {})[str(upper)] = group_summaries
        point_ratio, required_cycles = _required_cycles(
            mean,
            standard_error,
            alpha,
            reference_critical,
            groups,
            target,
            upper,
            limit,
            n_cycle,
        )
        projected_ratios = {
            str(int(length)): _projected_ratio(
                mean,
                standard_error,
                alpha,
                reference_critical,
                groups,
                target,
                upper,
                float(length),
                n_cycle,
            )
            for length in config["projection_cycle_options"]
        }
        required_blocks = _required_blocks(
            mean,
            standard_error,
            reference_critical,
            groups,
            target,
            upper,
            limit,
            n_block,
        )
        projections.setdefault(target, {})[str(upper)] = {
            "point_estimate_ratio_limit": point_ratio,
            "projected_required_cycles_at_fixed_blocks": required_cycles,
            "projected_required_blocks_at_fixed_cycles": required_blocks,
            "projected_upper_ratio_by_cycles": projected_ratios,
        }

    omega = float(config["protocol"]["omega"])
    discard = float(config["discard_time"])
    period = 2 * np.pi / omega
    baseline_runtime = float(parent_manifest["provenance"]["runtime_seconds_before_persistence"])
    baseline_horizon = discard + (n_cycle - 1 / int(config["n_phase"])) * period
    resource_options = {}
    for length in config["projection_cycle_options"]:
        horizon = discard + (int(length) - 1 / int(config["n_phase"])) * period
        factor = horizon / baseline_horizon
        resource_options[str(int(length))] = {
            "physical_integration_horizon": horizon,
            "relative_trajectory_horizon": factor,
            "projected_wall_seconds_from_parent_runtime": baseline_runtime * factor,
            "uncompressed_cycle_summary_bytes": raw_summary_bytes(
                n_block, int(length)
            ),
            "current_generator_materialization_lower_bound_bytes": (
                2 * raw_summary_bytes(n_block, int(length))
            ),
        }
    block_resource_options = {}
    required_cycle_resource_options = {}
    for target, by_prefix in projections.items():
        for upper, values in by_prefix.items():
            required_cycles = values["projected_required_cycles_at_fixed_blocks"]
            if required_cycles is not None:
                horizon = discard + (
                    required_cycles - 1 / int(config["n_phase"])
                ) * period
                required_cycle_resource_options.setdefault(target, {})[upper] = {
                    "cycle_count": required_cycles,
                    "physical_integration_horizon": horizon,
                    "relative_trajectory_horizon": horizon / baseline_horizon,
                    "projected_wall_seconds_from_parent_runtime": (
                        baseline_runtime * horizon / baseline_horizon
                    ),
                    "uncompressed_cycle_summary_bytes": raw_summary_bytes(
                        n_block, required_cycles
                    ),
                    "current_generator_materialization_lower_bound_bytes": (
                        2 * raw_summary_bytes(n_block, required_cycles)
                    ),
                }
            required_blocks = values["projected_required_blocks_at_fixed_cycles"]
            if required_blocks is None:
                continue
            block_resource_options.setdefault(target, {})[upper] = {
                "block_count": required_blocks,
                "relative_block_count": required_blocks / n_block,
                "projected_wall_seconds_from_parent_runtime": (
                    baseline_runtime * required_blocks / n_block
                ),
                "uncompressed_cycle_summary_bytes": raw_summary_bytes(
                    required_blocks, n_cycle
                ),
                "current_generator_materialization_lower_bound_bytes": (
                    2 * raw_summary_bytes(required_blocks, n_cycle)
                ),
            }

    section_masks = {
        section: np.asarray([feature.section == section for feature in scalar_features])
        for section in ("identification", "adequacy")
    }
    return {
        "interpretation": (
            "Exploratory reuse of retained cycle Fourier summaries. Cycle windows "
            "are averaged within each initial-state block and are never treated "
            "as independent replications. Cross-block variance, whole-block "
            "bootstrap, and window-position diagnostics remain separate."
        ),
        "parent_artifact": config["parent_artifact"],
        "family": {
            "block_count": n_block,
            "cycle_count": n_cycle,
            "reference_observation_cycles": n_cycle,
            "scalar_coordinate_count": n_feature,
            "members": [_feature_dict(feature) for feature in scalar_features],
        },
        "window_statistics": windows,
        "variance_scaling": {
            "cycle_lengths": lengths,
            "variance_exponent_alpha_realified": alpha,
            "alpha_quantiles_all": np.nanquantile(alpha, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]),
            "alpha_quantiles_by_section": {
                section: np.nanquantile(alpha[mask], [0.1, 0.5, 0.9])
                for section, mask in section_masks.items()
            },
            "variance_inflation_relative_to_shortest_window": variance_inflation,
            "variance_inflation_quantiles_by_length": {
                str(int(length)): np.nanquantile(
                    variance_inflation[index], [0.1, 0.5, 0.9]
                )
                for index, length in enumerate(lengths)
            },
        },
        "nested_prefix_bootstrap": nested,
        "adequacy_resource_projection": {
            "assumptions": (
                f"Full-{n_cycle}-cycle point estimates and max-statistic critical value "
                "are held fixed. Each scalar SE is extrapolated with its empirical "
                "non-overlapping-window variance exponent fitted on configured "
                "cycle lengths. This is a resource projection, not coverage "
                f"evidence beyond the observed {n_cycle} cycles."
            ),
            "candidates": projections,
            "candidate_feature_scaling": candidate_feature_scaling,
        },
        "resource_options": {
            "assumptions": (
                "Wall time scales linearly with the final physical integration "
                "horizon at fixed block/condition/worker counts, calibrated to "
                "the parent B=256 run. Solver and scheduling overhead may differ. "
                "Byte projections cover retained complex cycle summaries only. "
                "The current generator first retains a result list and then "
                "materializes a combined array, so its lower-bound peak is about "
                "twice final uncompressed summaries; endpoint runs require "
                "streaming, memmap, or online cycle-batch summaries."
            ),
            "by_cycle_count": resource_options,
            "by_projected_required_cycle_count": required_cycle_resource_options,
            "by_required_block_count": block_resource_options,
        },
    }


def _validate_reference_observation_cycles(parent_config, raw) -> int:
    """Match the declared parent observation length to retained raw summaries."""
    declared = parent_config.get("n_cycle")
    if (
        isinstance(declared, (bool, np.bool_))
        or not isinstance(declared, (int, np.integer))
        or declared <= 0
    ):
        raise ValueError("parent n_cycle must be a positive integer")
    positive = np.asarray(raw["positive_cycle_fourier"])
    if positive.ndim != 5:
        raise ValueError("parent positive cycle summaries must have five axes")
    retained = int(positive.shape[-2])
    if int(declared) != retained:
        raise ValueError(
            "parent config n_cycle does not match raw observation cycle axis"
        )
    return retained


def run_observation_efficiency_study(config_path) -> tuple[Path, dict, float]:
    """Analyze and persist an existing strength artifact without new integration."""
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    parent_dir = repo_root / config["parent_artifact"]
    parent_manifest_path = parent_dir / "manifest.json"
    parent_config_path = parent_dir / "config_snapshot.json"
    parent_raw_path = parent_dir / "raw_fourier_summaries.npz"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    if parent_manifest["files"]["config_snapshot.json"] != (
        f"sha256:{file_sha256(parent_config_path)}"
    ):
        raise ValueError("parent config hash does not match its manifest")
    if parent_manifest["files"]["raw_fourier_summaries.npz"] != (
        f"sha256:{file_sha256(parent_raw_path)}"
    ):
        raise ValueError("parent raw Fourier hash does not match its manifest")
    compatibility = {
        "discard_time": config["discard_time"] == parent_config["discard_time"],
        "n_phase": config["n_phase"] == parent_config["n_phase"],
        "omega": config["protocol"]["omega"]
        == parent_config["protocol"]["omega"],
        "direction": config["protocol"]["direction"]
        == parent_config["protocol"]["direction"],
        "bootstrap": config["bootstrap"] == parent_config["bootstrap"],
    }
    if not all(compatibility.values()):
        raise ValueError(f"analysis config is incompatible with parent: {compatibility}")
    for relative in (
        "src/lorenz/strength_bootstrap.py",
        "src/lorenz/strength_identifiability.py",
    ):
        expected = parent_manifest["provenance"]["code_identifiers"][relative]
        if expected != f"sha256:{file_sha256(repo_root / relative)}":
            raise ValueError(f"current {relative} does not match parent provenance")
    with np.load(parent_raw_path) as raw:
        _validate_reference_observation_cycles(parent_config, raw)
        derived = analyze_observation_efficiency(raw, config, parent_manifest)

    config_identifier = f"sha256:{file_sha256(config_path)}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = repo_root / config["output_root"] / (
        f"{timestamp}_{config_identifier[7:19]}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config_output = output_dir / "config_snapshot.json"
    derived_output = output_dir / "derived_diagnostics.json"
    write_json_atomic(config_output, config)
    write_json_atomic(derived_output, derived)

    runner_path = repo_root / config["runner_path"]
    git = git_provenance(repo_root)
    runtime = time.perf_counter() - start
    manifest = {
        "schema_version": 1,
        "classification": "exploratory_reanalysis",
        "study_id": config["study_id"],
        "files": {
            "config_snapshot.json": f"sha256:{file_sha256(config_output)}",
            "derived_diagnostics.json": f"sha256:{file_sha256(derived_output)}",
        },
        "parent_artifact": {
            "path": config["parent_artifact"],
            "manifest": f"sha256:{file_sha256(parent_manifest_path)}",
            "config_snapshot": f"sha256:{file_sha256(parent_config_path)}",
            "raw_fourier_summaries": f"sha256:{file_sha256(parent_raw_path)}",
            "compatibility_checks": compatibility,
        },
        "provenance": {
            "config_identifier": config_identifier,
            "code_identifiers": active_source_identifiers(repo_root, runner_path),
            "git_head": git["head"],
            "environment": environment_provenance(),
            "runtime_seconds_before_persistence": runtime,
        },
        "interpretation": derived["interpretation"],
    }
    manifest_output = output_dir / "manifest.json"
    write_json_atomic(manifest_output, manifest)
    return output_dir, manifest, time.perf_counter() - start
