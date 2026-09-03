#!/usr/bin/env python3
"""Estimate single-axis response coefficients from completed probe artifacts.

No trajectories are integrated.  The analysis consumes the crossed block-level
known-frequency coefficients already stored by the omega=5.938 high-order
probe lineage, uses the x/y/z forcing directions only, and retains the common
block as the uncertainty unit across all amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-matplotlib")
import matplotlib.pyplot as plt

from lorenz.artifacts import file_sha256, verify_file_identifiers, write_json_atomic
from lorenz.response import paired_order_contrasts
from lorenz.response_plots import apply_style
from lorenz.retention import paired_condition_indices, paired_condition_vectors
from lorenz.strength_series import (
    crossed_block_gls_fit,
    joint_crossed_block_model_comparison,
)


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "outputs/production/high_order_probe_omega_5p938_h10_h12_h14_h16_extension_v1"
    / "20260822T061155_b1e645f07e98"
)
DEFAULT_ADDITIONAL_ARTIFACTS = (
    REPO_ROOT
    / "outputs/production/single_axis_coefficients_omega_5p938_h5_h6_h7_v1"
    / "20260824T074945_7878c8a09d92",
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/analysis/single_axis_coefficients_omega_5p938_h5_h6_h7_v2"
)
STATE_NAMES = ("x", "y", "z")
CONFIDENCE = 0.95
LACK_OF_FIT_ALPHA = 0.05
MAX_DEGREE = 4
OUTPUT_FILES = {
    "coefficient_estimates.json",
    "first_order_matrix.csv",
    "first_order_amplitude_dependence.csv",
    "second_order_diagonal.csv",
    "second_order_amplitude_dependence.csv",
    "fit_range_sensitivity.csv",
    "first_order_amplitude_dependence.png",
    "second_order_amplitude_dependence.png",
    "second_order_fit_sensitivity.png",
    "intercept_stability.json",
    "intercept_stability.csv",
    "intercept_stability_summary.csv",
    "first_order_intercept_stability.png",
    "second_order_intercept_stability.png",
    "quadratic_extrapolation_curves.png",
    "quadratic_regime_diagnostic.json",
    "quadratic_regime_fits.csv",
    "quadratic_regime_Q_z_xx.png",
    "quadratic_regime_Q_z_yy.png",
    "quadratic_regime_Q_z_zz.png",
    "analysis_manifest.json",
}


def _prefix(omega: float) -> str:
    return f"omega_{format(omega, '.12g').replace('-', 'm').replace('.', 'p')}"


def _resolve_artifact(value) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def artifact_lineage(artifact: Path) -> list[dict]:
    """Verify and describe the base/extension chain without double counting it."""
    lineage = []
    current = artifact.resolve()
    seen = set()
    while True:
        if current in seen:
            raise ValueError("artifact extension lineage contains a cycle")
        seen.add(current)
        manifest_path = current / "manifest.json"
        config_path = current / "config_snapshot.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        verify_file_identifiers(current, manifest["files"])
        lineage.append({
            "artifact": str(current),
            "manifest_identifier": f"sha256:{file_sha256(manifest_path)}",
            "strengths": [float(value) for value in config["strengths"]],
            "extension_new_strengths": [
                float(value)
                for value in (config.get("extension") or {}).get("new_strengths", [])
            ],
        })
        extension = config.get("extension")
        if extension is None:
            break
        current = _resolve_artifact(extension["base_artifact"])
    return lineage


def load_completed_contrasts(artifact: Path) -> dict:
    """Load signed-axis n=1 odd and n=2 even contrasts for every block."""
    config = json.loads(
        (artifact / "config_snapshot.json").read_text(encoding="utf-8")
    )
    omega = float(config["omega"])
    prefix = _prefix(omega)
    with np.load(
        artifact / "checkpoints" / f"{prefix}.npz", allow_pickle=False
    ) as checkpoint:
        means = np.asarray(checkpoint[f"{prefix}_condition_means"])
        condition_vectors = np.asarray(
            checkpoint[f"{prefix}_condition_vectors"], dtype=float
        )
        strengths = np.asarray(checkpoint[f"{prefix}_strengths"], dtype=float)
        harmonics = np.asarray(checkpoint[f"{prefix}_harmonics"], dtype=int)
        block_ids = np.asarray(checkpoint[f"{prefix}_block_ids"], dtype=np.uint32)
    if omega != 5.938:
        raise ValueError("this analysis is defined for the selected omega=5.938 run")
    if means.shape != (
        len(block_ids), len(condition_vectors), 3, len(harmonics)
    ) or not np.iscomplexobj(means):
        raise ValueError("checkpoint condition means have invalid axes")
    if len(set(block_ids.tolist())) != len(block_ids):
        raise ValueError("checkpoint block IDs are not unique")
    harmonic_index = {int(value): index for index, value in enumerate(harmonics)}
    if 1 not in harmonic_index or 2 not in harmonic_index:
        raise ValueError("checkpoint must retain harmonics n=1 and n=2")
    directions = np.asarray(config["protocol"]["directions"], dtype=float)
    expected_axes = np.eye(3)
    if directions.shape[0] < 3 or not np.allclose(
        directions[:3], expected_axes, rtol=0.0, atol=1e-14
    ):
        raise ValueError("the first three forcing directions are not x, y, z")
    expected_vectors = paired_condition_vectors(
        directions, strengths, include_unforced=True
    )
    if not np.allclose(
        condition_vectors, expected_vectors, rtol=0.0, atol=1e-14
    ):
        raise ValueError("checkpoint condition axis is not canonical")

    odd_fundamental = np.empty(
        (len(block_ids), len(strengths), 3, 3), dtype=complex
    )
    even_second = np.empty_like(odd_fundamental)
    unforced = means[:, 0]
    for input_index, direction in enumerate(expected_axes):
        for strength_index, strength in enumerate(strengths):
            positive, negative = paired_condition_indices(
                condition_vectors, direction, float(strength)
            )
            odd, even = paired_order_contrasts(
                means[:, positive], means[:, negative], unforced
            )
            odd_fundamental[:, strength_index, :, input_index] = odd[
                :, :, harmonic_index[1]
            ]
            even_second[:, strength_index, :, input_index] = even[
                :, :, harmonic_index[2]
            ]
    return {
        "artifact": artifact,
        "config": config,
        "omega": omega,
        "strengths": strengths,
        "block_ids": block_ids,
        "unforced": unforced,
        "unforced_second": unforced[:, :, harmonic_index[2]],
        "odd_fundamental": odd_fundamental,
        "even_second": even_second,
    }


def _combination_settings(config: dict) -> dict:
    """Settings that must agree for blockwise amplitude combination."""
    inference = config.get("inference") or {}
    return {
        "omega": config["omega"],
        "block_count": config["block_count"],
        "discard_time": config["discard_time"],
        "n_phase": config["n_phase"],
        "harmonics": config["harmonics"],
        "observation_rule": config["observation_rule"],
        "lorenz": config["lorenz"],
        "solver": config["solver"],
        "initial_ensemble": config["initial_ensemble"],
        "phase": config["protocol"].get("phase", 0.0),
        "dense": config["dense"],
        "retention": config["retention"],
        "confidence": inference.get("confidence", 0.95),
        "fdr_alpha": inference.get("fdr_alpha", 0.05),
    }


def combine_completed_contrasts(datasets: list[dict]) -> dict:
    """Combine disjoint amplitudes after exact block and protocol checks."""
    if not datasets:
        raise ValueError("at least one completed artifact is required")
    reference = datasets[0]
    reference_settings = _combination_settings(reference["config"])
    unforced_differences = []
    all_strengths = []
    odd_pieces = []
    even_pieces = []
    seen_strengths = []
    for dataset_index, dataset in enumerate(datasets):
        if _combination_settings(dataset["config"]) != reference_settings:
            raise ValueError(
                f"artifact settings differ: {dataset['artifact']}"
            )
        if not np.array_equal(dataset["block_ids"], reference["block_ids"]):
            raise ValueError(
                f"artifact block IDs differ: {dataset['artifact']}"
            )
        directions = np.asarray(
            dataset["config"]["protocol"]["directions"], dtype=float
        )
        if dataset_index > 0 and not np.allclose(
            directions, np.eye(3), rtol=0.0, atol=1e-14
        ):
            raise ValueError(
                f"additional artifact is not exactly the x/y/z design: "
                f"{dataset['artifact']}"
            )
        difference = float(np.max(np.abs(
            dataset["unforced"] - reference["unforced"]
        )))
        if not np.allclose(
            dataset["unforced"], reference["unforced"],
            rtol=1e-10, atol=1e-11,
        ):
            raise ValueError(
                f"artifact unforced coefficients differ: {dataset['artifact']}"
            )
        unforced_differences.append({
            "artifact": str(dataset["artifact"]),
            "max_abs_difference_from_primary": difference,
        })
        for strength in dataset["strengths"]:
            if any(np.isclose(strength, prior, rtol=0.0, atol=1e-12)
                   for prior in seen_strengths):
                raise ValueError(f"artifact strengths overlap at h={strength:g}")
            seen_strengths.append(float(strength))
        all_strengths.append(dataset["strengths"])
        odd_pieces.append(dataset["odd_fundamental"])
        # Match extension semantics even if independently repeated unforced
        # integration differs only within the accepted numerical tolerance.
        unforced_adjustment = (
            dataset["unforced_second"] - reference["unforced_second"]
        )[:, None, :, None]
        even_pieces.append(dataset["even_second"] + unforced_adjustment)
    strengths = np.concatenate(all_strengths)
    order = np.argsort(strengths)
    return {
        "omega": reference["omega"],
        "strengths": strengths[order],
        "block_ids": reference["block_ids"],
        "odd_fundamental": np.concatenate(odd_pieces, axis=1)[:, order],
        "even_second": np.concatenate(even_pieces, axis=1)[:, order],
        "combination_audit": {
            "status": "accepted",
            "block_ids_exact": True,
            "settings_exact": True,
            "unforced_rtol": 1e-10,
            "unforced_atol": 1e-11,
            "unforced_comparisons": unforced_differences,
        },
    }


def signed_parts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine and sine amplitudes for coefficient = cos - i*sin."""
    values = np.asarray(values)
    return values.real, -values.imag


def scalar_gls_fit(
    block_values: np.ndarray,
    strengths: np.ndarray,
    *,
    base_order: int,
    degree: int,
) -> dict:
    """Fit one signed raw contrast using its crossed-strength block covariance."""
    orders = tuple(base_order + 2 * index for index in range(degree))
    return crossed_block_gls_fit(block_values, strengths, orders)


def interval(values: np.ndarray, confidence: float = CONFIDENCE) -> dict:
    """Point estimate and t interval over the leading block axis."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    return {
        "estimate": mean,
        "standard_error": standard_error,
        "ci95": [mean - critical * standard_error, mean + critical * standard_error],
    }


def _feature_matrix(raw: np.ndarray, pairs) -> np.ndarray:
    features = []
    for output_index, input_index in pairs:
        cosine, sine = signed_parts(raw[:, :, output_index, input_index])
        features.extend((cosine, sine))
    return np.stack(features, axis=2)


def joint_model_comparison(
    raw: np.ndarray,
    strengths: np.ndarray,
    *,
    base_order: int,
    pairs,
) -> list[dict]:
    """Global lack-of-fit comparison on symmetry-allowed signed components."""
    features = _feature_matrix(raw, pairs)
    strength_count = features.shape[1]
    # Retain at least one amplitude residual for a lack-of-fit statistic.
    maximum_degree = min(MAX_DEGREE, strength_count - 1)
    comparisons = joint_crossed_block_model_comparison(
        features,
        strengths,
        tuple(
            tuple(base_order + 2 * index for index in range(degree))
            for degree in range(1, maximum_degree + 1)
        ),
    )
    for degree, comparison in enumerate(comparisons, start=1):
        comparison["degree"] = degree
        comparison["raw_powers"] = list(comparison.pop("orders"))
    return comparisons


def select_degree(comparisons: list[dict]) -> int:
    """Select an adequate model, retaining supported subsequent terms."""
    selected = None
    for index, comparison in enumerate(comparisons):
        if comparison["lack_of_fit_p"] >= LACK_OF_FIT_ALPHA:
            selected = index
            break
    if selected is None:
        raise ValueError("no candidate power series passes the lack-of-fit check")
    while (
        selected + 1 < len(comparisons)
        and comparisons[selected + 1]["nested_improvement_p"] < LACK_OF_FIT_ALPHA
    ):
        selected += 1
    return int(comparisons[selected]["degree"])


def estimate_coefficients(raw, strengths, *, base_order: int, degree: int):
    entries = []
    fits = {}
    for output_index, output in enumerate(STATE_NAMES):
        for input_index, input_name in enumerate(STATE_NAMES):
            cosine, sine = signed_parts(raw[:, :, output_index, input_index])
            part_results = {}
            for part, block_values in (("cos", cosine), ("sin", sine)):
                fit = scalar_gls_fit(
                    block_values,
                    strengths,
                    base_order=base_order,
                    degree=degree,
                )
                part_results[part] = interval(fit["block_coefficients"][:, 0])
                fits[(output_index, input_index, part)] = fit
            entries.append({
                "output": output,
                "input": input_name,
                "cos": part_results["cos"],
                "sin": part_results["sin"],
            })
    return entries, fits


def amplitude_dependence(raw, strengths) -> list[dict]:
    rows = []
    block_count = raw.shape[0]
    critical = float(stats.t.ppf(0.975, block_count - 1))
    for output_index, output in enumerate(STATE_NAMES):
        for input_index, input_name in enumerate(STATE_NAMES):
            cosine, sine = signed_parts(raw[:, :, output_index, input_index])
            for strength_index, strength in enumerate(strengths):
                for part, values in (
                    ("cos", cosine[:, strength_index]),
                    ("sin", sine[:, strength_index]),
                ):
                    mean = float(values.mean())
                    se = float(values.std(ddof=1) / np.sqrt(block_count))
                    rows.append({
                        "output": output,
                        "input": input_name,
                        "strength": float(strength),
                        "part": part,
                        "estimate": mean,
                        "ci95_low": mean - critical * se,
                        "ci95_high": mean + critical * se,
                    })
    return rows


def fit_range_sensitivity(raw, strengths, *, base_order: int, degree: int):
    rows = []
    ranges = [
        np.arange(stop) for stop in range(degree + 2, len(strengths) + 1)
    ]
    ranges.extend((np.arange(1, len(strengths)), np.arange(2, len(strengths))))
    for output_index, output in enumerate(STATE_NAMES):
        for input_index, input_name in enumerate(STATE_NAMES):
            cosine, sine = signed_parts(raw[:, :, output_index, input_index])
            for part, all_values in (("cos", cosine), ("sin", sine)):
                for indices in ranges:
                    fit = scalar_gls_fit(
                        all_values[:, indices],
                        strengths[indices],
                        base_order=base_order,
                        degree=degree,
                    )
                    estimate = interval(fit["block_coefficients"][:, 0])
                    rows.append({
                        "output": output,
                        "input": input_name,
                        "part": part,
                        "degree": degree,
                        "strength_min": float(strengths[indices[0]]),
                        "strength_max": float(strengths[indices[-1]]),
                        "strength_count": len(indices),
                        "estimate": estimate["estimate"],
                        "ci95_low": estimate["ci95"][0],
                        "ci95_high": estimate["ci95"][1],
                    })
    return rows


def robustness_windows(strengths: np.ndarray) -> list[dict]:
    """Common windows for order and upper/lower amplitude sensitivity."""
    strengths = np.asarray(strengths, dtype=float)
    windows = []
    # Four amplitudes leave at least one residual amplitude after the
    # three-coefficient model; use the same windows for both model orders.
    for start in (0, 1):
        first_stop = start + 4
        for stop in range(first_stop, len(strengths) + 1):
            indices = np.arange(start, stop)
            windows.append({
                "label": f"{strengths[start]:g}-{strengths[stop - 1]:g}",
                "indices": indices,
                "strength_min": float(strengths[start]),
                "strength_max": float(strengths[stop - 1]),
                "excluded_smallest": bool(start == 1),
                "excluded_largest_count": int(len(strengths) - stop),
            })
    return windows


def intercept_robustness_records(
    raw,
    strengths,
    *,
    family: str,
    base_order: int,
    allowed_pairs,
) -> list[dict]:
    """Return every requested order/window intercept and block-level CI."""
    rows = []
    models = (
        (2, [base_order, base_order + 2]),
        (3, [base_order, base_order + 2, base_order + 4]),
    )
    for output_index, input_index in allowed_pairs:
        cosine, sine = signed_parts(raw[:, :, output_index, input_index])
        for part, all_values in (("cos", cosine), ("sin", sine)):
            for degree, raw_powers in models:
                for window in robustness_windows(strengths):
                    indices = window["indices"]
                    fit = scalar_gls_fit(
                        all_values[:, indices],
                        strengths[indices],
                        base_order=base_order,
                        degree=degree,
                    )
                    estimate = interval(fit["block_coefficients"][:, 0])
                    rows.append({
                        "family": family,
                        "output": STATE_NAMES[output_index],
                        "input": STATE_NAMES[input_index],
                        "part": part,
                        "degree": degree,
                        "raw_powers": raw_powers,
                        "window": window["label"],
                        "included_strengths": [
                            float(value) for value in strengths[indices]
                        ],
                        "strength_min": window["strength_min"],
                        "strength_max": window["strength_max"],
                        "excluded_smallest": window["excluded_smallest"],
                        "excluded_largest_count": window["excluded_largest_count"],
                        "estimate": estimate["estimate"],
                        "standard_error": estimate["standard_error"],
                        "ci95_low": estimate["ci95"][0],
                        "ci95_high": estimate["ci95"][1],
                    })
    return rows


def summarize_intercept_robustness(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate polynomial-order, amplitude-window, and sampling scales."""
    part_summaries = []
    part_keys = sorted({
        (row["family"], row["output"], row["input"], row["part"])
        for row in rows
    })
    classification_rank = {"stable": 0, "borderline": 1, "model_sensitive": 2}
    for family, output, input_name, part in part_keys:
        selected = [
            row for row in rows
            if (row["family"], row["output"], row["input"], row["part"])
            == (family, output, input_name, part)
        ]
        by_key = {(row["degree"], row["window"]): row for row in selected}
        windows = sorted({row["window"] for row in selected})
        order_shifts = [
            abs(by_key[(2, window)]["estimate"] - by_key[(3, window)]["estimate"])
            for window in windows
        ]
        window_spans = {}
        for degree in (2, 3):
            estimates = [
                row["estimate"] for row in selected if row["degree"] == degree
            ]
            window_spans[str(degree)] = max(estimates) - min(estimates)
        half_widths = [
            (row["ci95_high"] - row["ci95_low"]) / 2 for row in selected
        ]
        sampling_half_width = float(np.median(half_widths))
        order_shift = float(max(order_shifts))
        window_shift = float(max(window_spans.values()))
        order_ratio = order_shift / sampling_half_width
        window_ratio = window_shift / sampling_half_width
        common_ci_overlap = bool(
            max(row["ci95_low"] for row in selected)
            <= min(row["ci95_high"] for row in selected)
        )
        worst_ratio = max(order_ratio, window_ratio)
        if common_ci_overlap and worst_ratio <= 1.25:
            classification = "stable"
        elif common_ci_overlap and worst_ratio <= 2.0:
            classification = "borderline"
        else:
            classification = "model_sensitive"
        if order_ratio <= 1.0 and window_ratio <= 1.0:
            dominant_source = "sampling_uncertainty"
        elif order_ratio > 1.15 * window_ratio:
            dominant_source = "polynomial_order"
        elif window_ratio > 1.15 * order_ratio:
            dominant_source = "amplitude_window"
        else:
            dominant_source = "polynomial_order_and_amplitude_window"
        estimates = [row["estimate"] for row in selected]
        part_summaries.append({
            "family": family,
            "output": output,
            "input": input_name,
            "part": part,
            "classification": classification,
            "dominant_source": dominant_source,
            "polynomial_order_shift_max": order_shift,
            "amplitude_window_span_max": window_shift,
            "sampling_ci95_half_width_median": sampling_half_width,
            "sampling_ci95_half_width_min": float(min(half_widths)),
            "sampling_ci95_half_width_max": float(max(half_widths)),
            "order_shift_to_sampling_half_width": order_ratio,
            "window_span_to_sampling_half_width": window_ratio,
            "intercept_min": float(min(estimates)),
            "intercept_max": float(max(estimates)),
            "all_fit_ci95_common_overlap": common_ci_overlap,
        })

    coefficient_summaries = []
    coefficient_keys = sorted({
        (row["family"], row["output"], row["input"])
        for row in part_summaries
    })
    for family, output, input_name in coefficient_keys:
        parts = [
            row for row in part_summaries
            if (row["family"], row["output"], row["input"])
            == (family, output, input_name)
        ]
        worst = max(parts, key=lambda row: classification_rank[row["classification"]])
        coefficient_summaries.append({
            "family": family,
            "output": output,
            "input": input_name,
            "classification": worst["classification"],
            "component_classifications": {
                row["part"]: row["classification"] for row in parts
            },
            "dominant_sources": sorted({row["dominant_source"] for row in parts}),
            "maximum_order_shift_to_sampling_half_width": max(
                row["order_shift_to_sampling_half_width"] for row in parts
            ),
            "maximum_window_span_to_sampling_half_width": max(
                row["window_span_to_sampling_half_width"] for row in parts
            ),
        })
    return part_summaries, coefficient_summaries


def quadratic_regime_diagnostic(raw, strengths, *, allowed_pairs) -> dict:
    """Fit the candidate h=4..8 regime and its h=4..16 reference."""
    strengths = np.asarray(strengths, dtype=float)
    low_indices = np.flatnonzero((strengths >= 4.0) & (strengths <= 8.0))
    full_indices = np.flatnonzero(strengths >= 4.0)
    plot_indices = np.flatnonzero(strengths <= 8.0)
    if not np.array_equal(strengths[low_indices], [4, 5, 6, 7, 8]):
        raise ValueError("quadratic-regime analysis requires h=4,5,6,7,8")
    records = []
    fits = {}
    for output_index, input_index in allowed_pairs:
        cosine, sine = signed_parts(raw[:, :, output_index, input_index])
        for part, values in (("cos", cosine), ("sin", sine)):
            for window, indices in (("low_4_8", low_indices), ("full_4_16", full_indices)):
                fit = scalar_gls_fit(
                    values[:, indices], strengths[indices],
                    base_order=2, degree=2,
                )
                q = interval(fit["block_coefficients"][:, 0])
                c4 = interval(fit["block_coefficients"][:, 1])
                fits[(output_index, input_index, part, window)] = fit
                records.append({
                    "output": STATE_NAMES[output_index],
                    "input": STATE_NAMES[input_index],
                    "part": part,
                    "window": window,
                    "included_strengths": [
                        float(value) for value in strengths[indices]
                    ],
                    "Q_estimate": q["estimate"],
                    "Q_standard_error": q["standard_error"],
                    "Q_ci95_low": q["ci95"][0],
                    "Q_ci95_high": q["ci95"][1],
                    "C4_estimate": c4["estimate"],
                    "C4_standard_error": c4["standard_error"],
                    "C4_ci95_low": c4["ci95"][0],
                    "C4_ci95_high": c4["ci95"][1],
                })
    return {
        "plot_indices": plot_indices,
        "low_indices": low_indices,
        "full_indices": full_indices,
        "records": records,
        "fits": fits,
        "low_model_comparison": joint_model_comparison(
            raw[:, low_indices], strengths[low_indices],
            base_order=2, pairs=allowed_pairs,
        ),
        "full_model_comparison": joint_model_comparison(
            raw[:, full_indices], strengths[full_indices],
            base_order=2, pairs=allowed_pairs,
        ),
    }


def susceptibility_conversions(first_entries, second_entries):
    first = []
    for entry in first_entries:
        # chi1 = 2i L.  If L=C-iS, chi1=(2S)-i(-2C).
        cos = entry["sin"]
        sin = entry["cos"]
        first.append({
            "output": entry["output"],
            "input": entry["input"],
            "lorenz_parity_allowed": entry["lorenz_parity_allowed"],
            "cos": {
                "estimate": 2 * cos["estimate"],
                "ci95": [2 * value for value in cos["ci95"]],
            },
            "sin": {
                "estimate": -2 * sin["estimate"],
                "ci95": [-2 * sin["ci95"][1], -2 * sin["ci95"][0]],
            },
        })
    second = []
    for entry in second_entries:
        converted = {
            "output": entry["output"],
            "input": entry["input"],
            "lorenz_parity_allowed": entry["lorenz_parity_allowed"],
        }
        for part in ("cos", "sin"):
            source = entry[part]
            converted[part] = {
                "estimate": -4 * source["estimate"],
                "ci95": [-4 * source["ci95"][1], -4 * source["ci95"][0]],
            }
        second.append(converted)
    return first, second


def _write_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def coefficient_rows(entries):
    rows = []
    for entry in entries:
        row = {
            "output": entry["output"],
            "input": entry["input"],
            "lorenz_parity_allowed": entry["lorenz_parity_allowed"],
        }
        for part in ("cos", "sin"):
            row[f"{part}_estimate"] = entry[part]["estimate"]
            row[f"{part}_ci95_low"] = entry[part]["ci95"][0]
            row[f"{part}_ci95_high"] = entry[part]["ci95"][1]
        rows.append(row)
    return rows


def plot_amplitude_dependence(
    normalized,
    strengths,
    fits,
    *,
    base_order: int,
    degree: int,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    critical = float(stats.t.ppf(0.975, normalized.shape[0] - 1))
    dense_h = np.linspace(float(strengths[0]), float(strengths[-1]), 240)
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True)
    colors = {"cos": "#245b9e", "sin": "#b55532"}
    for output_index, output in enumerate(STATE_NAMES):
        for input_index, input_name in enumerate(STATE_NAMES):
            axis = axes[output_index, input_index]
            cosine, sine = signed_parts(
                normalized[:, :, output_index, input_index]
            )
            for part, values in (("cos", cosine), ("sin", sine)):
                mean = values.mean(axis=0)
                half = critical * values.std(axis=0, ddof=1) / np.sqrt(len(values))
                axis.errorbar(
                    strengths,
                    mean,
                    yerr=half,
                    color=colors[part],
                    marker="o",
                    markersize=3.5,
                    linewidth=0.9,
                    capsize=2,
                    label=part if (output_index, input_index) == (0, 0) else None,
                )
                coefficients = fits[(output_index, input_index, part)][
                    "mean_coefficients"
                ]
                normalized_fit = sum(
                    coefficients[index] * dense_h ** (2 * index)
                    for index in range(degree)
                )
                axis.plot(dense_h, normalized_fit, color=colors[part], linewidth=1.2)
                axis.axhline(
                    coefficients[0], color=colors[part], linestyle=":", linewidth=0.7
                )
            axis.axhline(0.0, color="#777777", linewidth=0.6)
            axis.set_title(f"{output} output <- {input_name} forcing")
            if output_index == 2:
                axis.set_xlabel("forcing amplitude h")
            if input_index == 0:
                axis.set_ylabel(ylabel)
    figure.legend(loc="upper right", frameon=False, ncol=2)
    figure.suptitle(
        f"{title}: points are block means with pointwise 95% CI; "
        f"curves use raw powers {list(range(base_order, base_order + 2 * degree, 2))}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_second_order_sensitivity(rows, primary_entries, path: Path) -> None:
    selected = [
        row for row in rows
        if row["output"] == "z" and row["input"] in STATE_NAMES
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), sharex=True)
    for column, input_name in enumerate(STATE_NAMES):
        primary = next(
            item for item in primary_entries
            if item["output"] == "z" and item["input"] == input_name
        )
        for row_index, part in enumerate(("cos", "sin")):
            axis = axes[row_index, column]
            values = [
                row for row in selected
                if row["input"] == input_name and row["part"] == part
            ]
            labels = [
                f"p{item['degree']}:{item['strength_min']:g}-{item['strength_max']:g}"
                for item in values
            ]
            y = np.asarray([item["estimate"] for item in values])
            low = np.asarray([item["ci95_low"] for item in values])
            high = np.asarray([item["ci95_high"] for item in values])
            axis.errorbar(
                np.arange(len(values)), y,
                yerr=np.vstack((y - low, high - y)),
                color="#245b9e" if part == "cos" else "#b55532",
                marker="o", linewidth=0.9, capsize=2,
            )
            axis.axhline(primary[part]["estimate"], color="#222222", linestyle="--")
            axis.axhline(0.0, color="#888888", linewidth=0.6)
            axis.set_title(f"Q(z,{input_name}{input_name}) {part}")
            axis.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    figure.suptitle(
        "Quadratic-coefficient sensitivity to amplitude range and polynomial order"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_intercept_robustness(
    rows,
    coefficient_summaries,
    *,
    family: str,
    allowed_pairs,
    title: str,
    path: Path,
) -> None:
    """Plot every requested intercept and CI without selecting one fit."""
    windows = list(dict.fromkeys(
        row["window"] for row in rows
        if row["family"] == family and row["degree"] == 2
    ))
    figure, axes = plt.subplots(
        2, len(allowed_pairs),
        figsize=(3.7 * len(allowed_pairs), 6.8),
        squeeze=False,
    )
    model_style = {
        2: ("#245b9e", "lower order"),
        3: ("#b55532", "higher order"),
    }
    for column, (output_index, input_index) in enumerate(allowed_pairs):
        output = STATE_NAMES[output_index]
        input_name = STATE_NAMES[input_index]
        summary = next(
            item for item in coefficient_summaries
            if (item["family"], item["output"], item["input"])
            == (family, output, input_name)
        )
        for row_index, part in enumerate(("cos", "sin")):
            axis = axes[row_index, column]
            for degree, offset in ((2, -0.08), (3, 0.08)):
                selected = {
                    row["window"]: row for row in rows
                    if row["family"] == family
                    and row["output"] == output
                    and row["input"] == input_name
                    and row["part"] == part
                    and row["degree"] == degree
                }
                values = [selected[window] for window in windows]
                estimate = np.asarray([item["estimate"] for item in values])
                low = np.asarray([item["ci95_low"] for item in values])
                high = np.asarray([item["ci95_high"] for item in values])
                color, label = model_style[degree]
                axis.errorbar(
                    np.arange(len(windows)) + offset,
                    estimate,
                    yerr=np.vstack((estimate - low, high - estimate)),
                    color=color,
                    marker="o",
                    markersize=3.5,
                    linewidth=0.9,
                    capsize=2,
                    label=label if (row_index, column) == (0, 0) else None,
                )
            axis.axhline(0.0, color="#888888", linewidth=0.6)
            axis.set_title(
                f"{output}<-{input_name} {part} ({summary['classification']})",
                fontsize=9,
            )
            axis.set_xticks(
                np.arange(len(windows)), windows, rotation=40, ha="right"
            )
            if column == 0:
                axis.set_ylabel("h->0 intercept")
    figure.legend(loc="upper right", frameon=False, ncol=2)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_quadratic_extrapolation_curves(
    raw,
    strengths,
    *,
    allowed_pairs,
    path: Path,
) -> None:
    """Show Q(h) data and all requested order/window extrapolation curves."""
    normalized = raw / strengths[None, :, None, None] ** 2
    critical = float(stats.t.ppf(0.975, len(raw) - 1))
    dense_h = np.linspace(0.0, float(strengths[-1]), 300)
    colors = {2: "#245b9e", 3: "#b55532"}
    figure, axes = plt.subplots(2, len(allowed_pairs), figsize=(13.0, 7.0))
    for column, (output_index, input_index) in enumerate(allowed_pairs):
        cosine, sine = signed_parts(normalized[:, :, output_index, input_index])
        raw_cosine, raw_sine = signed_parts(raw[:, :, output_index, input_index])
        for row_index, (part, values, raw_values) in enumerate((
            ("cos", cosine, raw_cosine),
            ("sin", sine, raw_sine),
        )):
            axis = axes[row_index, column]
            mean = values.mean(axis=0)
            half = critical * values.std(axis=0, ddof=1) / np.sqrt(len(values))
            axis.errorbar(
                strengths, mean, yerr=half,
                color="#222222", marker="o", markersize=4,
                linestyle="", capsize=2, zorder=5,
                label="Q(h) block mean and 95% CI" if (row_index, column) == (0, 0) else None,
            )
            for degree in (2, 3):
                for window in robustness_windows(strengths):
                    indices = window["indices"]
                    fit = scalar_gls_fit(
                        raw_values[:, indices], strengths[indices],
                        base_order=2, degree=degree,
                    )
                    coefficients = fit["mean_coefficients"]
                    curve_h = dense_h[
                        dense_h <= window["strength_max"]
                    ]
                    curve = sum(
                        coefficients[index] * curve_h ** (2 * index)
                        for index in range(degree)
                    )
                    full_window = (
                        window["strength_min"] == float(strengths[0])
                        and window["strength_max"] == float(strengths[-1])
                    )
                    axis.plot(
                        curve_h,
                        curve,
                        color=colors[degree],
                        alpha=0.85 if full_window else 0.18,
                        linewidth=1.4 if full_window else 0.8,
                        linestyle="--" if window["excluded_smallest"] else "-",
                        label=(
                            "lower-order fits" if degree == 2 else "higher-order fits"
                        ) if (row_index, column, full_window) == (0, 0, True) else None,
                    )
            axis.axhline(0.0, color="#888888", linewidth=0.6)
            axis.set_xlim(0.0, float(strengths[-1]))
            axis.set_title(
                f"Q({STATE_NAMES[output_index]},{STATE_NAMES[input_index]}"
                f"{STATE_NAMES[input_index]}) {part}"
            )
            if row_index == 1:
                axis.set_xlabel("forcing amplitude h")
            if column == 0:
                axis.set_ylabel("even n=2 / h^2")
    figure.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        frameon=False, ncol=3,
    )
    figure.suptitle(
        "Allowed quadratic amplitude dependence and all requested h->0 extrapolations",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_quadratic_regime(
    raw,
    strengths,
    diagnostic,
    *,
    output_index: int,
    input_index: int,
    path: Path,
) -> None:
    """Plot h<=8 data, low-range M2 fit, and full-range M2 reference."""
    normalized = raw / strengths[None, :, None, None] ** 2
    cosine, sine = signed_parts(normalized[:, :, output_index, input_index])
    plot_indices = diagnostic["plot_indices"]
    low_indices = diagnostic["low_indices"]
    critical = float(stats.t.ppf(0.975, raw.shape[0] - 1))
    dense_x = np.linspace(0.0, 64.0, 240)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), squeeze=False)
    for column, (part, values) in enumerate((("cos", cosine), ("sin", sine))):
        axis = axes[0, column]
        x = strengths[plot_indices] ** 2
        mean = values[:, plot_indices].mean(axis=0)
        half = (
            critical * values[:, plot_indices].std(axis=0, ddof=1)
            / np.sqrt(len(values))
        )
        noisy = strengths[plot_indices] == 2.0
        axis.errorbar(
            x[noisy], mean[noisy], yerr=half[noisy],
            color="#777777", marker="o", markerfacecolor="white",
            linewidth=0.9, capsize=2, label="h=2 noise reference",
        )
        axis.errorbar(
            x[~noisy], mean[~noisy], yerr=half[~noisy],
            color="#222222", marker="o", linewidth=0.9, capsize=2,
            label="h=4..8 block mean and 95% CI",
        )
        curves = {}
        for window, color, linestyle, label in (
            ("low_4_8", "#245b9e", "-", "M2 fit on h=4..8"),
            ("full_4_16", "#b55532", "--", "M2 fit on h=4..16"),
        ):
            fit = diagnostic["fits"][(
                output_index, input_index, part, window
            )]
            coefficients = fit["mean_coefficients"]
            curve = coefficients[0] + coefficients[1] * dense_x
            curves[window] = curve
            axis.plot(
                dense_x, curve, color=color, linestyle=linestyle,
                linewidth=1.5, label=label,
            )
            if window == "low_4_8":
                block_coefficients = fit["block_coefficients"]
                block_curves = (
                    block_coefficients[:, 0, None]
                    + block_coefficients[:, 1, None] * dense_x[None, :]
                )
                curve_half = (
                    critical * block_curves.std(axis=0, ddof=1)
                    / np.sqrt(len(block_curves))
                )
                axis.fill_between(
                    dense_x, curve - curve_half, curve + curve_half,
                    color=color, alpha=0.12, linewidth=0,
                )
        axis.axhline(0.0, color="#888888", linewidth=0.6)
        axis.set_xlim(0.0, 66.0)
        axis.set_xlabel(r"$h^2$")
        axis.set_title(
            f"Q({STATE_NAMES[output_index]},{STATE_NAMES[input_index]}"
            f"{STATE_NAMES[input_index]}) {part}"
        )
        if column == 0:
            axis.set_ylabel(r"even$(h)/h^2$")

        inset = axis.inset_axes((0.43, 0.52, 0.54, 0.43))
        low_x = strengths[low_indices] ** 2
        low_mean = values[:, low_indices].mean(axis=0)
        low_half = (
            critical * values[:, low_indices].std(axis=0, ddof=1)
            / np.sqrt(len(values))
        )
        inset.errorbar(
            low_x, low_mean, yerr=low_half,
            color="#222222", marker="o", markersize=3,
            linewidth=0.7, capsize=1.5,
        )
        inset.plot(dense_x, curves["low_4_8"], color="#245b9e", linewidth=1.0)
        inset.plot(
            dense_x, curves["full_4_16"], color="#b55532",
            linestyle="--", linewidth=1.0,
        )
        inset.axhline(0.0, color="#888888", linewidth=0.5)
        inset.set_xlim(14.0, 66.0)
        inset.set_title("h=4..8 zoom", fontsize=7)
        inset.tick_params(labelsize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=4, frameon=False,
    )
    figure.suptitle(
        "Candidate quadratic regime; no h>=10 observations shown",
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--additional-artifact", type=Path, action="append", default=None,
        help="completed disjoint-strength artifact to combine blockwise",
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    artifact = arguments.artifact.resolve()
    additional_artifacts = [
        value.resolve() for value in (
            arguments.additional_artifact
            if arguments.additional_artifact is not None
            else DEFAULT_ADDITIONAL_ARTIFACTS
        )
    ]
    artifacts = [artifact, *additional_artifacts]
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    if existing and not arguments.overwrite:
        raise ValueError(f"output directory is not clean: {output}")
    if not existing.issubset(OUTPUT_FILES):
        raise ValueError(f"output directory contains unexpected files: {sorted(existing)}")

    lineages = [artifact_lineage(value) for value in artifacts]
    data = combine_completed_contrasts([
        load_completed_contrasts(value) for value in artifacts
    ])
    strengths = data["strengths"]
    linear_allowed_pairs = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 2))
    second_allowed_pairs = ((2, 0), (2, 1), (2, 2))
    linear_comparison = joint_model_comparison(
        data["odd_fundamental"], strengths, base_order=1,
        pairs=linear_allowed_pairs,
    )
    second_comparison = joint_model_comparison(
        data["even_second"], strengths, base_order=2,
        pairs=second_allowed_pairs,
    )
    linear_degree = select_degree(linear_comparison)
    second_degree = select_degree(second_comparison)

    linear_entries, linear_fits = estimate_coefficients(
        data["odd_fundamental"], strengths,
        base_order=1, degree=linear_degree,
    )
    second_entries, second_fits = estimate_coefficients(
        data["even_second"], strengths,
        base_order=2, degree=second_degree,
    )
    parity = {"x": -1, "y": -1, "z": 1}
    for entry in linear_entries:
        entry["lorenz_parity_allowed"] = bool(
            parity[entry["output"]] * parity[entry["input"]] == 1
        )
    for entry in second_entries:
        entry["lorenz_parity_allowed"] = bool(entry["output"] == "z")
    chi1_entries, chi2_entries = susceptibility_conversions(
        linear_entries, second_entries
    )
    normalized_linear = data["odd_fundamental"] / strengths[None, :, None, None]
    normalized_second = data["even_second"] / strengths[None, :, None, None] ** 2
    first_amplitude = amplitude_dependence(normalized_linear, strengths)
    second_amplitude = amplitude_dependence(normalized_second, strengths)
    sensitivity = fit_range_sensitivity(
        data["even_second"], strengths,
        base_order=2, degree=second_degree,
    )
    if second_degree < MAX_DEGREE:
        sensitivity.extend(
            row for row in fit_range_sensitivity(
                data["even_second"], strengths,
                base_order=2, degree=second_degree + 1,
            )
            if row["strength_min"] == float(strengths[0])
            and row["strength_max"] == float(strengths[-1])
        )

    linear_robustness = intercept_robustness_records(
        data["odd_fundamental"], strengths,
        family="first_order_L", base_order=1,
        allowed_pairs=linear_allowed_pairs,
    )
    second_robustness = intercept_robustness_records(
        data["even_second"], strengths,
        family="second_order_Q", base_order=2,
        allowed_pairs=second_allowed_pairs,
    )
    robustness_rows = linear_robustness + second_robustness
    component_robustness, coefficient_robustness = (
        summarize_intercept_robustness(robustness_rows)
    )
    windows = robustness_windows(strengths)
    quadratic_regime = quadratic_regime_diagnostic(
        data["even_second"], strengths,
        allowed_pairs=second_allowed_pairs,
    )
    quadratic_regime_document = {
        "analysis": "single_axis_quadratic_regime_diagnostic_v1",
        "source_artifacts": [str(value) for value in artifacts],
        "omega": data["omega"],
        "block_count": len(data["block_ids"]),
        "plotted_strengths": [
            float(value) for value in strengths[quadratic_regime["plot_indices"]]
        ],
        "fit_model": "even(h)/h^2 = Q + C4*h^2",
        "low_fit_strengths": [
            float(value) for value in strengths[quadratic_regime["low_indices"]]
        ],
        "full_reference_strengths": [
            float(value) for value in strengths[quadratic_regime["full_indices"]]
        ],
        "h2_role": "plotted sampling-noise reference; excluded from both fits",
        "h10_and_above_role": (
            "excluded from plotted observations; included only in the dashed "
            "full-range fit reference"
        ),
        "uncertainty": (
            "pointwise 95% t intervals across crossed block-level GLS "
            "coefficient estimates"
        ),
        "fits": quadratic_regime["records"],
        "low_joint_model_comparison": quadratic_regime["low_model_comparison"],
        "full_joint_model_comparison": quadratic_regime["full_model_comparison"],
    }
    write_json_atomic(
        output / "quadratic_regime_diagnostic.json",
        quadratic_regime_document,
    )
    robustness_document = {
        "analysis": "single_axis_intercept_robustness_v2",
        "source_artifacts": [str(value) for value in artifacts],
        "source_manifest_identifiers": [
            lineage[0]["manifest_identifier"] for lineage in lineages
        ],
        "source_lineages": lineages,
        "combination_audit": data["combination_audit"],
        "omega": data["omega"],
        "block_count": len(data["block_ids"]),
        "strengths": strengths,
        "models": {
            "first_order_L": [
                {"degree": 2, "equation": "odd/h = L + C3*h^2", "raw_powers": [1, 3]},
                {"degree": 3, "equation": "odd/h = L + C3*h^2 + C5*h^4", "raw_powers": [1, 3, 5]},
            ],
            "second_order_Q": [
                {"degree": 2, "equation": "even/h^2 = Q + C4*h^2", "raw_powers": [2, 4]},
                {"degree": 3, "equation": "even/h^2 = Q + C4*h^2 + C6*h^4", "raw_powers": [2, 4, 6]},
            ],
        },
        "amplitude_windows": [
            {
                "label": window["label"],
                "included_strengths": [
                    float(value) for value in strengths[window["indices"]]
                ],
                "excluded_smallest": window["excluded_smallest"],
                "excluded_largest_count": window["excluded_largest_count"],
            }
            for window in windows
        ],
        "uncertainty": {
            "unit": "one initial-state block, crossed across included amplitudes",
            "confidence": CONFIDENCE,
            "interval": "pointwise t interval across block-level GLS intercept estimates",
            "covariance": "full within-window amplitude covariance estimated across blocks",
            "multiplicity_adjustment": None,
        },
        "diagnostic_classification": {
            "purpose": "compact cross-fit diagnostic, not a fit-selection rule",
            "stable": "all-fit CI overlap and both systematic shifts <=1.25 times the median sampling CI half-width",
            "borderline": "all-fit CI overlap and both systematic shifts <=2 times the median sampling CI half-width",
            "model_sensitive": "otherwise",
            "coefficient_rule": "worst classification of its signed cosine and sine components",
        },
        "fits": robustness_rows,
        "component_summary": component_robustness,
        "coefficient_summary": coefficient_robustness,
        "interpretation": (
            "No final intercept is selected here. Stability is assessed from the "
            "spread and overlap across every prespecified reasonable model/window."
        ),
        "limitations": [
            "Intervals are pointwise; the cross-fit spread is reported separately rather than folded into each CI.",
            "Amplitude covariance is estimated from the same finite set of blocks.",
            "Lorenz-parity-forbidden coefficients are constrained to zero and are not fit.",
            "No xy, xz, or yz cross-input coefficient is calculated.",
        ],
    }
    write_json_atomic(output / "intercept_stability.json", robustness_document)

    results = {
        "analysis": "single_axis_frequency_response_coefficients_v2",
        "source_artifacts": [str(value) for value in artifacts],
        "source_lineages": lineages,
        "combination_audit": data["combination_audit"],
        "omega": data["omega"],
        "block_count": len(data["block_ids"]),
        "block_ids": data["block_ids"],
        "strengths": strengths,
        "forcing_directions": list(STATE_NAMES),
        "harmonics": {"first_order": 1, "second_order": 2},
        "coefficient_convention": (
            "complex coefficient = signed cosine - i*signed sine; "
            "odd_n1(h)=h*L+h^3*C3+h^5*C5+...; "
            "even_n2(h)=h^2*Q+h^4*C4+..."
        ),
        "uncertainty": {
            "unit": "one initial-state block, kept crossed across every amplitude",
            "confidence": CONFIDENCE,
            "interval": "pointwise t interval across block-level GLS coefficient estimates",
            "multiplicity_adjustment": None,
        },
        "model_selection": {
            "rule": (
                "first polynomial degree with global generalized least-squares "
                "Hotelling-F lack-of-fit p>=0.05 over Lorenz-parity-allowed "
                "signed components, followed by any next term with nested "
                "improvement p<0.05"
            ),
            "covariance": "full crossed-amplitude covariance estimated across blocks",
            "linear": {
                "selected_degree": linear_degree,
                "selected_raw_powers": linear_comparison[linear_degree - 1]["raw_powers"],
                "comparison": linear_comparison,
            },
            "second_order": {
                "selected_degree": second_degree,
                "selected_raw_powers": second_comparison[second_degree - 1]["raw_powers"],
                "comparison": second_comparison,
            },
        },
        "first_order_L": linear_entries,
        "first_order_chi1_equals_2iL": chi1_entries,
        "first_order_amplitude_dependence": first_amplitude,
        "second_order_Q_at_2omega": second_entries,
        "second_harmonic_chi2_equals_minus4Q": chi2_entries,
        "second_order_amplitude_dependence": second_amplitude,
        "second_order_fit_range_sensitivity": sensitivity,
        "intercept_robustness": {
            "artifact": "intercept_stability.json",
            "coefficient_summary": coefficient_robustness,
            "interpretation": (
                "Requested order/window fits are retained without choosing a final "
                "coefficient from fit statistics."
            ),
        },
        "quadratic_regime_diagnostic": {
            "artifact": "quadratic_regime_diagnostic.json",
            "fit_table": "quadratic_regime_fits.csv",
            "low_fit_strengths": [4.0, 5.0, 6.0, 7.0, 8.0],
            "full_reference_strengths": [
                4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0,
            ],
        },
        "limitations": [
            "Intervals are pointwise and conditional on the selected polynomial order.",
            "The next even term is reported as model sensitivity rather than folded into the pointwise CI.",
            "Lorenz-parity-forbidden entries are reported as unconstrained diagnostics; the symmetry-constrained tensor sets them exactly to zero.",
            "No xy, xz, or yz cross-input coefficient is calculated.",
            "Completed artifacts contribute disjoint amplitudes on the same block IDs; base/extension provenance is not counted as additional replication.",
        ],
    }
    write_json_atomic(output / "coefficient_estimates.json", results)

    coefficient_fields = (
        "output", "input", "lorenz_parity_allowed",
        "cos_estimate", "cos_ci95_low", "cos_ci95_high",
        "sin_estimate", "sin_ci95_low", "sin_ci95_high",
    )
    _write_csv(
        output / "first_order_matrix.csv",
        coefficient_rows(chi1_entries), coefficient_fields,
    )
    _write_csv(
        output / "second_order_diagonal.csv",
        coefficient_rows(second_entries), coefficient_fields,
    )
    _write_csv(
        output / "first_order_amplitude_dependence.csv",
        first_amplitude,
        ("output", "input", "strength", "part", "estimate", "ci95_low", "ci95_high"),
    )
    _write_csv(
        output / "second_order_amplitude_dependence.csv",
        second_amplitude,
        ("output", "input", "strength", "part", "estimate", "ci95_low", "ci95_high"),
    )
    _write_csv(
        output / "fit_range_sensitivity.csv",
        sensitivity,
        (
            "output", "input", "part", "degree", "strength_min", "strength_max",
            "strength_count", "estimate", "ci95_low", "ci95_high",
        ),
    )
    stability_csv_rows = [
        {
            **row,
            "raw_powers": json.dumps(row["raw_powers"]),
            "included_strengths": json.dumps(row["included_strengths"]),
        }
        for row in robustness_rows
    ]
    _write_csv(
        output / "intercept_stability.csv",
        stability_csv_rows,
        (
            "family", "output", "input", "part", "degree", "raw_powers",
            "window", "included_strengths", "strength_min", "strength_max",
            "excluded_smallest", "excluded_largest_count", "estimate",
            "standard_error", "ci95_low", "ci95_high",
        ),
    )
    _write_csv(
        output / "intercept_stability_summary.csv",
        component_robustness,
        (
            "family", "output", "input", "part", "classification",
            "dominant_source", "polynomial_order_shift_max",
            "amplitude_window_span_max", "sampling_ci95_half_width_median",
            "sampling_ci95_half_width_min", "sampling_ci95_half_width_max",
            "order_shift_to_sampling_half_width",
            "window_span_to_sampling_half_width", "intercept_min",
            "intercept_max", "all_fit_ci95_common_overlap",
        ),
    )
    quadratic_regime_csv = [
        {
            **row,
            "included_strengths": json.dumps(row["included_strengths"]),
        }
        for row in quadratic_regime["records"]
    ]
    _write_csv(
        output / "quadratic_regime_fits.csv",
        quadratic_regime_csv,
        (
            "output", "input", "part", "window", "included_strengths",
            "Q_estimate", "Q_standard_error", "Q_ci95_low", "Q_ci95_high",
            "C4_estimate", "C4_standard_error", "C4_ci95_low", "C4_ci95_high",
        ),
    )

    apply_style()
    plot_amplitude_dependence(
        normalized_linear, strengths, linear_fits,
        base_order=1, degree=linear_degree,
        title="First-order coefficient L(h)=odd n=1 / h",
        ylabel="signed L(h)",
        path=output / "first_order_amplitude_dependence.png",
    )
    plot_amplitude_dependence(
        normalized_second, strengths, second_fits,
        base_order=2, degree=second_degree,
        title="Second-order coefficient Q(h)=even n=2 / h^2",
        ylabel="signed Q(h)",
        path=output / "second_order_amplitude_dependence.png",
    )
    plot_second_order_sensitivity(
        sensitivity, second_entries,
        output / "second_order_fit_sensitivity.png",
    )
    plot_intercept_robustness(
        robustness_rows, coefficient_robustness,
        family="first_order_L", allowed_pairs=linear_allowed_pairs,
        title="First-order h->0 intercepts across model order and amplitude window",
        path=output / "first_order_intercept_stability.png",
    )
    plot_intercept_robustness(
        robustness_rows, coefficient_robustness,
        family="second_order_Q", allowed_pairs=second_allowed_pairs,
        title="Second-order h->0 intercepts across model order and amplitude window",
        path=output / "second_order_intercept_stability.png",
    )
    plot_quadratic_extrapolation_curves(
        data["even_second"], strengths,
        allowed_pairs=second_allowed_pairs,
        path=output / "quadratic_extrapolation_curves.png",
    )
    for output_index, input_index in second_allowed_pairs:
        plot_quadratic_regime(
            data["even_second"], strengths, quadratic_regime,
            output_index=output_index, input_index=input_index,
            path=(
                output
                / f"quadratic_regime_Q_z_{STATE_NAMES[input_index]}"
                f"{STATE_NAMES[input_index]}.png"
            ),
        )

    manifest_files = sorted(OUTPUT_FILES.difference({"analysis_manifest.json"}))
    manifest = {
        "schema_version": 2,
        "classification": "single_axis_frequency_response_coefficients_v2",
        "source_manifest_identifiers": [
            lineage[0]["manifest_identifier"] for lineage in lineages
        ],
        "analysis_code_identifier": (
            f"sha256:{file_sha256(Path(__file__).resolve())}"
        ),
        "files": {
            name: f"sha256:{file_sha256(output / name)}" for name in manifest_files
        },
        "scientific_scope": (
            "omega=5.938; x/y/z single-axis inputs; first-order matrix and "
            "diagonal-input second-harmonic quadratic coefficients only"
        ),
    }
    write_json_atomic(output / "analysis_manifest.json", manifest)
    print(json.dumps({
        "output_directory": str(output),
        "omega": data["omega"],
        "block_count": len(data["block_ids"]),
        "strengths": strengths.tolist(),
        "linear_raw_powers": linear_comparison[linear_degree - 1]["raw_powers"],
        "second_order_raw_powers": second_comparison[second_degree - 1]["raw_powers"],
        "robustness_fit_count": len(robustness_rows),
        "robustness_classification": coefficient_robustness,
    }, indent=2))


if __name__ == "__main__":
    main()
