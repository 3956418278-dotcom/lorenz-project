#!/usr/bin/env python3
"""Analyze the completed B=512 h={4,6,8} single-axis refinement.

No trajectories are integrated.  The new artifact is the primary crossed
512-block dataset.  Existing B=256 artifacts are used only as nested-block and
previous-design references; overlapping blocks are never stacked as new
replications.
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

from analyze_single_axis_coefficients import (
    STATE_NAMES,
    _combination_settings,
    artifact_lineage,
    combine_completed_contrasts,
    interval,
    joint_model_comparison,
    load_completed_contrasts,
    quadratic_regime_diagnostic,
    scalar_gls_fit,
    signed_parts,
)
from lorenz.artifacts import file_sha256, write_json_atomic
from lorenz.response_plots import apply_style


DEFAULT_REFINEMENT = (
    REPO_ROOT
    / "outputs/production/single_axis_coefficients_omega_5p938_h4_h6_h8_b512_v1"
    / "20260824T125050_d90049066676"
)
DEFAULT_PRIMARY_B256 = (
    REPO_ROOT
    / "outputs/production/high_order_probe_omega_5p938_h10_h12_h14_h16_extension_v1"
    / "20260822T061155_b1e645f07e98"
)
DEFAULT_GAP_B256 = (
    REPO_ROOT
    / "outputs/production/single_axis_coefficients_omega_5p938_h5_h6_h7_v1"
    / "20260824T074945_7878c8a09d92"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/analysis/single_axis_quadratic_b512_refinement_v1"
)
ALLOWED_PAIRS = ((2, 0), (2, 1), (2, 2))
OUTPUT_FILES = {
    "quadratic_b512_refinement.json",
    "quadratic_b512_comparison.csv",
    "quadratic_b512_amplitudes.csv",
    "quadratic_b512_Q_z_xx.png",
    "quadratic_b512_Q_z_yy.png",
    "quadratic_b512_Q_z_zz.png",
    "analysis_manifest.json",
}


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _compatible_settings(config: dict) -> dict:
    settings = _combination_settings(config)
    settings.pop("block_count")
    return settings


def _nested_half_delta(values: np.ndarray) -> dict:
    """Full-mean minus first-half mean, using two independent 256-block halves."""
    values = np.asarray(values, dtype=float)
    if len(values) != 512:
        raise ValueError("nested comparison requires exactly 512 block values")
    first, second = values[:256], values[256:]
    estimate = 0.5 * float(second.mean() - first.mean())
    first_variance = float(first.var(ddof=1) / len(first))
    second_variance = float(second.var(ddof=1) / len(second))
    standard_error = 0.5 * np.sqrt(first_variance + second_variance)
    denominator = (
        first_variance**2 / (len(first) - 1)
        + second_variance**2 / (len(second) - 1)
    )
    degrees_of_freedom = (
        (first_variance + second_variance) ** 2 / denominator
        if denominator > 0 else float("inf")
    )
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    statistic = estimate / standard_error if standard_error else 0.0
    return {
        "estimate": estimate,
        "standard_error": float(standard_error),
        "degrees_of_freedom": float(degrees_of_freedom),
        "ci95": [
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ],
        "p_two_sided": float(2 * stats.t.sf(abs(statistic), degrees_of_freedom)),
    }


def _point_rows(raw, strengths, *, dataset: str) -> list[dict]:
    rows = []
    normalized = raw / strengths[None, :, None, None] ** 2
    for output_index, input_index in ALLOWED_PAIRS:
        cosine, sine = signed_parts(normalized[:, :, output_index, input_index])
        for part, values in (("cos", cosine), ("sin", sine)):
            for index, strength in enumerate(strengths):
                estimate = interval(values[:, index])
                rows.append({
                    "dataset": dataset,
                    "block_count": len(values),
                    "output": STATE_NAMES[output_index],
                    "input": STATE_NAMES[input_index],
                    "part": part,
                    "strength": float(strength),
                    "strength_squared": float(strength**2),
                    "estimate": estimate["estimate"],
                    "standard_error": estimate["standard_error"],
                    "ci95_low": estimate["ci95"][0],
                    "ci95_high": estimate["ci95"][1],
                })
    return rows


def analyze(refinement: dict, previous: dict) -> dict:
    strengths = np.asarray(refinement["strengths"], dtype=float)
    if not np.array_equal(strengths, [4.0, 6.0, 8.0]):
        raise ValueError("refinement strengths must be exactly h=4,6,8")
    if len(refinement["block_ids"]) != 512 or len(previous["block_ids"]) != 256:
        raise ValueError("expected nested B=512 and B=256 datasets")
    if not np.array_equal(refinement["block_ids"][:256], previous["block_ids"]):
        raise ValueError("the previous block IDs are not the B=512 prefix")

    previous_indices = np.asarray([
        int(np.flatnonzero(np.isclose(previous["strengths"], value))[0])
        for value in strengths
    ])
    previous_even = previous["even_second"][:, previous_indices]
    previous_odd = previous["odd_fundamental"][:, previous_indices]
    overlap_even_difference = float(np.max(np.abs(
        refinement["even_second"][:256] - previous_even
    )))
    overlap_odd_difference = float(np.max(np.abs(
        refinement["odd_fundamental"][:256] - previous_odd
    )))
    if overlap_even_difference != 0.0 or overlap_odd_difference != 0.0:
        raise ValueError("overlapping B=256 contrasts do not reproduce exactly")

    previous_five = quadratic_regime_diagnostic(
        previous["even_second"], previous["strengths"],
        allowed_pairs=ALLOWED_PAIRS,
    )
    previous_five_by_key = {
        (row["output"], row["input"], row["part"]): row
        for row in previous_five["records"]
        if row["window"] == "low_4_8"
    }
    expected_half_width_ratio = float(
        stats.t.ppf(0.975, 511) / stats.t.ppf(0.975, 255) / np.sqrt(2)
    )
    records = []
    fits = {}
    for output_index, input_index in ALLOWED_PAIRS:
        new_cosine, new_sine = signed_parts(
            refinement["even_second"][:, :, output_index, input_index]
        )
        old_cosine, old_sine = signed_parts(
            previous_even[:, :, output_index, input_index]
        )
        for part, new_values, old_values in (
            ("cos", new_cosine, old_cosine),
            ("sin", new_sine, old_sine),
        ):
            new_fit = scalar_gls_fit(
                new_values, strengths, base_order=2, degree=2
            )
            old_fit = scalar_gls_fit(
                old_values, strengths, base_order=2, degree=2
            )
            fits[(output_index, input_index, part, "B512")] = new_fit
            fits[(output_index, input_index, part, "B256_matched")] = old_fit
            new_q = interval(new_fit["block_coefficients"][:, 0])
            new_c4 = interval(new_fit["block_coefficients"][:, 1])
            old_q = interval(old_fit["block_coefficients"][:, 0])
            old_c4 = interval(old_fit["block_coefficients"][:, 1])
            q_delta = _nested_half_delta(new_fit["block_coefficients"][:, 0])
            c4_delta = _nested_half_delta(new_fit["block_coefficients"][:, 1])
            previous_five_row = previous_five_by_key[
                (STATE_NAMES[output_index], STATE_NAMES[input_index], part)
            ]
            q_half_width_ratio = (
                (new_q["ci95"][1] - new_q["ci95"][0])
                / (old_q["ci95"][1] - old_q["ci95"][0])
            )
            c4_half_width_ratio = (
                (new_c4["ci95"][1] - new_c4["ci95"][0])
                / (old_c4["ci95"][1] - old_c4["ci95"][0])
            )
            drift = 48.0 * new_c4["estimate"]
            records.append({
                "output": STATE_NAMES[output_index],
                "input": STATE_NAMES[input_index],
                "part": part,
                "B512_Q_estimate": new_q["estimate"],
                "B512_Q_standard_error": new_q["standard_error"],
                "B512_Q_ci95_low": new_q["ci95"][0],
                "B512_Q_ci95_high": new_q["ci95"][1],
                "B512_C4_estimate": new_c4["estimate"],
                "B512_C4_standard_error": new_c4["standard_error"],
                "B512_C4_ci95_low": new_c4["ci95"][0],
                "B512_C4_ci95_high": new_c4["ci95"][1],
                "B512_C4_excludes_zero": bool(
                    new_c4["ci95"][0] > 0 or new_c4["ci95"][1] < 0
                ),
                "B512_predicted_Qh_drift_h4_to_h8": drift,
                "B512_abs_drift_to_abs_Q": (
                    abs(drift / new_q["estimate"])
                    if new_q["estimate"] else float("inf")
                ),
                "B256_matched_Q_estimate": old_q["estimate"],
                "B256_matched_Q_ci95_low": old_q["ci95"][0],
                "B256_matched_Q_ci95_high": old_q["ci95"][1],
                "B256_matched_C4_estimate": old_c4["estimate"],
                "B256_matched_C4_ci95_low": old_c4["ci95"][0],
                "B256_matched_C4_ci95_high": old_c4["ci95"][1],
                "Q_ci95_half_width_ratio_B512_to_B256": q_half_width_ratio,
                "C4_ci95_half_width_ratio_B512_to_B256": c4_half_width_ratio,
                "expected_ci95_half_width_ratio": expected_half_width_ratio,
                "nested_Q_center_shift": q_delta["estimate"],
                "nested_Q_center_shift_ci95_low": q_delta["ci95"][0],
                "nested_Q_center_shift_ci95_high": q_delta["ci95"][1],
                "nested_Q_center_shift_p": q_delta["p_two_sided"],
                "nested_C4_center_shift": c4_delta["estimate"],
                "nested_C4_center_shift_ci95_low": c4_delta["ci95"][0],
                "nested_C4_center_shift_ci95_high": c4_delta["ci95"][1],
                "nested_C4_center_shift_p": c4_delta["p_two_sided"],
                "B256_five_amplitude_Q_estimate": previous_five_row["Q_estimate"],
                "B256_five_amplitude_Q_ci95_low": previous_five_row["Q_ci95_low"],
                "B256_five_amplitude_Q_ci95_high": previous_five_row["Q_ci95_high"],
                "B256_five_amplitude_C4_estimate": previous_five_row["C4_estimate"],
            })
    return {
        "records": records,
        "fits": fits,
        "point_rows": (
            _point_rows(refinement["even_second"], strengths, dataset="B512")
            + _point_rows(previous_even, strengths, dataset="B256_matched")
        ),
        "joint_model_comparison": joint_model_comparison(
            refinement["even_second"], strengths,
            base_order=2, pairs=ALLOWED_PAIRS,
        ),
        "expected_half_width_ratio": expected_half_width_ratio,
        "overlap_even_max_abs_difference": overlap_even_difference,
        "overlap_odd_max_abs_difference": overlap_odd_difference,
    }


def plot_component(analysis: dict, *, input_index: int, path: Path) -> None:
    points = analysis["point_rows"]
    records = analysis["records"]
    dense_x = np.linspace(0.0, 64.0, 240)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), squeeze=False)
    for column, part in enumerate(("cos", "sin")):
        axis = axes[0, column]
        for dataset, color, marker, face, label in (
            ("B256_matched", "#777777", "s", "white", "B=256 points"),
            ("B512", "#222222", "o", "#222222", "B=512 points"),
        ):
            selected = [
                row for row in points
                if row["dataset"] == dataset and row["input"] == STATE_NAMES[input_index]
                and row["part"] == part
            ]
            x = np.asarray([row["strength_squared"] for row in selected])
            y = np.asarray([row["estimate"] for row in selected])
            low = np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected])
            axis.errorbar(
                x, y, yerr=np.vstack((y - low, high - y)),
                color=color, marker=marker, markerfacecolor=face,
                linewidth=0.9, capsize=2, label=label,
            )
        for dataset, color, linestyle, label in (
            ("B256_matched", "#777777", "--", "B=256 M2 fit"),
            ("B512", "#245b9e", "-", "B=512 M2 fit"),
        ):
            fit = analysis["fits"][(2, input_index, part, dataset)]
            coefficients = fit["mean_coefficients"]
            curve = coefficients[0] + coefficients[1] * dense_x
            axis.plot(
                dense_x, curve, color=color, linestyle=linestyle,
                linewidth=1.4, label=label,
            )
            if dataset == "B512":
                block = fit["block_coefficients"]
                block_curve = block[:, 0, None] + block[:, 1, None] * dense_x
                critical = float(stats.t.ppf(0.975, len(block) - 1))
                half = critical * block_curve.std(axis=0, ddof=1) / np.sqrt(len(block))
                axis.fill_between(
                    dense_x, curve - half, curve + half,
                    color=color, alpha=0.12, linewidth=0,
                )
        axis.axhline(0.0, color="#888888", linewidth=0.6)
        axis.set_xlim(0.0, 66.0)
        axis.set_xlabel(r"$h^2$")
        axis.set_title(f"Q(z,{STATE_NAMES[input_index]}{STATE_NAMES[input_index]}) {part}")
        if column == 0:
            axis.set_ylabel(r"even$(h)/h^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("B=512 quadratic-regime refinement at h=4,6,8", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement-artifact", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--primary-b256-artifact", type=Path, default=DEFAULT_PRIMARY_B256)
    parser.add_argument("--gap-b256-artifact", type=Path, default=DEFAULT_GAP_B256)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    if existing and not arguments.overwrite:
        raise ValueError(f"output directory is not clean: {output}")
    if not existing.issubset(OUTPUT_FILES):
        raise ValueError(f"unexpected output files: {sorted(existing)}")

    paths = [
        arguments.refinement_artifact.resolve(),
        arguments.primary_b256_artifact.resolve(),
        arguments.gap_b256_artifact.resolve(),
    ]
    lineages = [artifact_lineage(path) for path in paths]
    refinement_dataset = load_completed_contrasts(paths[0])
    primary_dataset = load_completed_contrasts(paths[1])
    gap_dataset = load_completed_contrasts(paths[2])
    if _compatible_settings(refinement_dataset["config"]) != _compatible_settings(
        primary_dataset["config"]
    ) or _compatible_settings(refinement_dataset["config"]) != _compatible_settings(
        gap_dataset["config"]
    ):
        raise ValueError("refinement and B=256 numerical settings differ")
    directions = np.asarray(
        refinement_dataset["config"]["protocol"]["directions"], dtype=float
    )
    if not np.array_equal(directions, np.eye(3)):
        raise ValueError("refinement directions are not exactly x,y,z")
    previous = combine_completed_contrasts([primary_dataset, gap_dataset])
    if not np.array_equal(
        refinement_dataset["unforced"][:256], primary_dataset["unforced"]
    ):
        raise ValueError("refinement first-half unforced coefficients differ")

    analysis = analyze(refinement_dataset, previous)
    document = {
        "analysis": "single_axis_quadratic_b512_refinement_v1",
        "omega": refinement_dataset["omega"],
        "strengths": refinement_dataset["strengths"],
        "block_count": len(refinement_dataset["block_ids"]),
        "source_artifacts": [str(path) for path in paths],
        "source_manifest_identifiers": [
            lineage[0]["manifest_identifier"] for lineage in lineages
        ],
        "source_lineages": lineages,
        "coefficient_convention": "complex coefficient = signed cosine - i*signed sine",
        "fit_model": "even(h)/h^2 = Q + C4*h^2",
        "uncertainty": (
            "pointwise 95% t intervals across crossed block-level GLS "
            "coefficient estimates"
        ),
        "nested_comparison": (
            "B=256 IDs are the ordered first half of B=512; center-shift "
            "intervals use common B512 GLS weights on the two disjoint halves"
        ),
        "overlap_validation": {
            "block_ids_exact_prefix": True,
            "unforced_exact": True,
            "even_contrast_max_abs_difference": analysis[
                "overlap_even_max_abs_difference"
            ],
            "odd_contrast_max_abs_difference": analysis[
                "overlap_odd_max_abs_difference"
            ],
        },
        "expected_ci95_half_width_ratio": analysis["expected_half_width_ratio"],
        "joint_model_comparison": analysis["joint_model_comparison"],
        "coefficients": analysis["records"],
        "limitations": [
            "The B=512 primary fit uses h=4,6,8 only; h=5,7 remain B=256 and are not mixed into a ragged GLS fit.",
            "A nonzero C4 is a quartic finite-amplitude correction, not by itself evidence that C6 or higher terms dominate.",
            "No xy, xz, or yz cross-input coefficient is calculated.",
        ],
    }
    write_json_atomic(output / "quadratic_b512_refinement.json", document)
    _write_csv(
        output / "quadratic_b512_comparison.csv", analysis["records"],
        tuple(analysis["records"][0].keys()),
    )
    _write_csv(
        output / "quadratic_b512_amplitudes.csv", analysis["point_rows"],
        tuple(analysis["point_rows"][0].keys()),
    )
    apply_style()
    for _, input_index in ALLOWED_PAIRS:
        plot_component(
            analysis, input_index=input_index,
            path=output / f"quadratic_b512_Q_z_{STATE_NAMES[input_index]}{STATE_NAMES[input_index]}.png",
        )

    manifest_files = sorted(OUTPUT_FILES.difference({"analysis_manifest.json"}))
    manifest = {
        "schema_version": 1,
        "classification": "single_axis_quadratic_b512_refinement_v1",
        "source_manifest_identifiers": document["source_manifest_identifiers"],
        "analysis_code_identifier": f"sha256:{file_sha256(Path(__file__).resolve())}",
        "files": {
            name: f"sha256:{file_sha256(output / name)}" for name in manifest_files
        },
        "scientific_scope": (
            "omega=5.938; h=4,6,8; x/y/z inputs; allowed diagonal-input "
            "second-harmonic coefficients only"
        ),
    }
    write_json_atomic(output / "analysis_manifest.json", manifest)
    print(json.dumps({
        "output_directory": str(output),
        "block_count": len(refinement_dataset["block_ids"]),
        "strengths": refinement_dataset["strengths"].tolist(),
        "overlap_validation": document["overlap_validation"],
        "expected_ci95_half_width_ratio": analysis["expected_half_width_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
