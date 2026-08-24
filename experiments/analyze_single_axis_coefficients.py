#!/usr/bin/env python3
"""Estimate single-axis response coefficients from a completed probe artifact.

No trajectories are integrated.  The analysis consumes the crossed block-level
known-frequency coefficients already stored by the omega=5.938 high-order
probe, uses the x/y/z forcing directions only, and retains the common block as
the uncertainty unit across all amplitudes.
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
from scipy.linalg import cho_factor, cho_solve


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-matplotlib")
import matplotlib.pyplot as plt

from lorenz.artifacts import file_sha256, verify_file_identifiers, write_json_atomic
from lorenz.response import paired_order_contrasts
from lorenz.response_plots import apply_style
from lorenz.retention import paired_condition_indices


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "outputs/production/high_order_probe_omega_5p938_h10_h12_h14_h16_extension_v1"
    / "20260822T061155_b1e645f07e98"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/analysis/single_axis_coefficients_omega_5p938_v1"
)
STATE_NAMES = ("x", "y", "z")
CONFIDENCE = 0.95
LACK_OF_FIT_ALPHA = 0.05
MAX_DEGREE = 4
OUTPUT_FILES = {
    "coefficient_estimates.json",
    "first_order_matrix.csv",
    "second_order_diagonal.csv",
    "second_order_amplitude_dependence.csv",
    "fit_range_sensitivity.csv",
    "first_order_amplitude_dependence.png",
    "second_order_amplitude_dependence.png",
    "second_order_fit_sensitivity.png",
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
        "config": config,
        "omega": omega,
        "strengths": strengths,
        "block_ids": block_ids,
        "odd_fundamental": odd_fundamental,
        "even_second": even_second,
    }


def signed_parts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine and sine amplitudes for coefficient = cos - i*sin."""
    values = np.asarray(values)
    return values.real, -values.imag


def _scaled_design(strengths, base_order: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    scale = float(np.max(strengths))
    powers = base_order + 2 * np.arange(degree)
    design = np.column_stack([(strengths / scale) ** power for power in powers])
    return design, scale**powers


def scalar_gls_fit(
    block_values: np.ndarray,
    strengths: np.ndarray,
    *,
    base_order: int,
    degree: int,
) -> dict:
    """Fit one signed raw contrast using its crossed-strength block covariance."""
    block_values = np.asarray(block_values, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if block_values.shape != (block_values.shape[0], len(strengths)):
        raise ValueError("block values must have axes block,strength")
    covariance = np.cov(block_values, rowvar=False, ddof=1)
    design, physical_scales = _scaled_design(strengths, base_order, degree)
    factor = cho_factor(covariance, lower=True, check_finite=False)
    inverse_design = cho_solve(factor, design, check_finite=False)
    normal = design.T @ inverse_design
    coefficient_operator = np.linalg.solve(
        normal,
        design.T @ cho_solve(
            factor, np.eye(len(strengths)), check_finite=False
        ),
    )
    block_coefficients = block_values @ coefficient_operator.T
    block_coefficients = block_coefficients / physical_scales
    mean_coefficients = block_coefficients.mean(axis=0)
    fitted_mean = np.column_stack(
        [strengths ** (base_order + 2 * index) for index in range(degree)]
    ) @ mean_coefficients
    return {
        "block_coefficients": block_coefficients,
        "mean_coefficients": mean_coefficients,
        "fitted_mean": fitted_mean,
        "intercept_weights_on_raw_contrast": (
            coefficient_operator[0] / physical_scales[0]
        ),
    }


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
    block_count, strength_count, feature_count = features.shape
    mean = features.mean(axis=0).reshape(-1)
    covariance_of_mean = np.cov(
        features.reshape(block_count, -1), rowvar=False, ddof=1
    ) / block_count
    factor = cho_factor(covariance_of_mean, lower=True, check_finite=False)
    inverse_mean = cho_solve(factor, mean, check_finite=False)
    comparisons = []
    for degree in range(1, MAX_DEGREE + 1):
        design, _ = _scaled_design(strengths, base_order, degree)
        joint_design = np.kron(design, np.eye(feature_count))
        inverse_design = cho_solve(
            factor, joint_design, check_finite=False
        )
        coefficients = np.linalg.solve(
            joint_design.T @ inverse_design,
            joint_design.T @ inverse_mean,
        )
        residual = mean - joint_design @ coefficients
        statistic = float(
            residual @ cho_solve(factor, residual, check_finite=False)
        )
        degrees_of_freedom = int(
            strength_count * feature_count - degree * feature_count
        )
        hotelling_f = (
            (block_count - degrees_of_freedom)
            / (degrees_of_freedom * (block_count - 1))
            * statistic
        )
        comparisons.append({
            "degree": degree,
            "raw_powers": [base_order + 2 * index for index in range(degree)],
            "statistic": statistic,
            "degrees_of_freedom": degrees_of_freedom,
            "hotelling_f": hotelling_f,
            "f_denominator_degrees_of_freedom": block_count - degrees_of_freedom,
            "lack_of_fit_p": float(stats.f.sf(
                hotelling_f,
                degrees_of_freedom,
                block_count - degrees_of_freedom,
            )),
            "aic": statistic + 2 * degree * feature_count,
        })
    for index in range(1, len(comparisons)):
        simpler = comparisons[index - 1]
        current = comparisons[index]
        difference = simpler["statistic"] - current["statistic"]
        added = feature_count
        improvement_f = (
            (block_count - added) / (added * (block_count - 1)) * difference
        )
        current["nested_improvement_f"] = improvement_f
        current["nested_improvement_p"] = float(
            stats.f.sf(improvement_f, added, block_count - added)
        )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    artifact = arguments.artifact.resolve()
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    if existing and not arguments.overwrite:
        raise ValueError(f"output directory is not clean: {output}")
    if not existing.issubset(OUTPUT_FILES):
        raise ValueError(f"output directory contains unexpected files: {sorted(existing)}")

    lineage = artifact_lineage(artifact)
    data = load_completed_contrasts(artifact)
    strengths = data["strengths"]
    linear_comparison = joint_model_comparison(
        data["odd_fundamental"], strengths, base_order=1,
        pairs=((0, 0), (1, 0), (0, 1), (1, 1), (2, 2)),
    )
    second_comparison = joint_model_comparison(
        data["even_second"], strengths, base_order=2,
        pairs=((2, 0), (2, 1), (2, 2)),
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

    results = {
        "analysis": "single_axis_frequency_response_coefficients_v1",
        "source_artifact": str(artifact),
        "source_lineage": lineage,
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
        "second_order_Q_at_2omega": second_entries,
        "second_harmonic_chi2_equals_minus4Q": chi2_entries,
        "second_order_amplitude_dependence": second_amplitude,
        "second_order_fit_range_sensitivity": sensitivity,
        "limitations": [
            "Intervals are pointwise and conditional on the selected polynomial order.",
            "The next even term is reported as model sensitivity rather than folded into the pointwise CI.",
            "Lorenz-parity-forbidden entries are reported as unconstrained diagnostics; the symmetry-constrained tensor sets them exactly to zero.",
            "No xy, xz, or yz cross-input coefficient is calculated.",
            "The calculation uses the self-contained final artifact once; base/extension artifacts share block IDs and are not counted again.",
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

    manifest_files = sorted(OUTPUT_FILES.difference({"analysis_manifest.json"}))
    manifest = {
        "schema_version": 1,
        "classification": "single_axis_frequency_response_coefficients_v1",
        "source_manifest_identifier": lineage[0]["manifest_identifier"],
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
    }, indent=2))


if __name__ == "__main__":
    main()
