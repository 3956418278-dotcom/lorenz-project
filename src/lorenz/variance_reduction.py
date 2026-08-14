"""Held-out variance-reduction study for monochromatic response estimates.

The second-harmonic alternative is fixed before looking at held-out blocks:
the unforced nonzero Fourier coefficient has population expectation zero, so
it is retained as a bias diagnostic but is not subtracted from the estimator.
No control-variate coefficient is trained on the evaluation blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

from .frequency_reconnaissance import (
    FrequencyReconData,
    _file_sha256,
    _frequency_key,
    _json_ready,
    _single_frequency_config,
    _validate_config,
    _write_json_atomic,
)
from .strength_bootstrap import _bootstrap_max_statistic, _magnitude_interval
from .strength_identifiability import (
    STATE_NAMES,
    StrengthStudyData,
    fit_block_power_series,
    generate_strength_study,
)


SECOND_NULLS = np.array([True, True, False])


@dataclass(frozen=True)
class VarianceFeatureGroup:
    estimator: str
    section: str
    observable: int
    real_index: int
    imaginary_index: int
    strength: float | None = None
    prefix_upper_strength: float | None = None
    contribution: str | None = None


def generate_heldout_studies(config: dict, repo_root: Path) -> FrequencyReconData:
    """Integrate the frozen grid on block IDs disjoint from prior evidence."""
    checked = _validate_config(config)
    if (
        int(config["block_count"]) != config["block_count"]
        or int(config["observation_rule"]["minimum_cycles"])
        != config["observation_rule"]["minimum_cycles"]
        or not np.isfinite(checked["strengths"]).all()
        or np.any(checked["strengths"] <= 0)
    ):
        raise ValueError("held-out counts and strengths must be finite and positive")
    excluded = config["holdout_contract"]["excluded_block_id_ranges"]
    initial = config["initial_ensemble"]
    block_ids = tuple(
        range(
            int(initial["block_id_start"]),
            int(initial["block_id_start"]) + checked["block_count"],
        )
    )
    for lower, upper in excluded:
        if any(int(lower) <= block_id <= int(upper) for block_id in block_ids):
            raise ValueError("held-out block IDs overlap declared prior evidence")
    prior = repo_root / config["holdout_contract"]["reconnaissance_artifact"]
    expected_hash = config["holdout_contract"]["reconnaissance_manifest_sha256"]
    if _file_sha256(prior / "manifest.json") != expected_hash:
        raise ValueError("reconnaissance manifest hash changed after design freeze")
    prior_manifest = json.loads((prior / "manifest.json").read_text(encoding="utf-8"))
    for name, recorded in prior_manifest["files"].items():
        if f"sha256:{_file_sha256(prior / name)}" != recorded:
            raise ValueError(f"reconnaissance artifact hash mismatch for {name}")
    with np.load(prior / "raw_frequency_summaries.npz", allow_pickle=False) as raw:
        prior_block_ids = set(int(value) for value in raw["block_ids"])
    overlap = prior_block_ids.intersection(block_ids)
    if overlap:
        raise ValueError(f"held-out block IDs overlap reconnaissance: {sorted(overlap)}")

    studies = {}
    runtimes = {}
    sources = {}
    reference = None
    for omega in checked["frequencies"]:
        start = time.perf_counter()
        study = generate_strength_study(
            _single_frequency_config(config, omega, checked["cycles"][omega])
        )
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = study
        else:
            if study.block_ids != reference.block_ids:
                raise RuntimeError("held-out frequency block order changed")
            np.testing.assert_array_equal(study.raw_proposals, reference.raw_proposals)
            np.testing.assert_array_equal(study.initial_states, reference.initial_states)
            if study.child_spawn_keys != reference.child_spawn_keys:
                raise RuntimeError("held-out frequency spawn keys changed")
        studies[omega] = study
        runtimes[omega] = elapsed
        sources[omega] = {
            "mode": "new_heldout_integration",
            "new_runtime_seconds": elapsed,
            "equivalent_runtime_seconds": elapsed,
        }
    assert reference is not None
    return FrequencyReconData(
        block_ids=reference.block_ids,
        strengths=checked["strengths"],
        harmonics=checked["harmonics"],
        frequencies=checked["frequencies"],
        studies=studies,
        frequency_runtime_seconds=runtimes,
        source=sources,
    )


def second_harmonic_estimators(study: StrengthStudyData) -> dict[str, np.ndarray]:
    """Return current and frozen-alternative raw second-harmonic contrasts."""
    harmonic_index = {int(value): index for index, value in enumerate(study.harmonics)}
    index = harmonic_index[2]
    positive = study.positive_cycle_fourier.mean(axis=-2)[..., index]
    negative = study.negative_cycle_fourier.mean(axis=-2)[..., index]
    unforced = study.unforced_cycle_fourier.mean(axis=-2)[..., index]
    forced_even = (positive + negative) / 2
    return {
        "current_E2": forced_even - unforced[:, None],
        "alternative_A2": forced_even,
        "unforced_U2_diagnostic": unforced,
    }


def dc_components(study: StrengthStudyData) -> dict[str, np.ndarray]:
    """Return forced-even, unforced, and required-subtraction DC components."""
    harmonic_index = {int(value): index for index, value in enumerate(study.harmonics)}
    index = harmonic_index[0]
    positive = study.positive_cycle_fourier.mean(axis=-2)[..., index].real
    negative = study.negative_cycle_fourier.mean(axis=-2)[..., index].real
    unforced = study.unforced_cycle_fourier.mean(axis=-2)[..., index].real
    forced_even = (positive + negative) / 2
    return {
        "forced_even_A0": forced_even,
        "unforced_U0": unforced,
        "current_E0": forced_even - unforced[:, None],
    }


def _bootstrap_indices(n_block: int, config: dict) -> tuple[np.ndarray, dict]:
    root = np.random.SeedSequence(config["root_entropy"])
    rng = np.random.Generator(np.random.PCG64DXSM(root))
    indices = rng.integers(
        0, n_block, size=(int(config["resamples"]), n_block), dtype=np.int32
    )
    return indices, {
        "root_entropy": int(root.entropy),
        "root_spawn_key": tuple(root.spawn_key),
        "bit_generator": "PCG64DXSM",
    }


def _simultaneous_transformed_intervals(
    observed: np.ndarray,
    bootstrap_values: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fixed-scale max-bootstrap intervals for a pre-transformed metric family."""
    standard_error = bootstrap_values.std(axis=0, ddof=1)
    active = standard_error > 0
    standardized = np.zeros_like(bootstrap_values)
    standardized[:, active] = np.abs(
        (bootstrap_values[:, active] - observed[active]) / standard_error[active]
    )
    maximum = standardized[:, active].max(axis=1) if np.any(active) else np.zeros(len(bootstrap_values))
    critical = float(np.quantile(maximum, confidence, method="higher"))
    return (
        observed - critical * standard_error,
        observed + critical * standard_error,
        critical,
        standard_error,
    )


def analyze_second_variance_ratios(
    data: FrequencyReconData, config: dict, indices: np.ndarray
) -> dict:
    """Simultaneous paired log variance-ratio analysis over the held-out grid."""
    entries = []
    current_columns = []
    alternative_columns = []
    for omega in data.frequencies:
        estimators = second_harmonic_estimators(data.studies[omega])
        current = estimators["current_E2"]
        alternative = estimators["alternative_A2"]
        for strength_index, strength in enumerate(data.strengths):
            for observable in range(3):
                current_columns.append(current[:, strength_index, observable])
                alternative_columns.append(alternative[:, strength_index, observable])
                entries.append(
                    {
                        "omega": omega,
                        "strength": float(strength),
                        "observable": STATE_NAMES[observable],
                        "structural_null": bool(SECOND_NULLS[observable]),
                    }
                )
    current = np.column_stack(current_columns)
    alternative = np.column_stack(alternative_columns)

    def radial_variance(values, axis=0):
        return np.var(values.real, axis=axis, ddof=1) + np.var(
            values.imag, axis=axis, ddof=1
        )

    current_variance = radial_variance(current)
    alternative_variance = radial_variance(alternative)
    if np.any(current_variance <= 0) or np.any(alternative_variance <= 0):
        raise ValueError("variance-ratio family requires positive radial variances")
    observed_log_ratio = np.log(alternative_variance / current_variance)
    resampled_log_ratio = np.empty((len(indices), current.shape[1]))
    batch = int(config.get("batch_size", 64))
    for start in range(0, len(indices), batch):
        selected = indices[start : start + batch]
        current_sample = current[selected]
        alternative_sample = alternative[selected]
        resampled_log_ratio[start : start + len(selected)] = np.log(
            radial_variance(alternative_sample, axis=1)
            / radial_variance(current_sample, axis=1)
        )
    lower_log, upper_log, critical, standard_error = _simultaneous_transformed_intervals(
        observed_log_ratio, resampled_log_ratio, float(config["confidence"])
    )
    for index, entry in enumerate(entries):
        entry.update(
            {
                "current_radial_cross_block_variance": current_variance[index],
                "alternative_radial_cross_block_variance": alternative_variance[index],
                "alternative_to_current_variance_ratio": math.exp(observed_log_ratio[index]),
                "simultaneous_ratio_lower": math.exp(lower_log[index]),
                "simultaneous_ratio_upper": math.exp(upper_log[index]),
                "log_ratio_bootstrap_standard_error": standard_error[index],
                "simultaneously_confirms_variance_reduction": upper_log[index] < 0,
            }
        )
    allowed = [entry for entry in entries if not entry["structural_null"]]
    return {
        "estimand_equivalence": (
            "For stationary unforced X(t)~mu0 and an external phase theta=omega*t+phi, "
            "E[C0,n]=E_mu0[g(X)] times the uniform-phase Fourier coefficient, hence "
            "E[C0,n]=0 for every nonzero integer n. Therefore E[current_E2]="
            "E[alternative_A2]. U2 remains stored as a numerical/transient diagnostic."
        ),
        "frozen_estimators": {
            "current_E2": "(C_plus,n=2 + C_minus,n=2)/2 - C_unforced,n=2",
            "alternative_A2": "(C_plus,n=2 + C_minus,n=2)/2",
            "adaptive_coefficient": "none; beta=1 versus beta=0 fixed before held-out data",
        },
        "coverage": {
            "confidence": float(config["confidence"]),
            "critical_value": critical,
            "family_metric": "log radial cross-block variance ratio",
            "coordinate_count": len(entries),
            "resampling_unit": "whole initial-state block across all frequency/strength/component coordinates",
        },
        "entries": entries,
        "allowed_component_summary": {
            "coordinate_count": len(allowed),
            "all_upper_bounds_below_one": all(
                entry["simultaneous_ratio_upper"] < 1 for entry in allowed
            ),
            "point_ratio_quantiles": np.quantile(
                [entry["alternative_to_current_variance_ratio"] for entry in allowed],
                [0, 0.1, 0.5, 0.9, 1],
            ),
            "simultaneous_upper_quantiles": np.quantile(
                [entry["simultaneous_ratio_upper"] for entry in allowed],
                [0, 0.1, 0.5, 0.9, 1],
            ),
        },
    }


def analyze_unforced_second_diagnostic(data: FrequencyReconData, config: dict) -> dict:
    """Simultaneous held-out zero checks for retained unforced n=2 coefficients."""
    columns = []
    members = []
    for omega in data.frequencies:
        values = second_harmonic_estimators(data.studies[omega])[
            "unforced_U2_diagnostic"
        ]
        for observable in range(3):
            columns.append(values[:, observable].real)
            members.append((omega, observable, "real"))
        for observable in range(3):
            columns.append(values[:, observable].imag)
            members.append((omega, observable, "imaginary"))
    matrix = np.column_stack(columns)
    mean, standard_error, maximum, metadata = _bootstrap_max_statistic(
        matrix,
        resamples=int(config["resamples"]),
        root_entropy=config["root_entropy"],
        batch_size=int(config.get("batch_size", 64)),
    )
    critical = float(np.quantile(maximum, float(config["confidence"]), method="higher"))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    entries = []
    cursor = 0
    for omega in data.frequencies:
        real_indices = list(range(cursor, cursor + 3))
        imaginary_indices = list(range(cursor + 3, cursor + 6))
        cursor += 6
        for observable in range(3):
            real_index = real_indices[observable]
            imaginary_index = imaginary_indices[observable]
            real_distance = (
                0.0
                if lower[real_index] <= 0 <= upper[real_index]
                else min(abs(lower[real_index]), abs(upper[real_index]))
            )
            imaginary_distance = (
                0.0
                if lower[imaginary_index] <= 0 <= upper[imaginary_index]
                else min(abs(lower[imaginary_index]), abs(upper[imaginary_index]))
            )
            entries.append(
                {
                    "omega": omega,
                    "observable": STATE_NAMES[observable],
                    "mean_real": mean[real_index],
                    "mean_imaginary": mean[imaginary_index],
                    "simultaneous_real_interval": [lower[real_index], upper[real_index]],
                    "simultaneous_imaginary_interval": [
                        lower[imaginary_index],
                        upper[imaginary_index],
                    ],
                    "simultaneous_complex_rectangle_excludes_origin": math.hypot(
                        real_distance, imaginary_distance
                    )
                    > 0,
                }
            )
    return {
        "interpretation": (
            "U2 is excluded from the alternative response estimator but retained "
            "as a held-out finite-spinup/transient/numerical diagnostic. Inclusion "
            "of zero is not an equivalence proof; these intervals do not replace "
            "separate bias convergence checks."
        ),
        "coverage": {
            "confidence": float(config["confidence"]),
            "critical_value": critical,
            "scalar_coordinate_count": matrix.shape[1],
            "resampling_unit": "whole held-out initial-state block",
            "bootstrap": metadata,
            "scope": "separate simultaneous family for U2 diagnostics only",
        },
        "entries": entries,
        "exclusions": [
            entry
            for entry in entries
            if entry["simultaneous_complex_rectangle_excludes_origin"]
        ],
    }


def _append_complex(columns, groups, values, **metadata):
    start = len(columns)
    for observable in range(3):
        columns.append(values[:, observable].real)
    imaginary_start = len(columns)
    for observable in range(3):
        columns.append(values[:, observable].imag)
    for observable in range(3):
        groups.append(
            VarianceFeatureGroup(
                observable=observable,
                real_index=start + observable,
                imaginary_index=imaginary_start + observable,
                **metadata,
            )
        )


def _second_decision_family(data: FrequencyReconData):
    columns = []
    groups = []
    for omega in data.frequencies:
        estimators = second_harmonic_estimators(data.studies[omega])
        for estimator in ("current_E2", "alternative_A2"):
            values = estimators[estimator]
            for strength_index, strength in enumerate(data.strengths):
                _append_complex(
                    columns,
                    groups,
                    values[:, strength_index],
                    estimator=estimator,
                    section="identification",
                    strength=float(strength),
                )
            for stop in range(3, len(data.strengths) + 1):
                prefix = data.strengths[:stop]
                fit = fit_block_power_series(values[:, :stop], prefix, (2, 4))
                upper = float(prefix[-1])
                for contribution, coefficient_index, power in (
                    ("low", 0, 2),
                    ("higher", 1, 4),
                ):
                    _append_complex(
                        columns,
                        groups,
                        fit.coefficients[:, coefficient_index] * upper**power,
                        estimator=estimator,
                        section="adequacy",
                        prefix_upper_strength=upper,
                        contribution=contribution,
                    )
    return np.column_stack(columns), groups


def analyze_second_identification(data: FrequencyReconData, config: dict) -> dict:
    matrix, groups = _second_decision_family(data)
    mean, standard_error, maximum, metadata = _bootstrap_max_statistic(
        matrix,
        resamples=int(config["resamples"]),
        root_entropy=config["root_entropy"],
        batch_size=int(config.get("batch_size", 64)),
    )
    critical = float(np.quantile(maximum, float(config["confidence"]), method="higher"))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    result = {}
    offset = 0
    groups_per_frequency = len(groups) // len(data.frequencies)
    for omega in data.frequencies:
        frequency_groups = groups[offset : offset + groups_per_frequency]
        offset += groups_per_frequency
        by_estimator = {}
        for estimator in ("current_E2", "alternative_A2"):
            identification = {}
            contributions = {}
            for group in frequency_groups:
                if group.estimator != estimator:
                    continue
                magnitude_lower, magnitude_upper = _magnitude_interval(group, lower, upper)
                record = {
                    "observable": STATE_NAMES[group.observable],
                    "structural_null": bool(SECOND_NULLS[group.observable]),
                    "magnitude_lower": magnitude_lower,
                    "magnitude_upper": magnitude_upper,
                    "simultaneous_interval_excludes_origin": magnitude_lower > 0,
                }
                if group.section == "identification":
                    identification.setdefault(str(group.strength), []).append(record)
                else:
                    key = str(group.prefix_upper_strength)
                    contributions.setdefault(key, {"low": [], "higher": []})[
                        group.contribution
                    ].append(record)
            identification_decisions = {}
            for strength, components in identification.items():
                allowed = [item for item in components if not item["structural_null"]]
                identification_decisions[strength] = {
                    "identified": any(
                        item["simultaneous_interval_excludes_origin"] for item in allowed
                    ),
                    "components": components,
                }
            adequacy = {}
            for prefix, components in contributions.items():
                low = [item for item in components["low"] if not item["structural_null"]]
                higher = [
                    item for item in components["higher"] if not item["structural_null"]
                ]
                low_lower = max(item["magnitude_lower"] for item in low)
                higher_upper = max(item["magnitude_upper"] for item in higher)
                ratio = higher_upper / low_lower if low_lower > 0 else None
                adequacy[prefix] = {
                    "higher_to_dominant_low_upper": ratio,
                    "denominator_status": "simultaneously_resolved" if low_lower > 0 else "unresolved",
                    "adequate": bool(ratio <= config["higher_order_fraction_limit"])
                    if ratio is not None
                    else False,
                    "components": components,
                }
            by_estimator[estimator] = {
                "identification": identification_decisions,
                "adequacy": adequacy,
                "identification_window": {
                    prefix: {
                        "identified": identification_decisions[prefix]["identified"],
                        "adequate": record["adequate"],
                        "in_identification_window": identification_decisions[prefix]["identified"]
                        and record["adequate"],
                    }
                    for prefix, record in adequacy.items()
                },
            }
        result[str(omega)] = by_estimator
    null_exclusions = []
    for omega, estimators in result.items():
        for estimator, decisions in estimators.items():
            for strength, record in decisions["identification"].items():
                for component in record["components"]:
                    if component["structural_null"] and component[
                        "simultaneous_interval_excludes_origin"
                    ]:
                        null_exclusions.append(
                            {
                                "omega": omega,
                                "estimator": estimator,
                                "strength": strength,
                                "observable": component["observable"],
                            }
                        )
    return {
        "coverage": {
            "confidence": float(config["confidence"]),
            "critical_value": critical,
            "scalar_coordinate_count": matrix.shape[1],
            "resampling_unit": "whole held-out initial-state block across both estimators and complete grid",
            "bootstrap": metadata,
        },
        "by_frequency": result,
        "null_control_exclusions": null_exclusions,
        "normalization": "chi2_effective(h)=-4*raw_second_harmonic/h^2",
        "higher_order_fraction_limit": float(config["higher_order_fraction_limit"]),
    }


def _fisher_correlation(x, y, axis=0):
    x_centered = x - np.mean(x, axis=axis, keepdims=True)
    y_centered = y - np.mean(y, axis=axis, keepdims=True)
    covariance = np.sum(x_centered * y_centered, axis=axis) / (x.shape[axis] - 1)
    correlation = covariance / np.sqrt(
        np.var(x, axis=axis, ddof=1) * np.var(y, axis=axis, ddof=1)
    )
    return np.arctanh(np.clip(correlation, -0.999999, 0.999999))


def analyze_dc_variance(
    data: FrequencyReconData, config: dict, indices: np.ndarray
) -> dict:
    entries = []
    forced_columns = []
    unforced_columns = []
    for omega in data.frequencies:
        parts = dc_components(data.studies[omega])
        for strength_index, strength in enumerate(data.strengths):
            for observable in range(3):
                forced_columns.append(parts["forced_even_A0"][:, strength_index, observable])
                unforced_columns.append(parts["unforced_U0"][:, observable])
                entries.append(
                    {
                        "omega": omega,
                        "strength": float(strength),
                        "observable": STATE_NAMES[observable],
                        "structural_null": bool(observable < 2),
                    }
                )
    forced = np.column_stack(forced_columns)
    unforced = np.column_stack(unforced_columns)

    def metrics(a, u, axis=0):
        va = np.var(a, axis=axis, ddof=1)
        vu = np.var(u, axis=axis, ddof=1)
        difference = a - u
        vd = np.var(difference, axis=axis, ddof=1)
        independent = va + vu
        return np.stack(
            (np.log(vd / independent), np.log(vu / va), _fisher_correlation(a, u, axis=axis)),
            axis=-1,
        )

    observed = metrics(forced, unforced).reshape(-1)
    resampled = np.empty((len(indices), len(observed)))
    batch = int(config.get("batch_size", 64))
    for start in range(0, len(indices), batch):
        selected = indices[start : start + batch]
        resampled[start : start + len(selected)] = metrics(
            forced[selected], unforced[selected], axis=1
        ).reshape(len(selected), -1)
    lower, upper, critical, standard_error = _simultaneous_transformed_intervals(
        observed, resampled, float(config["confidence"])
    )
    observed = observed.reshape(len(entries), 3)
    lower = lower.reshape(len(entries), 3)
    upper = upper.reshape(len(entries), 3)
    standard_error = standard_error.reshape(len(entries), 3)
    baseline_fraction = float(config["independent_baseline_variance_fraction"])
    block_count = len(data.block_ids)
    for index, entry in enumerate(entries):
        va = float(np.var(forced[:, index], ddof=1))
        vu = float(np.var(unforced[:, index], ddof=1))
        covariance = float(np.cov(forced[:, index], unforced[:, index], ddof=1)[0, 1])
        ratio_vu_va = math.exp(observed[index, 1])
        required_baseline_blocks = math.ceil(
            ratio_vu_va * block_count * (1 - baseline_fraction) / baseline_fraction
        )
        entry.update(
            {
                "forced_even_variance": va,
                "unforced_variance": vu,
                "forced_unforced_covariance": covariance,
                "current_difference_variance": float(
                    np.var(forced[:, index] - unforced[:, index], ddof=1)
                ),
                "paired_to_hypothetical_independent_equal_B_variance_ratio": math.exp(
                    observed[index, 0]
                ),
                "paired_ratio_simultaneous_lower": math.exp(lower[index, 0]),
                "paired_ratio_simultaneous_upper": math.exp(upper[index, 0]),
                "unforced_to_forced_variance_ratio": ratio_vu_va,
                "unforced_to_forced_ratio_simultaneous_lower": math.exp(lower[index, 1]),
                "unforced_to_forced_ratio_simultaneous_upper": math.exp(upper[index, 1]),
                "correlation": math.tanh(observed[index, 2]),
                "correlation_simultaneous_lower": math.tanh(lower[index, 2]),
                "correlation_simultaneous_upper": math.tanh(upper[index, 2]),
                "required_independent_baseline_blocks_point": required_baseline_blocks,
            }
        )
    allowed = [entry for entry in entries if not entry["structural_null"]]
    by_frequency = {}
    for omega in data.frequencies:
        selected = [entry for entry in allowed if entry["omega"] == omega]
        required = max(entry["required_independent_baseline_blocks_point"] for entry in selected)
        by_frequency[str(omega)] = {
            "maximum_required_baseline_blocks_across_strengths": required,
            "relative_total_condition_cost": (12 * block_count + required)
            / (13 * block_count),
            "paired_ratio_point_quantiles": np.quantile(
                [entry["paired_to_hypothetical_independent_equal_B_variance_ratio"] for entry in selected],
                [0, 0.5, 1],
            ),
            "correlation_point_quantiles": np.quantile(
                [entry["correlation"] for entry in selected], [0, 0.5, 1]
            ),
        }
    return {
        "estimand_requirement": (
            "DC requires E0=forced_even_A0-mu0[g]. Unlike n=2, the unforced n=0 "
            "population coefficient is generally nonzero and cannot be omitted."
        ),
        "coverage": {
            "confidence": float(config["confidence"]),
            "critical_value": critical,
            "coordinate_count": len(observed.reshape(-1)),
            "metrics": (
                "log paired/(independent equal-B) variance, log unforced/forced "
                "variance, and Fisher-z correlation"
            ),
            "resampling_unit": "whole held-out initial-state block across complete DC family",
        },
        "independent_baseline_allocation": {
            "target_fraction": baseline_fraction,
            "rule": (
                "Choose M so Var(U)/M is at most target_fraction of "
                "Var(A)/B+Var(U)/M, using allowed z point variances."
            ),
            "by_frequency": by_frequency,
            "common_baseline_note": (
                "mu0[z] is common across frequencies. A separate common baseline "
                "ensemble may use the maximum precision requirement after a fixed "
                "observation design; its uncertainty must be propagated jointly."
            ),
        },
        "entries": entries,
        "control_variate_policy": (
            "No beta is estimated here. Any covariance-fitted beta must be trained "
            "on independent blocks or cross-fit, and baseline-mean uncertainty must "
            "remain in the final joint covariance/bootstrap."
        ),
    }


def analyze_variance_reduction(data: FrequencyReconData, config: dict) -> dict:
    bootstrap = config["bootstrap"]
    indices, index_metadata = _bootstrap_indices(len(data.block_ids), bootstrap)
    second_ratios = analyze_second_variance_ratios(data, bootstrap, indices)
    unforced_second = analyze_unforced_second_diagnostic(data, bootstrap)
    second_identification = analyze_second_identification(data, bootstrap)
    dc = analyze_dc_variance(data, config["dc_design"], indices)
    resources = {}
    for omega in data.frequencies:
        study = data.studies[omega]
        cycles = study.positive_cycle_fourier.shape[-2]
        duration = cycles * 2 * np.pi / omega
        resources[str(omega)] = {
            "cycle_count": cycles,
            "nominal_observation_duration": duration,
            "runtime_seconds": data.frequency_runtime_seconds[omega],
        }
    return {
        "interpretation": (
            "Held-out confirmation study for one frozen second-harmonic variance "
            "alternative and a diagnostic DC baseline allocation. It does not "
            "select frequencies or define a formal frequency range."
        ),
        "holdout_status": "new block IDs excluded from the reconnaissance family",
        "block_ids": data.block_ids,
        "frequencies": data.frequencies,
        "strengths": data.strengths,
        "frequency_resources": resources,
        "shared_resampling_indices": {
            **index_metadata,
            "resamples": len(indices),
            "scope": "variance-ratio and DC variance families share the same held-out block draws",
        },
        "second_harmonic_variance": second_ratios,
        "unforced_second_harmonic_diagnostic": unforced_second,
        "second_harmonic_identification": second_identification,
        "dc_variance": dc,
        "response_normalization": {
            "second_harmonic": "chi2_effective(h)=-4*raw/h^2",
            "dc": "Qdc_effective(h)=raw/h^2",
            "fourier": "mean over phase times exp(-i*n*theta)",
            "second_order_factorial": "no 1/2!",
        },
        "coverage_scope": (
            "Variance ratio, U2 diagnostic, second-harmonic identification, and "
            "DC decomposition are separately calibrated simultaneous 95% families; "
            "95% coverage is not asserted jointly across those four families."
        ),
    }


def _raw_arrays(data: FrequencyReconData) -> dict:
    arrays = {
        "block_ids": np.asarray(data.block_ids, dtype=np.uint32),
        "strengths": data.strengths,
        "harmonics": data.harmonics,
        "frequencies": np.asarray(data.frequencies),
    }
    for omega in data.frequencies:
        study = data.studies[omega]
        prefix = f"omega_{_frequency_key(omega)}"
        arrays[f"{prefix}_positive_cycle_fourier"] = study.positive_cycle_fourier
        arrays[f"{prefix}_negative_cycle_fourier"] = study.negative_cycle_fourier
        arrays[f"{prefix}_unforced_cycle_fourier"] = study.unforced_cycle_fourier
        arrays[f"{prefix}_raw_proposals"] = study.raw_proposals
        arrays[f"{prefix}_initial_states"] = study.initial_states
        arrays[f"{prefix}_child_spawn_keys"] = np.asarray(study.child_spawn_keys, dtype=np.uint32)
    return arrays


def _git_provenance(repo_root: Path) -> dict:
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repo_root, text=True, capture_output=True)
    status = subprocess.run(("git", "status", "--short"), cwd=repo_root, text=True, capture_output=True)
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def persist_variance_reduction(output_dir, data, derived, config, provenance) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_frequency_summaries.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    _write_json_atomic(config_path, config)
    temporary = raw_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **_raw_arrays(data))
    os.replace(temporary, raw_path)
    _write_json_atomic(derived_path, derived)
    manifest = {
        "schema_version": 1,
        "classification": "exploratory_heldout_variance_reduction",
        "study_id": config["study_id"],
        "files": {
            path.name: f"sha256:{_file_sha256(path)}"
            for path in (config_path, raw_path, derived_path)
        },
        "array_semantics": {
            "frequency_arrays": "block,strength,state,cycle,harmonic; exp(-i*n*theta)",
            "crossing": "all signs, strengths, baseline, and frequencies share each held-out block ID",
        },
        "holdout_contract": config["holdout_contract"],
        "second_harmonic_estimators": derived["second_harmonic_variance"]["frozen_estimators"],
        "coverage": {
            "variance_ratio": derived["second_harmonic_variance"]["coverage"],
            "unforced_second_harmonic": derived[
                "unforced_second_harmonic_diagnostic"
            ]["coverage"],
            "identification": derived["second_harmonic_identification"]["coverage"],
            "dc": derived["dc_variance"]["coverage"],
        },
        "response_normalization": derived["response_normalization"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def run_variance_reduction(config_path) -> tuple[Path, dict, float]:
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_heldout_studies(config, repo_root)
    derived = analyze_variance_reduction(data, config)
    sources = [
        config_path,
        Path(__file__).resolve(),
        repo_root / "src/lorenz/frequency_reconnaissance.py",
        repo_root / "src/lorenz/strength_identifiability.py",
        repo_root / "src/lorenz/strength_bootstrap.py",
        repo_root / "src/lorenz/core.py",
        repo_root / "src/lorenz/ensemble.py",
        repo_root / config["runner_path"],
    ]
    config_identifier = f"sha256:{_file_sha256(config_path)}"
    output_dir = repo_root / config["output_root"] / (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{config_identifier[7:19]}"
    )
    provenance = {
        "config_identifier": config_identifier,
        "code_identifiers": {
            str(path.relative_to(repo_root)): f"sha256:{_file_sha256(path)}" for path in sources
        },
        "git": _git_provenance(repo_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "runtime_seconds_before_persistence": time.perf_counter() - start,
    }
    manifest = persist_variance_reduction(output_dir, data, derived, config, provenance)
    return output_dir, manifest, time.perf_counter() - start
