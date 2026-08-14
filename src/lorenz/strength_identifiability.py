"""Crossed-block strength-series diagnostics for raw Fourier contrasts.

The statistical functions in this module are protocol agnostic.  Lorenz
integration and exploratory persistence live in the latter half so that the
power-series estimator can be tested independently of one pilot protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .artifacts import (
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    write_json_atomic,
    write_npz_atomic,
)
from .strength_series import (
    BlockPowerSeriesFit,
    TARGET_ORDERS,
    fit_block_power_series,
)
from .strength_study import (
    STATE_NAMES,
    StrengthStudyData,
    generate_strength_study,
    harmonic_role,
    strength_fourier_components,
)


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


def _real_imag_summary(values) -> dict:
    """Summarize block values without treating real/imaginary parts as blocks."""
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("summary values need at least two finite blocks")
    n_block = values.shape[0]
    mean = values.mean(axis=0)
    flat = values.reshape(n_block, -1)
    if np.iscomplexobj(values):
        realified = np.concatenate((flat.real, flat.imag), axis=1)
        feature_order = "all flattened real parts, then all flattened imaginary parts"
    else:
        realified = np.asarray(flat, dtype=float)
        feature_order = "flattened real values"
    centered = realified - realified.mean(axis=0)
    sample_covariance = centered.T @ centered / (n_block - 1)
    joint = {
        "realified_feature_order": feature_order,
        "sample_covariance_realified": sample_covariance,
        "covariance_of_mean_realified": sample_covariance / n_block,
    }
    if np.iscomplexobj(values):
        se_real = values.real.std(axis=0, ddof=1) / np.sqrt(n_block)
        se_imag = values.imag.std(axis=0, ddof=1) / np.sqrt(n_block)
        radial_se = np.sqrt(se_real**2 + se_imag**2)
        magnitude = np.abs(mean)
        ratio = np.divide(
            magnitude,
            radial_se,
            out=np.zeros_like(magnitude, dtype=float),
            where=radial_se > 0,
        )
        ratio = np.where((radial_se == 0) & (magnitude > 0), np.inf, ratio)
        return {
            "mean_real": mean.real,
            "mean_imag": mean.imag,
            "standard_error_real": se_real,
            "standard_error_imag": se_imag,
            "mean_magnitude": magnitude,
            "radial_standard_error": radial_se,
            "magnitude_over_radial_standard_error": ratio,
            **joint,
        }
    standard_error = values.std(axis=0, ddof=1) / np.sqrt(n_block)
    magnitude = np.abs(mean)
    ratio = np.divide(
        magnitude,
        standard_error,
        out=np.zeros_like(magnitude, dtype=float),
        where=standard_error > 0,
    )
    ratio = np.where((standard_error == 0) & (magnitude > 0), np.inf, ratio)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "mean_magnitude": magnitude,
        "magnitude_over_standard_error": ratio,
        **joint,
    }


def _signal_ratio(summary: dict) -> np.ndarray:
    key = (
        "magnitude_over_radial_standard_error"
        if "magnitude_over_radial_standard_error" in summary
        else "magnitude_over_standard_error"
    )
    return np.asarray(summary[key], dtype=float)


def _sample_size_projection(signal_ratio, n_block: int) -> dict:
    """Square-root projections, conditional on the current block variance."""
    signal_ratio = np.asarray(signal_ratio, dtype=float)
    result = {}
    for target in (2.0, 3.0, 5.0):
        inverse_ratio = np.divide(
            target,
            signal_ratio,
            out=np.full_like(signal_ratio, np.inf),
            where=signal_ratio > 0,
        )
        required = np.ceil(n_block * inverse_ratio**2)
        required = np.where(np.isfinite(required), required, np.inf)
        required = np.maximum(required, 2)
        result[f"ratio_{int(target)}"] = required
    return result


def _contribution_diagnostics(fit: BlockPowerSeriesFit, high_strength: float) -> dict:
    low = fit.coefficients[:, 0] * high_strength ** fit.orders[0]
    higher = fit.coefficients[:, 1] * high_strength ** fit.orders[1]
    low_summary = _real_imag_summary(low)
    higher_summary = _real_imag_summary(higher)
    low_magnitude = np.asarray(low_summary["mean_magnitude"])
    higher_magnitude = np.asarray(higher_summary["mean_magnitude"])
    higher_se = np.asarray(
        higher_summary.get(
            "radial_standard_error", higher_summary.get("standard_error")
        )
    )
    return {
        "highest_strength_in_fit": float(high_strength),
        "low_order_contribution": low_summary,
        "higher_order_contribution": higher_summary,
        "mean_higher_to_low_magnitude": np.divide(
            higher_magnitude,
            low_magnitude,
            out=np.full_like(higher_magnitude, np.inf, dtype=float),
            where=low_magnitude > 0,
        ),
        "two_se_upper_higher_to_low_magnitude": np.divide(
            higher_magnitude + 2 * higher_se,
            low_magnitude,
            out=np.full_like(higher_magnitude, np.inf, dtype=float),
            where=low_magnitude > 0,
        ),
    }


def _paired_change_summary(current, reference) -> dict:
    current = np.asarray(current)
    reference = np.asarray(reference)
    if current.shape != reference.shape:
        raise ValueError("paired change arrays must have equal shapes")
    difference = current - reference
    summary = _real_imag_summary(difference)
    summary["current_strength_normalized"] = _real_imag_summary(current)
    summary["reference_strength_normalized"] = _real_imag_summary(reference)
    return summary


def _target_series_diagnostics(values, strengths, orders) -> dict:
    n_block = values.shape[0]
    leading_order = orders[0]
    strength_shape = (1, len(strengths)) + (1,) * (values.ndim - 2)
    normalized = values / strengths.reshape(strength_shape) ** leading_order
    raw = []
    for strength_index, strength in enumerate(strengths):
        summary = _real_imag_summary(values[:, strength_index])
        raw.append(
            {
                "strength": float(strength),
                "raw_contrast": summary,
                "block_count_projection": _sample_size_projection(
                    _signal_ratio(summary), n_block
                ),
            }
        )
    adjacent_normalized_changes = {}
    for strength_index in range(len(strengths) - 1):
        label = f"{float(strengths[strength_index])}_to_{float(strengths[strength_index + 1])}"
        adjacent_normalized_changes[label] = _paired_change_summary(
            normalized[:, strength_index + 1], normalized[:, strength_index]
        )
    prefix_fits = {}
    for stop in range(max(3, len(orders)), len(strengths) + 1):
        subset_strengths = strengths[:stop]
        fit = fit_block_power_series(values[:, :stop], subset_strengths, orders)
        prefix_fits[str(float(subset_strengths[-1]))] = {
            "strengths_in_fit": subset_strengths,
            "orders": fit.orders,
            "coefficient_summaries": [
                _real_imag_summary(fit.coefficients[:, index])
                for index in range(len(fit.orders))
            ],
            "contribution_at_highest_strength": _contribution_diagnostics(
                fit, float(subset_strengths[-1])
            ),
            "residual_rms_per_block": np.sqrt(
                np.mean(np.abs(fit.residuals) ** 2, axis=tuple(range(1, fit.residuals.ndim)))
            ),
        }
    return {
        "raw_by_strength": raw,
        "strength_normalized_by_strength": [
            _real_imag_summary(normalized[:, index])
            for index in range(len(strengths))
        ],
        "adjacent_strength_normalized_changes": adjacent_normalized_changes,
        "nested_prefix_power_fits": prefix_fits,
    }


def analyze_strength_series(
    block_ids,
    strengths,
    harmonics,
    odd_fourier,
    even_fourier,
) -> dict:
    """Analyze block-level raw odd/even Fourier contrasts across strengths."""
    block_ids = tuple(block_ids)
    strengths = _checked_strengths(strengths)
    harmonics = np.asarray(harmonics)
    odd_fourier = np.asarray(odd_fourier)
    even_fourier = np.asarray(even_fourier)
    if len(block_ids) < 2 or len(set(block_ids)) != len(block_ids):
        raise ValueError("block_ids must contain at least two unique IDs")
    expected_prefix = (len(block_ids), len(strengths))
    if (
        odd_fourier.shape != even_fourier.shape
        or odd_fourier.ndim != 4
        or odd_fourier.shape[:2] != expected_prefix
        or odd_fourier.shape[-1] != len(harmonics)
        or not np.isfinite(odd_fourier).all()
        or not np.isfinite(even_fourier).all()
    ):
        raise ValueError(
            "contrasts must be finite with axes block, strength, observable, harmonic"
        )
    if (
        harmonics.ndim != 1
        or not np.issubdtype(harmonics.dtype, np.integer)
        or len(np.unique(harmonics)) != len(harmonics)
        or not {0, 1, 2}.issubset(set(int(value) for value in harmonics))
    ):
        raise ValueError("harmonics must be unique integers including 0, 1, and 2")

    harmonic_index = {int(value): index for index, value in enumerate(harmonics)}
    odd_target = odd_fourier[..., harmonic_index[1]]
    even_second_target = even_fourier[..., harmonic_index[2]]
    even_dc_target = even_fourier[..., harmonic_index[0]].real
    harmonic_diagnostics = {"odd": {}, "even": {}}
    for contrast_name, values in (("odd", odd_fourier), ("even", even_fourier)):
        for index, harmonic in enumerate(harmonics):
            summaries = [
                _real_imag_summary(values[:, strength_index, :, index])
                for strength_index in range(len(strengths))
            ]
            harmonic_diagnostics[contrast_name][str(int(harmonic))] = {
                        "role": harmonic_role(contrast_name, int(harmonic)),
                "by_strength": summaries,
            }

    return {
        "interpretation": (
            "Exploratory crossed-block diagnostics fitted to unnormalized raw "
            "Fourier contrasts. Signal/SE and square-root block projections "
            "describe sampling resolution under the current protocol. Higher-"
            "order ratios and two-SE proxies are model diagnostics, not formal "
            "equivalence tests or accepted asymptotic-range thresholds."
        ),
        "block_count": len(block_ids),
        "block_ids": block_ids,
        "strengths": strengths,
        "observable_order": STATE_NAMES,
        "target_series": {
            "odd_fundamental": {
                "raw_expansion": "h * beta1 + h^3 * beta3",
                **_target_series_diagnostics(
                    odd_target, strengths, TARGET_ORDERS["odd_fundamental"]
                ),
            },
            "even_second_harmonic": {
                "raw_expansion": "h^2 * gamma2 + h^4 * gamma4",
                **_target_series_diagnostics(
                    even_second_target,
                    strengths,
                    TARGET_ORDERS["even_second_harmonic"],
                ),
            },
            "even_dc": {
                "raw_expansion": "h^2 * delta2 + h^4 * delta4",
                **_target_series_diagnostics(
                    even_dc_target, strengths, TARGET_ORDERS["even_dc"]
                ),
            },
        },
        "harmonic_diagnostics": harmonic_diagnostics,
        "range_semantics": {
            "asymptotic_range": (
                "strengths where target-harmonic higher-order contamination is "
                "small enough for the intended scientific precision"
            ),
            "identification_window": (
                "strengths that are both in the asymptotic range and resolved "
                "against across-block sampling variation"
            ),
        },
        "response_normalization": {
            "beta1": "odd fundamental coefficient of h; chi1[a] = 2i * beta1",
            "gamma2": (
                "even second-harmonic coefficient of h^2; "
                "chi2[a,a] = -4 * gamma2"
            ),
            "delta2": "even DC coefficient of h^2; Qdc[a,a] = delta2",
            "fourier": "mean over phase of value * exp(-i*n*theta)",
            "second_order_factorial": "Volterra convention without 1/2!",
        },
    }


def analyze_strength_study(data: StrengthStudyData) -> dict:
    components = strength_fourier_components(data)
    odd = components.odd
    even = components.forced_even - components.unforced[:, None]
    return analyze_strength_series(
        data.block_ids, data.strengths, data.harmonics, odd, even
    )


def persist_strength_study(output_dir, data, derived, config, provenance) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_fourier_summaries.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    write_json_atomic(config_path, config)
    write_npz_atomic(
        raw_path,
        {
            "block_ids": np.asarray(data.block_ids, dtype=np.uint32),
            "strengths": data.strengths,
            "harmonics": data.harmonics,
            "omega": np.asarray(data.omega),
            "n_phase": np.asarray(data.n_phase),
            "positive_cycle_fourier": data.positive_cycle_fourier,
            "negative_cycle_fourier": data.negative_cycle_fourier,
            "unforced_cycle_fourier": data.unforced_cycle_fourier,
            "raw_proposals": data.raw_proposals,
            "initial_states": data.initial_states,
            "child_spawn_keys": np.asarray(data.child_spawn_keys, dtype=np.uint32),
        },
    )
    write_json_atomic(derived_path, derived)
    manifest = {
        "schema_version": 1,
        "classification": "exploratory",
        "study_id": config["study_id"],
        "protocol_status": "provisional_method_validation_only",
        "files": {
            "config_snapshot.json": f"sha256:{file_sha256(config_path)}",
            "raw_fourier_summaries.npz": f"sha256:{file_sha256(raw_path)}",
            "derived_diagnostics.json": f"sha256:{file_sha256(derived_path)}",
        },
        "array_semantics": {
            "positive_cycle_fourier": (
                "block, strength, state, cycle, harmonic; exp(-i*n*theta)"
            ),
            "negative_cycle_fourier": (
                "block, strength, state, cycle, harmonic; exp(-i*n*theta)"
            ),
            "unforced_cycle_fourier": (
                "block, state, cycle, harmonic; one shared baseline per block"
            ),
            "block_ids": "initial-state sampling replication unit",
            "strength_pairing": "+h, -h, and unforced crossed within every block",
        },
        "protocol": config["protocol"],
        "strengths": config["strengths"],
        "response_normalization": derived["response_normalization"],
        "initial_ensemble_generation": data.generation_metadata,
        "lorenz": config["lorenz"],
        "solver": config["solver"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    if "confirmed_block_bootstrap" in derived:
        bootstrap = derived["confirmed_block_bootstrap"]
        manifest["confirmed_block_bootstrap"] = {
            "coverage": bootstrap["coverage"],
            "scalar_coordinate_count": bootstrap["family"][
                "scalar_coordinate_count"
            ],
            "physical_component_count": bootstrap["family"][
                "physical_component_count"
            ],
            "structural_null_outputs": bootstrap["structural_null_outputs"],
            "denominator_policy": bootstrap["denominator_policy"],
            "model_scope": bootstrap["model_scope"],
        }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def run_strength_study(config_path) -> tuple[Path, dict, float]:
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_strength_study(config)
    derived = analyze_strength_study(data)
    if "bootstrap" in config:
        from .strength_bootstrap import analyze_block_bootstrap

        bootstrap = config["bootstrap"]
        derived["confirmed_block_bootstrap"] = analyze_block_bootstrap(
            data.block_ids,
            data.strengths,
            strength_target_contrasts(data),
            bootstrap["structural_null_outputs"],
            confidence=float(bootstrap["confidence"]),
            resamples=int(bootstrap["resamples"]),
            root_entropy=bootstrap["root_entropy"],
            batch_size=int(bootstrap.get("batch_size", 128)),
            higher_order_fraction_limit=float(
                bootstrap["higher_order_fraction_limit"]
            ),
        )
    runner_path = repo_root / config.get(
        "runner_path", "experiments/run_strength_identifiability_pilot.py"
    )
    config_identifier = f"sha256:{file_sha256(config_path)}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = repo_root / config["output_root"] / (
        f"{timestamp}_{config_identifier[7:19]}"
    )
    runtime = time.perf_counter() - start
    provenance = {
        "config_identifier": config_identifier,
        "code_identifiers": active_source_identifiers(repo_root, runner_path),
        "git": git_provenance(repo_root),
        "environment": environment_provenance(),
        "runtime_seconds_before_persistence": runtime,
    }
    manifest = persist_strength_study(output_dir, data, derived, config, provenance)
    return output_dir, manifest, time.perf_counter() - start
