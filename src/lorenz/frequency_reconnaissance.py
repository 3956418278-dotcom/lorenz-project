"""Minimal cross-frequency reconnaissance for the provisional x-forcing protocol.

This module deliberately keeps the octave grid and its exploratory decision
family outside the stable response estimator.  Every frequency uses the same
initial-state block IDs and strength conditions.  One bootstrap draw resamples
the shared block row across the complete frequency family.
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

from .strength_bootstrap import (
    TARGET_ORDERS,
    _bootstrap_max_statistic,
    _magnitude_interval,
    build_decision_family,
)
from .strength_identifiability import (
    STATE_NAMES,
    StrengthStudyData,
    _harmonic_role,
    generate_strength_study,
    strength_target_contrasts,
)


@dataclass(frozen=True)
class FrequencyReconData:
    """Crossed block summaries keyed by forcing frequency."""

    block_ids: tuple[int, ...]
    strengths: np.ndarray
    harmonics: np.ndarray
    frequencies: tuple[float, ...]
    studies: dict[float, StrengthStudyData]
    frequency_runtime_seconds: dict[float, float]
    source: dict[float, dict]


def observation_cycle_count(
    omega: float, minimum_cycles: int, minimum_physical_time: float
) -> int:
    """Smallest integer cycle count satisfying both reconnaissance floors."""
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if isinstance(minimum_cycles, (bool, np.bool_)) or minimum_cycles < 1:
        raise ValueError("minimum_cycles must be a positive integer")
    if int(minimum_cycles) != minimum_cycles:
        raise ValueError("minimum_cycles must be a positive integer")
    if not np.isfinite(minimum_physical_time) or minimum_physical_time <= 0:
        raise ValueError("minimum_physical_time must be finite and positive")
    time_cycles = math.ceil(minimum_physical_time * omega / (2 * np.pi))
    return max(int(minimum_cycles), time_cycles)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _frequency_key(omega: float) -> str:
    return format(float(omega), ".12g").replace("-", "m").replace(".", "p")


def _validate_config(config: dict) -> dict:
    frequencies = tuple(float(value) for value in config["frequencies"])
    if (
        len(frequencies) < 2
        or any(not np.isfinite(value) or value <= 0 for value in frequencies)
        or tuple(sorted(frequencies)) != frequencies
        or len(set(frequencies)) != len(frequencies)
    ):
        raise ValueError("frequencies must be distinct, positive, and increasing")
    strengths = np.asarray(config["strengths"], dtype=float)
    harmonics = np.asarray(config["harmonics"], dtype=int)
    block_count = int(config["block_count"])
    minimum_cycles = int(config["observation_rule"]["minimum_cycles"])
    minimum_time = float(
        config["observation_rule"]["minimum_physical_time"]
    )
    cycles = {
        omega: observation_cycle_count(omega, minimum_cycles, minimum_time)
        for omega in frequencies
    }
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    if strengths.ndim != 1 or len(strengths) < 3 or np.any(np.diff(strengths) <= 0):
        raise ValueError("strengths must contain at least three increasing values")
    if harmonics.ndim != 1 or not {0, 1, 2}.issubset(set(harmonics.tolist())):
        raise ValueError("harmonics must include 0, 1, and 2")
    return {
        "frequencies": frequencies,
        "strengths": strengths,
        "harmonics": harmonics,
        "block_count": block_count,
        "cycles": cycles,
    }


def _validate_parent_artifact(
    repo_root: Path, config: dict, checked: dict
) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    reuse = config["reuse_parent_frequency"]
    parent_dir = repo_root / reuse["artifact"]
    manifest_path = parent_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_config = json.loads(
        (parent_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    expected_manifest_hash = reuse.get("manifest_sha256")
    if expected_manifest_hash and _file_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("parent manifest hash does not match the configured hash")
    for name, recorded in manifest["files"].items():
        actual = f"sha256:{_file_sha256(parent_dir / name)}"
        if actual != recorded:
            raise ValueError(f"parent artifact hash mismatch for {name}")
    for relative, recorded in manifest["provenance"]["code_identifiers"].items():
        path = repo_root / relative
        actual = f"sha256:{_file_sha256(path)}"
        if actual != recorded:
            raise ValueError(
                f"current code no longer matches parent provenance for {relative}"
            )

    omega = float(reuse["omega"])
    expected = {
        "strengths": checked["strengths"],
        "harmonics": checked["harmonics"],
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(np.asarray(parent_config[name]), values)
    scalar_expectations = {
        "discard_time": float(config["discard_time"]),
        "n_phase": int(config["n_phase"]),
        "n_cycle": int(checked["cycles"][omega]),
    }
    for name, value in scalar_expectations.items():
        if parent_config[name] != value:
            raise ValueError(f"parent {name} does not match reconnaissance")
    if parent_config["protocol"] != {
        "status": "provisional_method_validation_only",
        "omega": omega,
        "direction": config["protocol"]["direction"],
        "phase": float(config["protocol"].get("phase", 0.0)),
    }:
        raise ValueError("parent protocol does not match the reuse contract")
    for section in ("lorenz", "solver", "initial_ensemble"):
        if parent_config[section] != config[section]:
            raise ValueError(f"parent {section} does not match reconnaissance")

    raw = np.load(parent_dir / "raw_fourier_summaries.npz", allow_pickle=False)
    required = {
        "block_ids",
        "strengths",
        "harmonics",
        "omega",
        "n_phase",
        "positive_cycle_fourier",
        "negative_cycle_fourier",
        "unforced_cycle_fourier",
        "raw_proposals",
        "initial_states",
        "child_spawn_keys",
    }
    if not required.issubset(raw.files):
        raw.close()
        raise ValueError("parent raw artifact lacks required arrays")
    try:
        np.testing.assert_array_equal(raw["strengths"], checked["strengths"])
        np.testing.assert_array_equal(raw["harmonics"], checked["harmonics"])
        if float(raw["omega"]) != omega or int(raw["n_phase"]) != int(
            config["n_phase"]
        ):
            raise ValueError("parent raw protocol metadata do not match config")
        manifest_ids = np.asarray(
            manifest["initial_ensemble_generation"]["block_ids"], dtype=np.uint32
        )
        manifest_spawn_keys = np.asarray(
            manifest["initial_ensemble_generation"]["child_spawn_keys"],
            dtype=np.uint32,
        )
        np.testing.assert_array_equal(raw["block_ids"], manifest_ids)
        np.testing.assert_array_equal(raw["child_spawn_keys"], manifest_spawn_keys)
        parent_block_count = len(manifest_ids)
        if (
            raw["raw_proposals"].shape != (parent_block_count, 3)
            or raw["initial_states"].shape != (parent_block_count, 3)
            or not np.isfinite(raw["raw_proposals"]).all()
            or not np.isfinite(raw["initial_states"]).all()
            or not np.isfinite(raw["positive_cycle_fourier"]).all()
            or not np.isfinite(raw["negative_cycle_fourier"]).all()
            or not np.isfinite(raw["unforced_cycle_fourier"]).all()
        ):
            raise ValueError("parent raw arrays have invalid shape or nonfinite values")
    except Exception:
        raw.close()
        raise
    return manifest, parent_config, raw


def _study_from_parent(
    repo_root: Path, config: dict, checked: dict
) -> tuple[StrengthStudyData, dict]:
    manifest, _, raw = _validate_parent_artifact(repo_root, config, checked)
    try:
        count = checked["block_count"]
        block_ids = tuple(int(value) for value in raw["block_ids"][:count])
        initial = config["initial_ensemble"]
        expected_ids = tuple(
            range(
                int(initial["block_id_start"]),
                int(initial["block_id_start"]) + count,
            )
        )
        if block_ids != expected_ids:
            raise ValueError("parent block IDs are not the configured crossed subset")
        omega = float(config["reuse_parent_frequency"]["omega"])
        positive = np.asarray(raw["positive_cycle_fourier"][:count]).copy()
        negative = np.asarray(raw["negative_cycle_fourier"][:count]).copy()
        unforced = np.asarray(raw["unforced_cycle_fourier"][:count]).copy()
        expected_forced = (
            count,
            len(checked["strengths"]),
            3,
            checked["cycles"][omega],
            len(checked["harmonics"]),
        )
        if positive.shape != expected_forced or negative.shape != expected_forced:
            raise ValueError("parent forced array axes do not match the reuse contract")
        if unforced.shape != (
            count,
            3,
            checked["cycles"][omega],
            len(checked["harmonics"]),
        ):
            raise ValueError("parent unforced array axes do not match the reuse contract")
        generation = dict(manifest["initial_ensemble_generation"])
        generation["block_ids"] = list(block_ids)
        generation["child_spawn_keys"] = [
            list(row) for row in np.asarray(raw["child_spawn_keys"][:count])
        ]
        data = StrengthStudyData(
            block_ids=block_ids,
            strengths=np.asarray(raw["strengths"]).copy(),
            harmonics=np.asarray(raw["harmonics"]).copy(),
            omega=float(raw["omega"]),
            n_phase=int(raw["n_phase"]),
            positive_cycle_fourier=positive,
            negative_cycle_fourier=negative,
            unforced_cycle_fourier=unforced,
            raw_proposals=np.asarray(raw["raw_proposals"][:count]).copy(),
            initial_states=np.asarray(raw["initial_states"][:count]).copy(),
            child_spawn_keys=tuple(
                tuple(int(item) for item in row)
                for row in np.asarray(raw["child_spawn_keys"][:count])
            ),
            generation_metadata=generation,
        )
    finally:
        raw.close()
    parent_runtime = float(manifest["provenance"]["runtime_seconds_before_persistence"])
    equivalent_runtime = parent_runtime * checked["block_count"] / int(
        manifest["initial_ensemble_generation"]["block_ids"][-1]
        - manifest["initial_ensemble_generation"]["block_ids"][0]
        + 1
    )
    source = {
        "mode": "validated_parent_subset",
        "artifact": config["reuse_parent_frequency"]["artifact"],
        "manifest_sha256": _file_sha256(
            repo_root / config["reuse_parent_frequency"]["artifact"] / "manifest.json"
        ),
        "raw_sha256": manifest["files"]["raw_fourier_summaries.npz"],
        "parent_block_count": len(manifest["initial_ensemble_generation"]["block_ids"]),
        "subset_rule": "first configured block_count rows after exact block-ID check",
        "equivalent_runtime_seconds": equivalent_runtime,
        "new_runtime_seconds": 0.0,
    }
    return data, source


def _single_frequency_config(config: dict, omega: float, n_cycle: int) -> dict:
    return {
        "strengths": config["strengths"],
        "harmonics": config["harmonics"],
        "discard_time": config["discard_time"],
        "n_cycle": n_cycle,
        "n_phase": config["n_phase"],
        "block_count": config["block_count"],
        "workers": config.get("workers", 1),
        "lorenz": config["lorenz"],
        "solver": config["solver"],
        "initial_ensemble": config["initial_ensemble"],
        "protocol": {
            "omega": omega,
            "direction": config["protocol"]["direction"],
            "phase": float(config["protocol"].get("phase", 0.0)),
        },
    }


def generate_frequency_reconnaissance(config: dict, repo_root: Path) -> FrequencyReconData:
    """Generate new frequencies and validate/reuse the declared parent subset."""
    checked = _validate_config(config)
    reuse_omega = float(config["reuse_parent_frequency"]["omega"])
    if reuse_omega not in checked["frequencies"]:
        raise ValueError("reuse frequency must belong to the configured grid")
    studies: dict[float, StrengthStudyData] = {}
    runtime: dict[float, float] = {}
    source: dict[float, dict] = {}
    parent, parent_source = _study_from_parent(repo_root, config, checked)
    studies[reuse_omega] = parent
    runtime[reuse_omega] = float(parent_source["equivalent_runtime_seconds"])
    source[reuse_omega] = parent_source

    for omega in checked["frequencies"]:
        if omega == reuse_omega:
            continue
        start = time.perf_counter()
        study = generate_strength_study(
            _single_frequency_config(config, omega, checked["cycles"][omega])
        )
        elapsed = time.perf_counter() - start
        if study.block_ids != parent.block_ids:
            raise RuntimeError("generated frequency does not preserve crossed block IDs")
        np.testing.assert_array_equal(study.raw_proposals, parent.raw_proposals)
        np.testing.assert_array_equal(study.initial_states, parent.initial_states)
        if study.child_spawn_keys != parent.child_spawn_keys:
            raise RuntimeError("generated frequency does not preserve spawn keys")
        studies[omega] = study
        runtime[omega] = elapsed
        source[omega] = {
            "mode": "new_integration",
            "new_runtime_seconds": elapsed,
            "equivalent_runtime_seconds": elapsed,
        }
    return FrequencyReconData(
        block_ids=parent.block_ids,
        strengths=checked["strengths"],
        harmonics=checked["harmonics"],
        frequencies=checked["frequencies"],
        studies=studies,
        frequency_runtime_seconds=runtime,
        source=source,
    )


def _summary(values: np.ndarray, critical: float) -> dict:
    values = np.asarray(values)
    n_block = values.shape[0]
    mean = values.mean(axis=0)
    if values.ndim != 1 or n_block < 2:
        raise ValueError("component summary requires one scalar per block")
    if np.iscomplexobj(values):
        variance_real = float(values.real.var(ddof=1))
        variance_imag = float(values.imag.var(ddof=1))
        covariance = float(np.cov(values.real, values.imag, ddof=1)[0, 1])
        se_real = math.sqrt(variance_real / n_block)
        se_imag = math.sqrt(variance_imag / n_block)
        radial_se = math.hypot(se_real, se_imag)
        halfwidth = critical * radial_se
        return {
            "mean_real": float(mean.real),
            "mean_imag": float(mean.imag),
            "mean_magnitude": float(abs(mean)),
            "cross_block_variance_real": variance_real,
            "cross_block_variance_imaginary": variance_imag,
            "cross_block_covariance_real_imaginary": covariance,
            "standard_error_real": se_real,
            "standard_error_imaginary": se_imag,
            "radial_standard_error": radial_se,
            "joint_radial_bound_scale": halfwidth,
            "signal_to_joint_bound": float(abs(mean) / halfwidth)
            if halfwidth > 0
            else None,
            "radial_block_standard_deviation": math.sqrt(
                variance_real + variance_imag
            ),
        }
    variance = float(values.var(ddof=1))
    se = math.sqrt(variance / n_block)
    halfwidth = critical * se
    return {
        "mean": float(mean),
        "mean_magnitude": float(abs(mean)),
        "cross_block_variance": variance,
        "standard_error": se,
        "joint_bound_scale": halfwidth,
        "signal_to_joint_bound": float(abs(mean) / halfwidth)
        if halfwidth > 0
        else None,
        "radial_block_standard_deviation": math.sqrt(variance),
    }


def _normalized_target(target: str, values: np.ndarray, strength: float) -> np.ndarray:
    if target == "odd_fundamental":
        return 2j * values / strength
    if target == "even_second_harmonic":
        return -4 * values / strength**2
    if target == "even_dc":
        return values / strength**2
    raise KeyError(target)


def _required_information_multiplier(values: np.ndarray, critical: float) -> float | None:
    values = np.asarray(values)
    mean = values.mean()
    if np.iscomplexobj(values):
        coordinates = (
            (abs(float(mean.real)), values.real.std(ddof=1) / math.sqrt(len(values))),
            (abs(float(mean.imag)), values.imag.std(ddof=1) / math.sqrt(len(values))),
        )
    else:
        coordinates = ((abs(float(mean)), values.std(ddof=1) / math.sqrt(len(values))),)
    candidates = [
        (critical * se / signal) ** 2
        for signal, se in coordinates
        if signal > 0 and se > 0
    ]
    return min(candidates) if candidates else None


def _target_reports(data: FrequencyReconData, critical: float) -> dict:
    reports = {}
    for omega in data.frequencies:
        study = data.studies[omega]
        targets = strength_target_contrasts(study)
        duration = study.positive_cycle_fourier.shape[-2] * 2 * np.pi / omega
        equivalent_runtime = data.frequency_runtime_seconds[omega]
        frequency_report = {}
        for target, values in targets.items():
            by_strength = {}
            for strength_index, strength in enumerate(data.strengths):
                components = []
                for observable in range(values.shape[-1]):
                    raw = values[:, strength_index, observable]
                    normalized = _normalized_target(target, raw, float(strength))
                    normalized_summary = _summary(normalized, critical)
                    multiplier = _required_information_multiplier(normalized, critical)
                    radial_sd = normalized_summary["radial_block_standard_deviation"]
                    components.append(
                        {
                            "observable": STATE_NAMES[observable],
                            "raw_target_contrast": _summary(raw, critical),
                            "normalized_response": normalized_summary,
                            "normalization": {
                                "odd_fundamental": "2i * raw / h",
                                "even_second_harmonic": "-4 * raw / h^2",
                                "even_dc": "raw / h^2",
                            }[target],
                            "noise_scale": {
                                "physical_time_normalized": radial_sd
                                * math.sqrt(duration),
                                "compute_normalized": radial_sd
                                * math.sqrt(equivalent_runtime / len(data.block_ids)),
                            },
                            "rough_information_projection": {
                                "multiplier_for_one_coordinate_to_clear_joint_bound": multiplier,
                                "blocks_at_fixed_observation": (
                                    int(math.ceil(len(data.block_ids) * multiplier))
                                    if multiplier is not None
                                    else None
                                ),
                                "physical_observation_time_at_fixed_blocks": (
                                    duration * multiplier if multiplier is not None else None
                                ),
                            },
                        }
                    )
                by_strength[str(float(strength))] = components
            frequency_report[target] = by_strength
        reports[str(omega)] = frequency_report
    return reports


def _multi_frequency_bootstrap(data: FrequencyReconData, config: dict) -> dict:
    bootstrap = config["bootstrap"]
    nulls = bootstrap["structural_null_outputs"]
    matrices = []
    mappings = {}
    offset = 0
    for omega in data.frequencies:
        if data.studies[omega].block_ids != data.block_ids:
            raise ValueError(
                f"frequency {omega} block IDs/order do not match the crossed family"
            )
        matrix, features, groups, _ = build_decision_family(
            strength_target_contrasts(data.studies[omega]), data.strengths, nulls
        )
        matrices.append(matrix)
        mappings[omega] = {
            "features": features,
            "groups": groups,
            "offset": offset,
            "stop": offset + matrix.shape[1],
        }
        offset += matrix.shape[1]
    family = np.concatenate(matrices, axis=1)
    mean, standard_error, statistics, metadata = _bootstrap_max_statistic(
        family,
        resamples=int(bootstrap["resamples"]),
        root_entropy=bootstrap["root_entropy"],
        batch_size=int(bootstrap.get("batch_size", 128)),
    )
    confidence = float(bootstrap["confidence"])
    critical = float(np.quantile(statistics, confidence, method="higher"))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    per_frequency = {}
    null_exclusions = []
    limit = float(bootstrap["higher_order_fraction_limit"])
    for omega in data.frequencies:
        mapping = mappings[omega]
        identification = {target: {} for target in TARGET_ORDERS}
        adequacy_components = {}
        for local_group in mapping["groups"]:
            group = local_group.__class__(
                **{
                    **local_group.__dict__,
                    "real_index": local_group.real_index + mapping["offset"],
                    "imaginary_index": (
                        local_group.imaginary_index + mapping["offset"]
                        if local_group.imaginary_index is not None
                        else None
                    ),
                }
            )
            magnitude_lower, magnitude_upper = _magnitude_interval(group, lower, upper)
            record = {
                "observable": STATE_NAMES[group.observable],
                "structural_null": bool(group.structural_null),
                "magnitude_lower": magnitude_lower,
                "magnitude_upper": magnitude_upper,
                "simultaneous_interval_excludes_origin": magnitude_lower > 0,
            }
            if group.section == "identification":
                key = str(float(group.strength))
                identification[group.target].setdefault(key, []).append(record)
                if group.structural_null and magnitude_lower > 0:
                    null_exclusions.append(
                        {
                            "omega": omega,
                            "target": group.target,
                            "strength": group.strength,
                            "observable": STATE_NAMES[group.observable],
                        }
                    )
            else:
                key = (group.target, float(group.prefix_upper_strength))
                adequacy_components.setdefault(key, {"low": [], "higher": []})[
                    group.contribution
                ].append(record)

        identification_decisions = {target: {} for target in TARGET_ORDERS}
        for target, by_strength in identification.items():
            for strength, components in by_strength.items():
                allowed = [item for item in components if not item["structural_null"]]
                identification_decisions[target][strength] = {
                    "identified": any(
                        item["simultaneous_interval_excludes_origin"]
                        for item in allowed
                    ),
                    "components": components,
                }
        adequacy = {target: {} for target in TARGET_ORDERS}
        for (target, prefix), contributions in adequacy_components.items():
            allowed_low = [item for item in contributions["low"] if not item["structural_null"]]
            allowed_higher = [
                item for item in contributions["higher"] if not item["structural_null"]
            ]
            low_lower = max(item["magnitude_lower"] for item in allowed_low)
            higher_upper = max(item["magnitude_upper"] for item in allowed_higher)
            ratio = higher_upper / low_lower if low_lower > 0 else None
            adequacy[target][str(prefix)] = {
                "dominant_low_order_magnitude_lower": low_lower,
                "maximum_higher_order_magnitude_upper": higher_upper,
                "higher_to_dominant_low_upper": ratio,
                "denominator_status": "simultaneously_resolved" if low_lower > 0 else "unresolved",
                "adequate": bool(ratio <= limit) if ratio is not None else False,
                "fraction_limit": limit,
                "components": contributions,
            }
        windows = {target: {} for target in TARGET_ORDERS}
        for target in TARGET_ORDERS:
            for prefix, adequacy_record in adequacy[target].items():
                identified = identification_decisions[target][prefix]["identified"]
                windows[target][prefix] = {
                    "identified": identified,
                    "adequate": adequacy_record["adequate"],
                    "in_identification_window": identified and adequacy_record["adequate"],
                }
        per_frequency[str(omega)] = {
            "identification": identification_decisions,
            "adequacy": adequacy,
            "identification_window": windows,
        }

    members = []
    for omega in data.frequencies:
        for feature in mappings[omega]["features"]:
            members.append(
                {
                    "omega": omega,
                    "section": feature.section,
                    "target": feature.target,
                    "observable": STATE_NAMES[feature.observable],
                    "part": feature.part,
                    "strength": feature.strength,
                    "prefix_upper_strength": feature.prefix_upper_strength,
                    "contribution": feature.contribution,
                    "structural_null": feature.structural_null,
                }
            )
    return {
        "coverage": {
            "confidence": confidence,
            "resamples": int(bootstrap["resamples"]),
            "critical_value": critical,
            "quantile_method": "higher",
            "resampling_unit": "whole initial-state block shared across all frequencies",
            "bootstrap": metadata,
        },
        "family": {
            "scalar_coordinate_count": family.shape[1],
            "members": members,
            "mean_realified": mean,
            "standard_error_realified": standard_error,
            "simultaneous_lower_realified": lower,
            "simultaneous_upper_realified": upper,
            "frequency_major_order": list(data.frequencies),
        },
        "structural_null_outputs": nulls,
        "null_control_exclusions": null_exclusions,
        "by_frequency": per_frequency,
        "denominator_policy": (
            "Structural-null outputs are controls and never enter ratios. "
            "Unresolved leading denominators fail adequacy without forming a ratio."
        ),
    }


def _non_target_diagnostics(data: FrequencyReconData, critical: float) -> dict:
    result = {}
    for omega in data.frequencies:
        study = data.studies[omega]
        positive = study.positive_cycle_fourier.mean(axis=-2)
        negative = study.negative_cycle_fourier.mean(axis=-2)
        unforced = study.unforced_cycle_fourier.mean(axis=-2)
        contrasts = {
            "odd": (positive - negative) / 2,
            "even": (positive + negative) / 2 - unforced[:, None],
        }
        entries = []
        for contrast_name, values in contrasts.items():
            for harmonic_index, harmonic in enumerate(study.harmonics):
                role = _harmonic_role(contrast_name, int(harmonic))
                if role == "target":
                    continue
                for strength_index, strength in enumerate(study.strengths):
                    components = [
                        {
                            "observable": STATE_NAMES[observable],
                            **_summary(
                                values[:, strength_index, observable, harmonic_index],
                                critical,
                            ),
                        }
                        for observable in range(3)
                    ]
                    entries.append(
                        {
                            "contrast": contrast_name,
                            "harmonic": int(harmonic),
                            "role": role,
                            "strength": float(strength),
                            "components": components,
                            "maximum_signal_to_joint_bound": max(
                                item["signal_to_joint_bound"] or 0.0
                                for item in components
                            ),
                        }
                    )
        result[str(omega)] = entries
    return result


def analyze_frequency_reconnaissance(data: FrequencyReconData, config: dict) -> dict:
    joint = _multi_frequency_bootstrap(data, config)
    critical = joint["coverage"]["critical_value"]
    target_reports = _target_reports(data, critical)
    frequency_resources = {}
    for omega in data.frequencies:
        cycles = data.studies[omega].positive_cycle_fourier.shape[-2]
        period = 2 * np.pi / omega
        duration = cycles * period
        sample_span = (cycles - 1 / data.studies[omega].n_phase) * period
        frequency_resources[str(omega)] = {
            "cycle_count": cycles,
            "period": period,
            "nominal_observation_duration": duration,
            "sample_time_span": sample_span,
            "integration_horizon_through_last_sample": float(config["discard_time"])
            + sample_span,
            "new_runtime_seconds": data.source[omega]["new_runtime_seconds"],
            "equivalent_runtime_seconds": data.source[omega][
                "equivalent_runtime_seconds"
            ],
            "source": data.source[omega],
        }
    return {
        "interpretation": (
            "Minimal exploratory frequency reconnaissance on a fixed octave grid. "
            "It tests whether the provisional omega=2 difficulty is plausibly "
            "frequency-specific; it is neither a formal frequency scan nor a "
            "basis for selecting an isolated favorable frequency."
        ),
        "block_ids": data.block_ids,
        "strengths": data.strengths,
        "harmonics": data.harmonics,
        "frequencies": data.frequencies,
        "frequency_resources": frequency_resources,
        "joint_bootstrap": joint,
        "target_reports": target_reports,
        "non_target_diagnostics": _non_target_diagnostics(data, critical),
        "response_normalization": {
            "odd_fundamental": "chi1_effective(h) = 2i * O_hat_1(h) / h",
            "even_second_harmonic": "chi2_effective(h) = -4 * E_hat_2(h) / h^2",
            "even_dc": "Qdc_effective(h) = E_hat_0(h) / h^2",
            "fourier": "mean over phase of value * exp(-i*n*theta)",
            "second_order_factorial": "Volterra convention without 1/2!",
        },
        "projection_limitations": (
            "Rough B/time projections hold current means, cross-block variance, "
            "and the full-family critical value fixed and assume inverse information "
            "scaling. They address identification only, not higher-order adequacy, "
            "point-estimate drift, or convergence beyond observed windows."
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
        arrays[f"{prefix}_child_spawn_keys"] = np.asarray(
            study.child_spawn_keys, dtype=np.uint32
        )
    return arrays


def _git_provenance(repo_root: Path) -> dict:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo_root, text=True, capture_output=True
    )
    status = subprocess.run(
        ("git", "status", "--short"), cwd=repo_root, text=True, capture_output=True
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def persist_frequency_reconnaissance(
    output_dir: Path,
    data: FrequencyReconData,
    derived: dict,
    config: dict,
    provenance: dict,
) -> dict:
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
        "classification": "exploratory_frequency_reconnaissance",
        "study_id": config["study_id"],
        "protocol_status": "provisional_design_diagnostic_only",
        "files": {
            path.name: f"sha256:{_file_sha256(path)}"
            for path in (config_path, raw_path, derived_path)
        },
        "array_semantics": {
            "frequency_arrays": (
                "omega_<key> positive/negative: block,strength,state,cycle,harmonic; "
                "unforced: block,state,cycle,harmonic; exp(-i*n*theta)"
            ),
            "crossing": (
                "identical initial-state block IDs, proposals, post-spinup states, "
                "strengths, signs, and baseline condition across every frequency"
            ),
            "variable_cycle_axis": "cycle count is frequency-specific and recorded",
        },
        "frequencies": list(data.frequencies),
        "frequency_resources": derived["frequency_resources"],
        "joint_bootstrap": {
            "coverage": derived["joint_bootstrap"]["coverage"],
            "scalar_coordinate_count": derived["joint_bootstrap"]["family"][
                "scalar_coordinate_count"
            ],
            "null_control_exclusions": derived["joint_bootstrap"][
                "null_control_exclusions"
            ],
            "denominator_policy": derived["joint_bootstrap"]["denominator_policy"],
        },
        "response_normalization": derived["response_normalization"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def run_frequency_reconnaissance(config_path) -> tuple[Path, dict, float]:
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_frequency_reconnaissance(config, repo_root)
    derived = analyze_frequency_reconnaissance(data, config)
    source_paths = [
        config_path,
        Path(__file__).resolve(),
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
            str(path.relative_to(repo_root)): f"sha256:{_file_sha256(path)}"
            for path in source_paths
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
    manifest = persist_frequency_reconnaissance(
        output_dir, data, derived, config, provenance
    )
    return output_dir, manifest, time.perf_counter() - start
