"""Crossed-block strength-series diagnostics for raw Fourier contrasts.

The statistical functions in this module are protocol agnostic.  Lorenz
integration and exploratory persistence live in the latter half so that the
power-series estimator can be tested independently of one pilot protocol.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

from .core import simulate_phase_samples
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from .response import phase_fourier


STATE_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class BlockPowerSeriesFit:
    """Power-series coefficients fitted independently inside every block."""

    strengths: np.ndarray
    orders: tuple[int, ...]
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray


@dataclass(frozen=True)
class StrengthStudyData:
    """Cycle Fourier summaries for a crossed strength experiment.

    ``positive_cycle_fourier`` and ``negative_cycle_fourier`` have axes
    ``block, strength, state, cycle, harmonic``.  The unforced array omits the
    strength axis because one matched baseline is shared by every strength.
    """

    block_ids: tuple[int, ...]
    strengths: np.ndarray
    harmonics: np.ndarray
    omega: float
    n_phase: int
    positive_cycle_fourier: np.ndarray
    negative_cycle_fourier: np.ndarray
    unforced_cycle_fourier: np.ndarray
    raw_proposals: np.ndarray
    initial_states: np.ndarray
    child_spawn_keys: tuple[tuple[int, ...], ...]
    generation_metadata: dict


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


def _harmonic_role(contrast: str, harmonic: int) -> str:
    if (contrast == "odd" and harmonic == 1) or (
        contrast == "even" and harmonic in (0, 2)
    ):
        return "target"
    if (harmonic % 2 == 1) == (contrast == "odd"):
        return "allowed_higher_harmonic"
    return "parity_forbidden_harmonic"


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
                "role": _harmonic_role(contrast_name, int(harmonic)),
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
                **_target_series_diagnostics(odd_target, strengths, (1, 3)),
            },
            "even_second_harmonic": {
                "raw_expansion": "h^2 * gamma2 + h^4 * gamma4",
                **_target_series_diagnostics(
                    even_second_target, strengths, (2, 4)
                ),
            },
            "even_dc": {
                "raw_expansion": "h^2 * delta2 + h^4 * delta4",
                **_target_series_diagnostics(even_dc_target, strengths, (2, 4)),
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


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_config(config: dict) -> dict:
    strengths = _checked_strengths(config["strengths"])
    harmonics = np.asarray(config["harmonics"])
    protocol = config["protocol"]
    omega = float(protocol["omega"])
    phase = float(protocol.get("phase", 0.0))
    direction = np.asarray(protocol["direction"], dtype=float)
    discard_time = float(config["discard_time"])
    n_cycle = _positive_integer(config["n_cycle"], "n_cycle")
    n_phase = _positive_integer(config["n_phase"], "n_phase")
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("direction must be a finite nonzero vector with shape (3,)")
    if not np.isfinite(discard_time) or discard_time < 0:
        raise ValueError("discard_time must be finite and nonnegative")
    if (
        harmonics.ndim != 1
        or not np.issubdtype(harmonics.dtype, np.integer)
        or len(np.unique(harmonics)) != len(harmonics)
        or not {0, 1, 2}.issubset(set(int(value) for value in harmonics))
        or len(np.unique(np.mod(harmonics, n_phase))) != len(harmonics)
    ):
        raise ValueError("harmonics must be unique, alias-free integers including 0, 1, 2")
    if 2 * int(np.max(np.abs(harmonics))) >= n_phase:
        raise ValueError("configured harmonics must lie strictly below Nyquist")
    return {
        "strengths": strengths,
        "harmonics": harmonics.astype(int),
        "omega": omega,
        "phase": phase,
        "direction": direction,
        "discard_time": discard_time,
        "n_cycle": n_cycle,
        "n_phase": n_phase,
    }


def _integrate_cycle_fourier(arguments):
    initial_state, forcing, checked, cfg = arguments
    samples = simulate_phase_samples(
        initial_state,
        forcing,
        checked["omega"],
        checked["phase"],
        checked["discard_time"],
        checked["n_cycle"],
        checked["n_phase"],
        cfg,
    )
    return phase_fourier(
        samples.values, checked["harmonics"], samples.phase_offset
    )


def generate_strength_study(config: dict) -> StrengthStudyData:
    """Generate one explicitly exploratory crossed-block strength study."""
    checked = _checked_config(config)
    block_count = _positive_integer(config["block_count"], "block_count")
    workers = _positive_integer(config.get("workers", 1), "workers")
    initial = config["initial_ensemble"]
    block_id_start = int(initial["block_id_start"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    proposal_cfg = initial["proposal"]
    proposal = SymmetricXYUniformProposal(
        x_half_width=proposal_cfg["x_half_width"],
        y_half_width=proposal_cfg["y_half_width"],
        z_bounds=tuple(proposal_cfg["z_bounds"]),
    )
    cfg = {"lorenz": dict(config["lorenz"]), "solver": dict(config["solver"])}
    blocks = generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        proposal,
        float(initial["spinup_time"]),
        cfg,
    )
    forcing_vectors = [np.zeros(3)]
    forcing_vectors.extend(
        sign * strength * checked["direction"]
        for strength in checked["strengths"]
        for sign in (1.0, -1.0)
    )
    tasks = [
        (blocks.final_states[block_index], forcing, checked, cfg)
        for block_index in range(block_count)
        for forcing in forcing_vectors
    ]
    if workers == 1:
        integrated = list(map(_integrate_cycle_fourier, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            integrated = list(executor.map(_integrate_cycle_fourier, tasks))
    all_conditions = np.asarray(integrated).reshape(
        block_count,
        1 + 2 * len(checked["strengths"]),
        3,
        checked["n_cycle"],
        len(checked["harmonics"]),
    )
    unforced = all_conditions[:, 0]
    paired = all_conditions[:, 1:].reshape(
        block_count,
        len(checked["strengths"]),
        2,
        3,
        checked["n_cycle"],
        len(checked["harmonics"]),
    )
    return StrengthStudyData(
        block_ids=blocks.block_ids,
        strengths=checked["strengths"],
        harmonics=checked["harmonics"],
        omega=checked["omega"],
        n_phase=checked["n_phase"],
        positive_cycle_fourier=paired[:, :, 0],
        negative_cycle_fourier=paired[:, :, 1],
        unforced_cycle_fourier=unforced,
        raw_proposals=blocks.raw_proposals,
        initial_states=blocks.final_states,
        child_spawn_keys=blocks.child_spawn_keys,
        generation_metadata={
            "root_entropy": blocks.root_entropy,
            "root_spawn_key": blocks.root_spawn_key,
            "bit_generator": blocks.bit_generator,
            "block_ids": blocks.block_ids,
            "child_spawn_keys": blocks.child_spawn_keys,
            "proposal": {
                "x_half_width": proposal.x_half_width,
                "y_half_width": proposal.y_half_width,
                "z_bounds": proposal.z_bounds,
            },
            "spinup_time": blocks.spinup_time,
        },
    )


def strength_target_contrasts(data: StrengthStudyData) -> dict[str, np.ndarray]:
    """Return block-level raw target contrasts from retained cycle summaries."""
    positive = np.asarray(data.positive_cycle_fourier).mean(axis=-2)
    negative = np.asarray(data.negative_cycle_fourier).mean(axis=-2)
    unforced = np.asarray(data.unforced_cycle_fourier).mean(axis=-2)
    expected = (len(data.block_ids), len(data.strengths), 3, len(data.harmonics))
    if positive.shape != expected or negative.shape != expected:
        raise ValueError("forced Fourier summaries have invalid axes")
    if unforced.shape != (len(data.block_ids), 3, len(data.harmonics)):
        raise ValueError("unforced Fourier summaries have invalid axes")
    odd = (positive - negative) / 2
    even = (positive + negative) / 2 - unforced[:, None]
    harmonic_index = {
        int(harmonic): index for index, harmonic in enumerate(data.harmonics)
    }
    return {
        "odd_fundamental": odd[..., harmonic_index[1]],
        "even_second_harmonic": even[..., harmonic_index[2]],
        "even_dc": even[..., harmonic_index[0]].real,
    }


def analyze_strength_study(data: StrengthStudyData) -> dict:
    positive = np.asarray(data.positive_cycle_fourier).mean(axis=-2)
    negative = np.asarray(data.negative_cycle_fourier).mean(axis=-2)
    unforced = np.asarray(data.unforced_cycle_fourier).mean(axis=-2)
    odd = (positive - negative) / 2
    even = (positive + negative) / 2 - unforced[:, None]
    return analyze_strength_series(
        data.block_ids, data.strengths, data.harmonics, odd, even
    )


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo_root: Path) -> dict:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "head": result.stdout.strip() if result.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def persist_strength_study(output_dir, data, derived, config, provenance) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_fourier_summaries.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    _write_json_atomic(config_path, config)
    temporary_raw = raw_path.with_suffix(".npz.tmp")
    with temporary_raw.open("wb") as stream:
        np.savez_compressed(
            stream,
            block_ids=np.asarray(data.block_ids, dtype=np.uint32),
            strengths=data.strengths,
            harmonics=data.harmonics,
            omega=np.asarray(data.omega),
            n_phase=np.asarray(data.n_phase),
            positive_cycle_fourier=data.positive_cycle_fourier,
            negative_cycle_fourier=data.negative_cycle_fourier,
            unforced_cycle_fourier=data.unforced_cycle_fourier,
            raw_proposals=data.raw_proposals,
            initial_states=data.initial_states,
            child_spawn_keys=np.asarray(data.child_spawn_keys, dtype=np.uint32),
        )
    os.replace(temporary_raw, raw_path)
    _write_json_atomic(derived_path, derived)
    manifest = {
        "schema_version": 1,
        "classification": "exploratory",
        "study_id": config["study_id"],
        "protocol_status": "provisional_method_validation_only",
        "files": {
            "config_snapshot.json": f"sha256:{_file_sha256(config_path)}",
            "raw_fourier_summaries.npz": f"sha256:{_file_sha256(raw_path)}",
            "derived_diagnostics.json": f"sha256:{_file_sha256(derived_path)}",
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
    _write_json_atomic(manifest_path, manifest)
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
    source_paths = [
        config_path,
        Path(__file__).resolve(),
        repo_root / "src/lorenz/core.py",
        repo_root / "src/lorenz/ensemble.py",
        repo_root / "src/lorenz/response.py",
        repo_root
        / config.get(
            "runner_path", "experiments/run_strength_identifiability_pilot.py"
        ),
    ]
    if "bootstrap" in config:
        source_paths.append(repo_root / "src/lorenz/strength_bootstrap.py")
    config_identifier = f"sha256:{_file_sha256(config_path)}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = repo_root / config["output_root"] / (
        f"{timestamp}_{config_identifier[7:19]}"
    )
    runtime = time.perf_counter() - start
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
        "runtime_seconds_before_persistence": runtime,
    }
    manifest = persist_strength_study(output_dir, data, derived, config, provenance)
    return output_dir, manifest, time.perf_counter() - start
