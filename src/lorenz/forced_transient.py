"""Block-preserving diagnostics for forced-transient removal."""

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
from scipy.integrate import solve_ivp

from .core import _forcing_angle, check_solution, lorenz_rhs
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from .response import phase_fourier


CONDITION_NAMES = ("positive", "negative", "unforced")
STATE_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class ForcedTransientStudyData:
    """Cycle Fourier summaries for nested physical discard times.

    ``cycle_fourier`` has axes
    ``(discard, block, condition, state, cycle, harmonic)``. Conditions are
    ordered as :data:`CONDITION_NAMES`. Every discard uses the same number of
    post-discard cycles and phase points.
    """

    block_ids: tuple[int, ...]
    discard_times: np.ndarray
    harmonics: np.ndarray
    omega: float
    n_phase: int
    cycle_fourier: np.ndarray
    phase_offsets: np.ndarray
    raw_proposals: np.ndarray
    initial_states: np.ndarray
    child_spawn_keys: tuple[tuple[int, ...], ...]
    generation_metadata: dict


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_study_grid(config: dict):
    discard_times = np.asarray(config["discard_times"], dtype=float)
    if (
        discard_times.ndim != 1
        or len(discard_times) < 2
        or not np.isfinite(discard_times).all()
        or discard_times[0] < 0
        or np.any(np.diff(discard_times) <= 0)
    ):
        raise ValueError(
            "discard_times must be finite, strictly increasing, nonnegative, "
            "and contain at least two values"
        )
    harmonics = np.asarray(config["harmonics"])
    if (
        harmonics.ndim != 1
        or len(harmonics) < 1
        or not np.issubdtype(harmonics.dtype, np.integer)
        or len(np.unique(harmonics)) != len(harmonics)
    ):
        raise ValueError("harmonics must be a one-dimensional vector of unique integers")
    required = {0, 1, 2}
    if not required.issubset(set(int(value) for value in harmonics)):
        raise ValueError("harmonics must include the target harmonics 0, 1, and 2")
    omega = float(config["protocol"]["omega"])
    phase = float(config["protocol"].get("phase", 0.0))
    strength = float(config["protocol"]["strength"])
    direction = np.asarray(config["protocol"]["direction"], dtype=float)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("direction must be a finite vector with shape (3,)")
    if not np.any(direction):
        raise ValueError("direction must be nonzero")
    n_cycle = _positive_integer(config["n_cycle"], "n_cycle")
    n_phase = _positive_integer(config["n_phase"], "n_phase")
    if len(np.unique(np.mod(harmonics, n_phase))) != len(harmonics):
        raise ValueError("harmonics must not alias on the configured phase grid")
    return discard_times, harmonics.astype(int), omega, phase, strength, direction, n_cycle, n_phase


def _nested_cycle_fourier(arguments):
    (
        initial_state,
        forcing_vector,
        discard_times,
        harmonics,
        omega,
        phase,
        n_cycle,
        n_phase,
        cfg,
    ) = arguments
    period = 2 * np.pi / omega
    cycle = np.arange(n_cycle, dtype=float)
    phase_index = np.arange(n_phase, dtype=float)
    windows = np.asarray(
        [
            discard
            + (cycle[:, None] + phase_index[None, :] / n_phase) * period
            for discard in discard_times
        ]
    )
    flat_windows = windows.reshape(len(discard_times), -1)
    unique_times, inverse = np.unique(flat_windows, return_inverse=True)
    if not np.isfinite(unique_times).all() or np.any(np.diff(unique_times) <= 0):
        raise ValueError("nested sampling times must be finite and increasing")

    if unique_times[-1] == 0.0:
        sampled = np.asarray(initial_state, dtype=float)[:, None]
    else:
        sol = solve_ivp(
            lambda t, state: lorenz_rhs(
                t, state, cfg, forcing_vector, omega, phase
            ),
            (0.0, float(unique_times[-1])),
            np.asarray(initial_state, dtype=float),
            t_eval=unique_times,
            **cfg["solver"],
        )
        check_solution(sol, (3, len(unique_times)))
        sampled = sol.y
    values = sampled[:, inverse].reshape(
        3, len(discard_times), n_cycle, n_phase
    ).transpose(1, 0, 2, 3)
    coefficients = np.empty(
        (len(discard_times), 3, n_cycle, len(harmonics)), dtype=complex
    )
    for discard_index, discard in enumerate(discard_times):
        phase_offset = np.mod(_forcing_angle(discard, omega, phase), 2 * np.pi)
        coefficients[discard_index] = phase_fourier(
            values[discard_index], harmonics, phase_offset
        )
    return coefficients


def generate_forced_transient_study(config: dict) -> ForcedTransientStudyData:
    """Generate nested-discard summaries for one explicitly configured protocol."""
    (
        discard_times,
        harmonics,
        omega,
        phase,
        strength,
        direction,
        n_cycle,
        n_phase,
    ) = _checked_study_grid(config)
    block_count = _positive_integer(config["block_count"], "block_count")
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    workers = _positive_integer(config.get("workers", 1), "workers")
    initial = config["initial_ensemble"]
    block_id_start = int(initial["block_id_start"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    proposal_config = initial["proposal"]
    proposal = SymmetricXYUniformProposal(
        x_half_width=proposal_config["x_half_width"],
        y_half_width=proposal_config["y_half_width"],
        z_bounds=tuple(proposal_config["z_bounds"]),
    )
    cfg = {"lorenz": dict(config["lorenz"]), "solver": dict(config["solver"])}
    blocks = generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        proposal,
        float(initial["spinup_time"]),
        cfg,
    )
    forcing_vectors = (
        strength * direction,
        -strength * direction,
        np.zeros(3),
    )
    tasks = [
        (
            blocks.final_states[block_index],
            forcing_vector,
            discard_times,
            harmonics,
            omega,
            phase,
            n_cycle,
            n_phase,
            cfg,
        )
        for block_index in range(block_count)
        for forcing_vector in forcing_vectors
    ]
    if workers == 1:
        integrated = list(map(_nested_cycle_fourier, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            integrated = list(executor.map(_nested_cycle_fourier, tasks))
    cycle_fourier = np.asarray(integrated).reshape(
        block_count,
        len(CONDITION_NAMES),
        len(discard_times),
        3,
        n_cycle,
        len(harmonics),
    ).transpose(2, 0, 1, 3, 4, 5)
    phase_offsets = np.mod(
        _forcing_angle(discard_times, omega, phase), 2 * np.pi
    )
    return ForcedTransientStudyData(
        block_ids=blocks.block_ids,
        discard_times=discard_times,
        harmonics=harmonics,
        omega=omega,
        n_phase=n_phase,
        cycle_fourier=cycle_fourier,
        phase_offsets=phase_offsets,
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


def _realify(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    flat = values.reshape(values.shape[0], -1)
    if np.iscomplexobj(values):
        return np.concatenate((flat.real, flat.imag), axis=1)
    return np.asarray(flat, dtype=float)


def _paired_drift_summary(current, reference) -> dict:
    current = np.asarray(current)
    reference = np.asarray(reference)
    if current.shape != reference.shape or current.ndim < 2:
        raise ValueError("paired diagnostic arrays must match and retain block axis")
    if current.shape[0] < 2:
        raise ValueError("at least two blocks are required")
    current_real = _realify(current)
    reference_real = _realify(reference)
    delta = current_real - reference_real
    n_block = delta.shape[0]
    mean_delta = delta.mean(axis=0)
    delta_se = delta.std(axis=0, ddof=1) / np.sqrt(n_block)
    reference_mean = reference_real.mean(axis=0)
    reference_se = reference_real.std(axis=0, ddof=1) / np.sqrt(n_block)

    def rms(values):
        return float(np.sqrt(np.mean(np.asarray(values) ** 2)))

    mean_delta_rms = rms(mean_delta)
    paired_se_rms = rms(delta_se)
    reference_se_rms = rms(reference_se)
    return {
        "n_block": n_block,
        "value_shape": current.shape[1:],
        "mean_delta_realified": mean_delta,
        "paired_standard_error_realified": delta_se,
        "reference_mean_realified": reference_mean,
        "reference_standard_error_realified": reference_se,
        "mean_delta_rms": mean_delta_rms,
        "paired_standard_error_rms": paired_se_rms,
        "reference_standard_error_rms": reference_se_rms,
        "mean_delta_over_paired_se_rms": (
            mean_delta_rms / paired_se_rms if paired_se_rms > 0 else None
        ),
        "mean_delta_over_reference_se_rms": (
            mean_delta_rms / reference_se_rms if reference_se_rms > 0 else None
        ),
        "block_difference_rms": rms(delta),
    }


def analyze_forced_transient_study(data: ForcedTransientStudyData) -> dict:
    """Compare block-level Fourier contrasts across nested discard windows.

    The longest discard is only an internal reference. Reported standard errors
    quantify uncertainty of each *discard sensitivity*, not production response
    uncertainty and not a bound on residual transient bias.
    """
    coefficients = np.asarray(data.cycle_fourier)
    if not np.isfinite(data.omega) or data.omega <= 0:
        raise ValueError("study omega must be finite and positive")
    _positive_integer(data.n_phase, "study n_phase")
    if len(set(data.block_ids)) != len(data.block_ids):
        raise ValueError("study block_ids must be unique")
    if (
        np.asarray(data.phase_offsets).shape != np.asarray(data.discard_times).shape
        or not np.isfinite(data.phase_offsets).all()
    ):
        raise ValueError("phase_offsets must be finite and match discard_times")
    expected = (
        len(data.discard_times),
        len(data.block_ids),
        len(CONDITION_NAMES),
        3,
    )
    if coefficients.shape[:4] != expected or coefficients.ndim != 6:
        raise ValueError(
            "cycle_fourier must have axes discard, block, condition, state, cycle, harmonic"
        )
    if coefficients.shape[-1] != len(data.harmonics):
        raise ValueError("cycle_fourier harmonic axis must match harmonics")
    if not np.isfinite(coefficients).all():
        raise ValueError("cycle_fourier must be finite")
    condition_mean = coefficients.mean(axis=-2)
    positive = condition_mean[:, :, 0]
    negative = condition_mean[:, :, 1]
    unforced = condition_mean[:, :, 2]
    contrasts = {
        "odd": (positive - negative) / 2,
        "even": (positive + negative) / 2 - unforced,
    }
    target_keys = {"odd": {1}, "even": {0, 2}}
    by_contrast = {}
    for contrast_name, values in contrasts.items():
        harmonic_results = {}
        for harmonic_index, harmonic in enumerate(data.harmonics):
            discard_results = {}
            for discard_index, discard in enumerate(data.discard_times[:-1]):
                discard_results[str(float(discard))] = _paired_drift_summary(
                    values[discard_index, ..., harmonic_index],
                    values[-1, ..., harmonic_index],
                )
            adjacent_results = {}
            for discard_index in range(len(data.discard_times) - 1):
                pair_name = (
                    f"{float(data.discard_times[discard_index])}_to_"
                    f"{float(data.discard_times[discard_index + 1])}"
                )
                adjacent_results[pair_name] = _paired_drift_summary(
                    values[discard_index, ..., harmonic_index],
                    values[discard_index + 1, ..., harmonic_index],
                )
            harmonic = int(harmonic)
            if harmonic in target_keys[contrast_name]:
                role = "target"
            elif (harmonic % 2 == 1) == (contrast_name == "odd"):
                role = "higher_harmonic_diagnostic"
            else:
                role = "parity_forbidden_diagnostic"
            harmonic_results[str(harmonic)] = {
                "role": role,
                "comparisons_to_longest_discard": discard_results,
                "adjacent_discard_comparisons": adjacent_results,
            }
        by_contrast[contrast_name] = harmonic_results
    observation_cycle_count = coefficients.shape[-2]
    period = 2 * np.pi / float(data.omega)
    return {
        "interpretation": (
            "Exploratory paired block-level drift of raw Fourier contrasts. "
            "The longest discard is an internal reference, and paired standard "
            "errors describe uncertainty in the observed sensitivity rather "
            "than production sampling covariance or residual-bias bounds."
        ),
        "condition_order": CONDITION_NAMES,
        "state_coordinate_order": STATE_NAMES,
        "discard_times_physical": data.discard_times,
        "harmonics": data.harmonics,
        "observation_window": {
            "cycle_count": observation_cycle_count,
            "phase_points_per_cycle": int(data.n_phase),
            "forcing_period_physical": period,
            "represented_duration_physical": observation_cycle_count * period,
            "first_to_last_sample_span_physical": (
                observation_cycle_count - 1 / int(data.n_phase)
            )
            * period,
            "same_for_every_discard": True,
        },
        "contrast_diagnostics": by_contrast,
    }


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
    def run(*arguments):
        result = subprocess.run(
            arguments,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "worktree_dirty": bool(run("git", "status", "--short")),
    }


def persist_forced_transient_study(
    output_dir, data: ForcedTransientStudyData, derived: dict, config: dict, provenance: dict
) -> dict:
    """Persist raw Fourier summaries, derived diagnostics, and provenance."""
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
            discard_times=data.discard_times,
            harmonics=data.harmonics,
            omega=np.asarray(data.omega),
            n_phase=np.asarray(data.n_phase),
            cycle_fourier=data.cycle_fourier,
            phase_offsets=data.phase_offsets,
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
        "files": {
            "config_snapshot.json": f"sha256:{_file_sha256(config_path)}",
            "raw_fourier_summaries.npz": f"sha256:{_file_sha256(raw_path)}",
            "derived_diagnostics.json": f"sha256:{_file_sha256(derived_path)}",
        },
        "array_semantics": {
            "cycle_fourier": (
                "discard, block, condition, state, cycle, harmonic; complex "
                "Fourier coefficients using exp(-i*n*theta)"
            ),
            "condition_order": CONDITION_NAMES,
            "state_coordinate_order": STATE_NAMES,
            "discard_times": "physical Lorenz time from forcing onset",
            "omega": "forcing angular frequency in inverse Lorenz time",
            "n_phase": "uniform phase points per represented forcing cycle",
            "block_ids": "initial-state sampling replication unit",
        },
        "protocol_status": "provisional_method_validation_only",
        "protocol": config["protocol"],
        "initial_ensemble_generation": data.generation_metadata,
        "lorenz": config["lorenz"],
        "solver": config["solver"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def run_forced_transient_study(config_path) -> tuple[Path, dict, float]:
    """Run and persist one configured exploratory forced-transient study."""
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_forced_transient_study(config)
    derived = analyze_forced_transient_study(data)
    source_paths = [
        config_path,
        Path(__file__).resolve(),
        repo_root / "src/lorenz/core.py",
        repo_root / "src/lorenz/ensemble.py",
        repo_root / "src/lorenz/response.py",
        repo_root / "experiments/run_forced_transient_pilot.py",
    ]
    config_identifier = f"sha256:{_file_sha256(config_path)}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = (
        repo_root
        / config["output_root"]
        / f"{timestamp}_{config_identifier[7:19]}"
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
    manifest = persist_forced_transient_study(
        output_dir, data, derived, config, provenance
    )
    return output_dir, manifest, time.perf_counter() - start
