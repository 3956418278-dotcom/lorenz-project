"""Block-level numerical convergence diagnostics for forced experiments.

This module compares observation windows, phase grids, and solver profiles.
Its paired standard errors quantify sensitivity-study resolution only; they are
not production response covariance estimates or residual-bias bounds.
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
from scipy.integrate import solve_ivp

from .core import _forcing_angle, check_solution, lorenz_rhs
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from .forced_transient import CONDITION_NAMES, STATE_NAMES
from .response import phase_fourier


@dataclass(frozen=True)
class NumericalConvergenceStudyData:
    """Cycle Fourier summaries across solver and phase-grid variants.

    ``cycle_fourier`` axes are ``solver, phase_variant, block, condition,
    state, cycle, harmonic``. Each phase resolution has an unshifted and a
    half-cell-shifted variant sampled from one common master grid.
    """

    block_ids: tuple[int, ...]
    solver_names: tuple[str, ...]
    reference_solver: str
    phase_variant_names: tuple[str, ...]
    phase_resolutions: np.ndarray
    phase_shifted: np.ndarray
    observation_cycles: np.ndarray
    harmonics: np.ndarray
    omega: float
    discard_time: float
    master_phase_count: int
    cycle_fourier: np.ndarray
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


def _checked_config(config: dict) -> dict:
    protocol = config["protocol"]
    omega = float(protocol["omega"])
    phase = float(protocol.get("phase", 0.0))
    strength = float(protocol["strength"])
    direction = np.asarray(protocol["direction"], dtype=float)
    discard_time = float(config["discard_time"])
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("direction must be a finite nonzero vector with shape (3,)")
    if not np.isfinite(discard_time) or discard_time < 0:
        raise ValueError("discard_time must be finite and nonnegative")

    observation_cycles = np.asarray(config["observation_cycles"])
    if (
        observation_cycles.ndim != 1
        or len(observation_cycles) < 2
        or not np.issubdtype(observation_cycles.dtype, np.integer)
        or np.any(observation_cycles <= 0)
        or np.any(np.diff(observation_cycles) <= 0)
    ):
        raise ValueError("observation_cycles must be increasing positive integers")

    phase_levels = np.asarray(config["phase_resolutions"])
    if (
        phase_levels.ndim != 1
        or len(phase_levels) < 2
        or not np.issubdtype(phase_levels.dtype, np.integer)
        or np.any(phase_levels <= 0)
        or np.any(np.diff(phase_levels) <= 0)
    ):
        raise ValueError("phase_resolutions must be increasing positive integers")
    finest = int(phase_levels[-1])
    if any(finest % int(level) for level in phase_levels):
        raise ValueError("phase resolutions must be nested divisors of the finest grid")
    master_phase_count = 2 * finest
    if any((master_phase_count // int(level)) % 2 for level in phase_levels):
        raise ValueError("each phase grid must support a half-cell shifted subgrid")

    harmonics = np.asarray(config["harmonics"])
    if (
        harmonics.ndim != 1
        or len(harmonics) < 1
        or not np.issubdtype(harmonics.dtype, np.integer)
        or len(np.unique(harmonics)) != len(harmonics)
    ):
        raise ValueError("harmonics must be a vector of unique integers")
    if not {0, 1, 2}.issubset(set(int(value) for value in harmonics)):
        raise ValueError("harmonics must include 0, 1, and 2")
    if np.max(np.abs(harmonics)) * 2 >= int(phase_levels[0]):
        raise ValueError("coarsest phase grid must exceed twice the largest harmonic")

    solvers = config["solver_profiles"]
    if not isinstance(solvers, dict) or len(solvers) < 2:
        raise ValueError("solver_profiles must contain at least two named profiles")
    reference_solver = config["reference_solver"]
    if reference_solver not in solvers:
        raise ValueError("reference_solver must name a configured solver profile")
    for name, solver in solvers.items():
        if not isinstance(name, str) or not name or not isinstance(solver, dict):
            raise ValueError("solver profiles require nonempty names and mappings")

    return {
        "omega": omega,
        "phase": phase,
        "strength": strength,
        "direction": direction,
        "discard_time": discard_time,
        "observation_cycles": observation_cycles.astype(int),
        "phase_levels": phase_levels.astype(int),
        "master_phase_count": master_phase_count,
        "harmonics": harmonics.astype(int),
        "solver_names": tuple(solvers),
        "reference_solver": reference_solver,
    }


def _integrate_phase_variants(arguments):
    (
        initial_state,
        forcing_vector,
        omega,
        phase,
        discard_time,
        n_cycle,
        master_phase_count,
        phase_levels,
        harmonics,
        lorenz,
        solver,
    ) = arguments
    period = 2 * np.pi / omega
    cycle = np.arange(n_cycle, dtype=float)
    phase_index = np.arange(master_phase_count, dtype=float)
    sample_times = discard_time + (
        cycle[:, None] + phase_index[None, :] / master_phase_count
    ) * period
    flat_times = sample_times.ravel()
    cfg = {"lorenz": lorenz, "solver": solver}
    sol = solve_ivp(
        lambda t, state: lorenz_rhs(t, state, cfg, forcing_vector, omega, phase),
        (0.0, float(flat_times[-1])),
        np.asarray(initial_state, dtype=float),
        t_eval=flat_times,
        **solver,
    )
    check_solution(sol, (3, n_cycle * master_phase_count))
    values = sol.y.reshape(3, n_cycle, master_phase_count)
    base_offset = np.mod(_forcing_angle(discard_time, omega, phase), 2 * np.pi)
    variants = []
    for resolution in phase_levels:
        stride = master_phase_count // int(resolution)
        for shifted in (False, True):
            start = stride // 2 if shifted else 0
            indices = start + stride * np.arange(int(resolution))
            phase_offset = base_offset + 2 * np.pi * start / master_phase_count
            variants.append(
                phase_fourier(values[..., indices], harmonics, phase_offset)
            )
    return np.asarray(variants)


def generate_numerical_convergence_study(config: dict) -> NumericalConvergenceStudyData:
    """Generate one configurable, explicitly exploratory convergence study."""
    checked = _checked_config(config)
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
    generation_cfg = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["initial_ensemble"]["solver"]),
    }
    blocks = generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        proposal,
        float(initial["spinup_time"]),
        generation_cfg,
    )
    forcing_vectors = (
        checked["strength"] * checked["direction"],
        -checked["strength"] * checked["direction"],
        np.zeros(3),
    )
    solvers = config["solver_profiles"]
    tasks = [
        (
            blocks.final_states[block_index],
            forcing_vector,
            checked["omega"],
            checked["phase"],
            checked["discard_time"],
            int(checked["observation_cycles"][-1]),
            checked["master_phase_count"],
            checked["phase_levels"],
            checked["harmonics"],
            dict(config["lorenz"]),
            dict(solvers[solver_name]),
        )
        for solver_name in checked["solver_names"]
        for block_index in range(block_count)
        for forcing_vector in forcing_vectors
    ]
    if workers == 1:
        integrated = list(map(_integrate_phase_variants, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            integrated = list(executor.map(_integrate_phase_variants, tasks))
    variant_names = tuple(
        f"n{int(level)}_{shift}"
        for level in checked["phase_levels"]
        for shift in ("base", "shifted")
    )
    phase_resolutions = np.repeat(checked["phase_levels"], 2)
    phase_shifted = np.tile(np.array([False, True]), len(checked["phase_levels"]))
    cycle_fourier = np.asarray(integrated).reshape(
        len(checked["solver_names"]),
        block_count,
        len(CONDITION_NAMES),
        len(variant_names),
        3,
        int(checked["observation_cycles"][-1]),
        len(checked["harmonics"]),
    ).transpose(0, 3, 1, 2, 4, 5, 6)
    return NumericalConvergenceStudyData(
        block_ids=blocks.block_ids,
        solver_names=checked["solver_names"],
        reference_solver=checked["reference_solver"],
        phase_variant_names=variant_names,
        phase_resolutions=phase_resolutions,
        phase_shifted=phase_shifted,
        observation_cycles=checked["observation_cycles"],
        harmonics=checked["harmonics"],
        omega=checked["omega"],
        discard_time=checked["discard_time"],
        master_phase_count=checked["master_phase_count"],
        cycle_fourier=cycle_fourier,
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
            "solver": generation_cfg["solver"],
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
    if current.shape != reference.shape or current.ndim < 2 or current.shape[0] < 2:
        raise ValueError("paired arrays must match and retain at least two blocks")
    delta = _realify(current) - _realify(reference)
    current_real = _realify(current)
    reference_real = _realify(reference)
    n_block = delta.shape[0]
    mean_delta = delta.mean(axis=0)
    delta_se = delta.std(axis=0, ddof=1) / np.sqrt(n_block)
    reference_mean = reference_real.mean(axis=0)
    reference_se = reference_real.std(axis=0, ddof=1) / np.sqrt(n_block)

    def rms(values):
        return float(np.sqrt(np.mean(np.asarray(values) ** 2)))

    mean_delta_rms = rms(mean_delta)
    paired_se_rms = rms(delta_se)
    return {
        "n_block": n_block,
        "value_shape": current.shape[1:],
        "current_mean_realified": current_real.mean(axis=0),
        "reference_mean_realified": reference_mean,
        "reference_standard_error_realified": reference_se,
        "mean_delta_realified": mean_delta,
        "paired_standard_error_realified": delta_se,
        "mean_delta_rms": mean_delta_rms,
        "paired_standard_error_rms": paired_se_rms,
        "mean_delta_over_paired_se_rms": (
            mean_delta_rms / paired_se_rms if paired_se_rms > 0 else None
        ),
        "block_difference_rms": rms(delta),
    }


def _contrasts(cycle_fourier: np.ndarray) -> dict[str, np.ndarray]:
    positive = cycle_fourier[..., 0, :, :, :]
    negative = cycle_fourier[..., 1, :, :, :]
    unforced = cycle_fourier[..., 2, :, :, :]
    return {
        "odd": (positive - negative) / 2,
        "even": (positive + negative) / 2 - unforced,
    }


def _role(contrast_name: str, harmonic: int) -> str:
    if (contrast_name == "odd" and harmonic == 1) or (
        contrast_name == "even" and harmonic in (0, 2)
    ):
        return "target"
    if (harmonic % 2 == 1) == (contrast_name == "odd"):
        return "higher_harmonic_diagnostic"
    return "parity_forbidden_diagnostic"


def _all_harmonic_comparisons(
    current: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    harmonics: np.ndarray,
) -> dict:
    result = {}
    for contrast_name in ("odd", "even"):
        result[contrast_name] = {}
        for harmonic_index, harmonic in enumerate(harmonics):
            result[contrast_name][str(int(harmonic))] = {
                "role": _role(contrast_name, int(harmonic)),
                "drift": _paired_drift_summary(
                    current[contrast_name][..., harmonic_index],
                    reference[contrast_name][..., harmonic_index],
                ),
            }
    return result


def analyze_numerical_convergence_study(data: NumericalConvergenceStudyData) -> dict:
    """Analyze three separate block-level numerical-sensitivity checks."""
    values = np.asarray(data.cycle_fourier)
    expected = (
        len(data.solver_names),
        len(data.phase_variant_names),
        len(data.block_ids),
        len(CONDITION_NAMES),
        3,
        int(data.observation_cycles[-1]),
        len(data.harmonics),
    )
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("cycle_fourier shape or values are invalid")
    if data.reference_solver not in data.solver_names:
        raise ValueError("reference_solver is not present")
    if len(set(data.block_ids)) != len(data.block_ids):
        raise ValueError("block_ids must be unique")
    reference_solver_index = data.solver_names.index(data.reference_solver)
    finest = int(np.max(data.phase_resolutions))
    finest_base_index = next(
        index
        for index, (resolution, shifted) in enumerate(
            zip(data.phase_resolutions, data.phase_shifted)
        )
        if int(resolution) == finest and not shifted
    )

    observation = {}
    reference_cycles = values[
        reference_solver_index, finest_base_index, ..., : int(data.observation_cycles[-1]), :
    ]
    reference_contrasts = {
        name: contrast.mean(axis=-2)
        for name, contrast in _contrasts(reference_cycles).items()
    }
    for n_cycle in data.observation_cycles[:-1]:
        candidate = values[
            reference_solver_index, finest_base_index, ..., : int(n_cycle), :
        ]
        candidate_contrasts = {
            name: contrast.mean(axis=-2)
            for name, contrast in _contrasts(candidate).items()
        }
        observation[str(int(n_cycle))] = _all_harmonic_comparisons(
            candidate_contrasts, reference_contrasts, data.harmonics
        )

    phase_ref_values = values[
        reference_solver_index, finest_base_index, ..., : int(data.observation_cycles[-1]), :
    ]
    phase_ref = {
        name: contrast.mean(axis=-2)
        for name, contrast in _contrasts(phase_ref_values).items()
    }
    refinement = {}
    shifted_checks = {}
    for resolution in np.unique(data.phase_resolutions):
        base_index = next(
            index
            for index, (level, shifted) in enumerate(
                zip(data.phase_resolutions, data.phase_shifted)
            )
            if int(level) == int(resolution) and not shifted
        )
        shifted_index = next(
            index
            for index, (level, shifted) in enumerate(
                zip(data.phase_resolutions, data.phase_shifted)
            )
            if int(level) == int(resolution) and shifted
        )
        base = {
            name: contrast.mean(axis=-2)
            for name, contrast in _contrasts(
                values[reference_solver_index, base_index]
            ).items()
        }
        shifted = {
            name: contrast.mean(axis=-2)
            for name, contrast in _contrasts(
                values[reference_solver_index, shifted_index]
            ).items()
        }
        refinement[str(int(resolution))] = _all_harmonic_comparisons(
            base, phase_ref, data.harmonics
        )
        shifted_checks[str(int(resolution))] = _all_harmonic_comparisons(
            base, shifted, data.harmonics
        )

    solver_checks = {}
    for solver_index, solver_name in enumerate(data.solver_names):
        if solver_name == data.reference_solver:
            continue
        candidate = {
            name: contrast.mean(axis=-2)
            for name, contrast in _contrasts(
                values[solver_index, finest_base_index]
            ).items()
        }
        solver_checks[solver_name] = _all_harmonic_comparisons(
            candidate, phase_ref, data.harmonics
        )

    period = 2 * np.pi / data.omega
    return {
        "interpretation": (
            "Exploratory block-level sensitivity checks for observation-window, "
            "phase-grid, and solver choices. Paired standard errors quantify the "
            "resolution of these checks; they are neither production response "
            "covariance nor formal bounds on residual numerical/transient bias."
        ),
        "protocol_status": "provisional_method_validation_only",
        "condition_order": CONDITION_NAMES,
        "state_coordinate_order": STATE_NAMES,
        "reference_settings": {
            "solver": data.reference_solver,
            "phase_points_per_cycle": finest,
            "observation_cycles": int(data.observation_cycles[-1]),
            "observation_duration_physical": float(data.observation_cycles[-1] * period),
            "discard_time_physical": float(data.discard_time),
        },
        "observation_window_semantics": {
            "forcing_period_physical": period,
            "cycle_counts": data.observation_cycles,
            "represented_durations_physical": data.observation_cycles * period,
            "comparison": "nested prefixes with a common start after discard",
        },
        "phase_grid_semantics": {
            "resolutions": np.unique(data.phase_resolutions),
            "master_phase_count": int(data.master_phase_count),
            "refinement": "nested base subgrids from one master trajectory",
            "shift_check": "half-cell shifted subgrid at each resolution",
        },
        "observation_window_comparisons_to_longest": observation,
        "phase_refinement_comparisons_to_finest": refinement,
        "phase_shifted_subgrid_comparisons": shifted_checks,
        "solver_profile_comparisons_to_reference": solver_checks,
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
            arguments, cwd=repo_root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "worktree_dirty": bool(run("git", "status", "--short")),
    }


def persist_numerical_convergence_study(
    output_dir,
    data: NumericalConvergenceStudyData,
    derived: dict,
    config: dict,
    provenance: dict,
) -> dict:
    """Persist raw cycle summaries, derived diagnostics, and provenance."""
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
            solver_names=np.asarray(data.solver_names),
            reference_solver=np.asarray(data.reference_solver),
            phase_variant_names=np.asarray(data.phase_variant_names),
            phase_resolutions=data.phase_resolutions,
            phase_shifted=data.phase_shifted,
            observation_cycles=data.observation_cycles,
            harmonics=data.harmonics,
            omega=np.asarray(data.omega),
            discard_time=np.asarray(data.discard_time),
            master_phase_count=np.asarray(data.master_phase_count),
            cycle_fourier=data.cycle_fourier,
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
                "solver, phase_variant, block, condition, state, cycle, harmonic; "
                "complex coefficients using exp(-i*n*theta)"
            ),
            "condition_order": CONDITION_NAMES,
            "state_coordinate_order": STATE_NAMES,
            "block_ids": "initial-state sampling replication unit",
            "observation_cycles": "nested cycle-prefix lengths",
            "discard_time": "physical Lorenz time from forcing onset",
        },
        "protocol_status": "provisional_method_validation_only",
        "protocol": config["protocol"],
        "initial_ensemble_generation": data.generation_metadata,
        "lorenz": config["lorenz"],
        "solver_profiles": config["solver_profiles"],
        "reference_solver": data.reference_solver,
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def run_numerical_convergence_study(config_path) -> tuple[Path, dict, float]:
    """Run and persist a configured exploratory numerical convergence study."""
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_numerical_convergence_study(config)
    derived = analyze_numerical_convergence_study(data)
    source_paths = [
        config_path,
        Path(__file__).resolve(),
        repo_root / "src/lorenz/core.py",
        repo_root / "src/lorenz/ensemble.py",
        repo_root / "src/lorenz/response.py",
        repo_root / "experiments/run_numerical_convergence_pilot.py",
    ]
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
    manifest = persist_numerical_convergence_study(
        output_dir, data, derived, config, provenance
    )
    return output_dir, manifest, time.perf_counter() - start
