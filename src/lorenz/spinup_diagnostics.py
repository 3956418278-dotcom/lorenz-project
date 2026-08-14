"""Exploratory diagnostics for unforced initial-ensemble spinup."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
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

from .core import check_solution, lorenz_rhs
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks


STATE_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SpinupStudyData:
    """Raw proposals and nested unforced endpoints for each proposal family."""

    proposal_names: tuple[str, ...]
    burnin_times: np.ndarray
    block_ids: np.ndarray
    raw_proposals: np.ndarray
    endpoint_states: np.ndarray
    child_spawn_keys: np.ndarray
    generation_metadata: tuple[dict, ...]
    batch_ids: np.ndarray | None = None


def _checked_samples(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
        raise ValueError(f"{name} must have shape (sample, 3)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def energy_distance(first, second, scale=None) -> float:
    """Return the descriptive multivariate energy distance between samples."""
    first = _checked_samples(first, "first")
    second = _checked_samples(second, "second")
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("scale must be a finite positive vector with shape (3,)")
        first = first / scale
        second = second / scale
    def mean_pairwise_distance(left, right):
        differences = left[:, None, :] - right[None, :, :]
        return np.sqrt(np.sum(differences**2, axis=-1)).mean()

    value = (
        2.0 * mean_pairwise_distance(first, second)
        - mean_pairwise_distance(first, first)
        - mean_pairwise_distance(second, second)
    )
    return float(max(value, 0.0))


def _standard_error(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.full(values.shape[1:], np.nan)
    return values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _lorenz_vector_field(states: np.ndarray, lorenz: dict) -> np.ndarray:
    x, y, z = states.T
    return np.column_stack(
        (
            lorenz["sigma"] * (y - x),
            lorenz["rho"] * x - y - x * z,
            x * y - lorenz["beta"] * z,
        )
    )


def summarize_states(states, lorenz: dict, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Summarize one endpoint ensemble without making a convergence decision."""
    states = _checked_samples(states, "states")
    n_sample = states.shape[0]
    second_moment = states.T @ states / n_sample
    rms = np.sqrt(np.diag(second_moment))
    marginal_scale = states.std(axis=0, ddof=1) if n_sample > 1 else np.ones(3)
    marginal_scale = np.where(marginal_scale > 0, marginal_scale, 1.0)
    reflected = states * np.array([-1.0, -1.0, 1.0])
    field = _lorenz_vector_field(states, lorenz)
    field_rms = np.sqrt(np.mean(field**2, axis=0))
    normalized_field_mean = np.divide(
        field.mean(axis=0),
        field_rms,
        out=np.zeros(3),
        where=field_rms > 0,
    )
    q_values = np.quantile(states, quantiles, axis=0)
    return {
        "n_block": int(n_sample),
        "mean": states.mean(axis=0),
        "mean_standard_error": _standard_error(states),
        "covariance": np.cov(states, rowvar=False, ddof=1),
        "second_moment": second_moment,
        "quantiles": {
            f"{float(q):.2f}": q_values[index]
            for index, q in enumerate(quantiles)
        },
        "lobe_and_symmetry": {
            "x_positive_fraction": float(np.mean(states[:, 0] > 0)),
            "x_lobe_balance": float(2 * np.mean(states[:, 0] > 0) - 1),
            "xy_sign_agreement_fraction": float(
                np.mean(np.signbit(states[:, 0]) == np.signbit(states[:, 1]))
            ),
            "odd_mean_over_rms": np.divide(
                states.mean(axis=0)[:2],
                rms[:2],
                out=np.zeros(2),
                where=rms[:2] > 0,
            ),
            "reflection_energy_distance_standardized": energy_distance(
                states, reflected, marginal_scale
            ),
        },
        "vector_field_balance": {
            "mean": field.mean(axis=0),
            "mean_standard_error": _standard_error(field),
            "rms": field_rms,
            "mean_over_rms": normalized_field_mean,
        },
    }


def paired_endpoint_drift(earlier, later) -> dict:
    """Describe time-separated endpoints that share a raw-proposal block."""
    earlier = _checked_samples(earlier, "earlier")
    later = _checked_samples(later, "later")
    if earlier.shape != later.shape:
        raise ValueError("paired endpoint arrays must have equal shapes")
    delta = later - earlier
    distance = np.linalg.norm(delta, axis=1)
    return {
        "mean_delta": delta.mean(axis=0),
        "mean_delta_standard_error": _standard_error(delta),
        "euclidean_distance_median": float(np.median(distance)),
        "euclidean_distance_rms": float(np.sqrt(np.mean(distance**2))),
        "euclidean_distance_max": float(distance.max()),
        "x_lobe_switch_fraction": float(
            np.mean(np.signbit(earlier[:, 0]) != np.signbit(later[:, 0]))
        ),
    }


def _distribution_summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("diagnostic distribution must be a nonempty finite vector")
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values,
    }


def analyze_replicated_batches(data: SpinupStudyData, lorenz: dict, common_scale) -> dict:
    """Estimate diagnostic sampling scales from preassigned block batches."""
    if data.batch_ids is None:
        raise ValueError("batch_ids are required for replicated-batch analysis")
    batch_ids = np.asarray(data.batch_ids)
    endpoints = np.asarray(data.endpoint_states, dtype=float)
    if batch_ids.shape != endpoints.shape[:1] + endpoints.shape[2:3]:
        raise ValueError("batch_ids must have shape (proposal, block)")
    batch_labels = tuple(np.unique(batch_ids[0]).tolist())
    if len(batch_labels) < 2:
        raise ValueError("at least two batches are required")
    masks = []
    for proposal_index in range(len(data.proposal_names)):
        if tuple(np.unique(batch_ids[proposal_index]).tolist()) != batch_labels:
            raise ValueError("proposal families must share the same batch labels")
        proposal_masks = []
        sizes = []
        for label in batch_labels:
            mask = batch_ids[proposal_index] == label
            proposal_masks.append(mask)
            sizes.append(int(mask.sum()))
        if len(set(sizes)) != 1 or sizes[0] < 2:
            raise ValueError("all replicated batches must have equal size of at least two")
        masks.append(proposal_masks)

    times = np.asarray(data.burnin_times, dtype=float)
    per_proposal = {}
    for proposal_index, proposal_name in enumerate(data.proposal_names):
        proposal_result = {}
        for time_index, burnin_time in enumerate(times):
            drift_values = []
            same_time_reference = []
            batch_means = []
            batch_lobe_balances = []
            batch_field_means = []
            for batch_index, mask in enumerate(masks[proposal_index]):
                current = endpoints[proposal_index, time_index, mask]
                longest = endpoints[proposal_index, -1, mask]
                drift_values.append(energy_distance(current, longest, common_scale))
                batch_means.append(current.mean(axis=0))
                batch_lobe_balances.append(2 * np.mean(current[:, 0] > 0) - 1)
                field = _lorenz_vector_field(current, lorenz)
                field_rms = np.sqrt(np.mean(field**2, axis=0))
                batch_field_means.append(
                    np.divide(
                        field.mean(axis=0),
                        field_rms,
                        out=np.zeros(3),
                        where=field_rms > 0,
                    )
                )
                for other_index in range(batch_index):
                    other = endpoints[
                        proposal_index, time_index, masks[proposal_index][other_index]
                    ]
                    same_time_reference.append(energy_distance(current, other, common_scale))
            proposal_result[str(float(burnin_time))] = {
                "batch_energy_to_longest_burnin_standardized": _distribution_summary(
                    drift_values
                ),
                "same_time_between_batch_energy_standardized": _distribution_summary(
                    same_time_reference
                ),
                "batch_state_means": np.asarray(batch_means),
                "batch_x_lobe_balances": np.asarray(batch_lobe_balances),
                "batch_vector_field_mean_over_rms": np.asarray(batch_field_means),
            }
        per_proposal[proposal_name] = proposal_result

    proposal_sensitivity = {}
    for first_index, second_index in combinations(range(len(data.proposal_names)), 2):
        pair_name = f"{data.proposal_names[first_index]}__vs__{data.proposal_names[second_index]}"
        by_time = {}
        for time_index, burnin_time in enumerate(times):
            sensitivity = []
            reference = []
            for batch_index in range(len(batch_labels)):
                first = endpoints[first_index, time_index, masks[first_index][batch_index]]
                second = endpoints[second_index, time_index, masks[second_index][batch_index]]
                sensitivity.append(energy_distance(first, second, common_scale))
            for proposal_index in (first_index, second_index):
                for left, right in combinations(range(len(batch_labels)), 2):
                    reference.append(
                        energy_distance(
                            endpoints[
                                proposal_index,
                                time_index,
                                masks[proposal_index][left],
                            ],
                            endpoints[
                                proposal_index,
                                time_index,
                                masks[proposal_index][right],
                            ],
                            common_scale,
                        )
                    )
            by_time[str(float(burnin_time))] = {
                "cross_proposal_batch_energy_standardized": _distribution_summary(
                    sensitivity
                ),
                "same_proposal_between_batch_reference_standardized": _distribution_summary(
                    reference
                ),
            }
        proposal_sensitivity[pair_name] = by_time

    result = {
        "interpretation": (
            "Batch distributions quantify diagnostic sampling variability. "
            "They are descriptive and do not define a formal acceptance tolerance."
        ),
        "batch_labels": batch_labels,
        "blocks_per_batch": int(masks[0][0].sum()),
        "proposals": per_proposal,
        "alternate_proposal_sensitivity": proposal_sensitivity,
    }
    return result


def analyze_spinup_study(data: SpinupStudyData, lorenz: dict) -> dict:
    """Compute endpoint, drift, and alternate-proposal diagnostics."""
    endpoints = np.asarray(data.endpoint_states, dtype=float)
    times = np.asarray(data.burnin_times, dtype=float)
    expected_shape = (
        len(data.proposal_names),
        len(times),
        data.block_ids.shape[1],
        3,
    )
    if endpoints.shape != expected_shape or not np.isfinite(endpoints).all():
        raise ValueError(f"endpoint_states must have shape {expected_shape} and be finite")
    if len(times) < 2 or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("burnin_times must be strictly increasing and positive")

    pooled_final = endpoints[:, -1].reshape(-1, 3)
    common_scale = pooled_final.std(axis=0, ddof=1)
    if np.any(common_scale <= 0):
        raise ValueError("final pooled endpoint scale must be positive")

    proposal_results = {}
    for proposal_index, proposal_name in enumerate(data.proposal_names):
        states_by_time = endpoints[proposal_index]
        endpoint_summaries = {}
        comparisons_to_final = {}
        adjacent_drifts = {}
        final_states = states_by_time[-1]
        for time_index, burnin_time in enumerate(times):
            label = str(float(burnin_time))
            states = states_by_time[time_index]
            summary = summarize_states(states, lorenz)
            summary["alternating_split_half_energy_reference"] = {
                "interpretation": (
                    "Descriptive finite-sample scale from alternating ordered "
                    "blocks; each half has half the endpoint sample size."
                ),
                "energy_distance_physical": energy_distance(states[::2], states[1::2]),
                "energy_distance_common_standardized": energy_distance(
                    states[::2], states[1::2], common_scale
                ),
            }
            endpoint_summaries[label] = summary
            comparisons_to_final[label] = {
                "energy_distance_physical": energy_distance(states, final_states),
                "energy_distance_common_standardized": energy_distance(
                    states, final_states, common_scale
                ),
                "paired_endpoint_drift": paired_endpoint_drift(states, final_states),
            }
            if time_index:
                previous_label = str(float(times[time_index - 1]))
                adjacent_drifts[f"{previous_label}_to_{label}"] = {
                    "energy_distance_physical": energy_distance(
                        states_by_time[time_index - 1], states
                    ),
                    "energy_distance_common_standardized": energy_distance(
                        states_by_time[time_index - 1], states, common_scale
                    ),
                    "paired_endpoint_drift": paired_endpoint_drift(
                        states_by_time[time_index - 1], states
                    ),
                }
        proposal_results[proposal_name] = {
            "endpoint_summaries": endpoint_summaries,
            "comparisons_to_longest_burnin": comparisons_to_final,
            "adjacent_burnin_drifts": adjacent_drifts,
        }

    proposal_sensitivity = {}
    for first_index, second_index in combinations(range(len(data.proposal_names)), 2):
        pair_name = f"{data.proposal_names[first_index]}__vs__{data.proposal_names[second_index]}"
        by_time = {}
        for time_index, burnin_time in enumerate(times):
            first = endpoints[first_index, time_index]
            second = endpoints[second_index, time_index]
            by_time[str(float(burnin_time))] = {
                "energy_distance_physical": energy_distance(first, second),
                "energy_distance_common_standardized": energy_distance(
                    first, second, common_scale
                ),
                "mean_difference_second_minus_first": second.mean(axis=0)
                - first.mean(axis=0),
            }
        proposal_sensitivity[pair_name] = by_time

    result = {
        "interpretation": (
            "Exploratory distributional drift only. The longest burn-in endpoint "
            "is an internal reference, not a known sample from mu0, and no "
            "failure-to-reject criterion is used."
        ),
        "state_coordinate_order": STATE_NAMES,
        "common_standardization_scale_from_pooled_longest_burnin": common_scale,
        "proposals": proposal_results,
        "alternate_proposal_sensitivity": proposal_sensitivity,
    }
    if data.batch_ids is not None:
        result["replicated_batch_diagnostics"] = analyze_replicated_batches(
            data, lorenz, common_scale
        )
    return result


def _integrate_nested_endpoints(arguments):
    raw_state, burnin_times, cfg = arguments
    burnin_times = np.asarray(burnin_times, dtype=float)
    sol = solve_ivp(
        lambda t, state: lorenz_rhs(t, state, cfg),
        (0.0, float(burnin_times[-1])),
        np.asarray(raw_state, dtype=float),
        t_eval=burnin_times,
        **cfg["solver"],
    )
    check_solution(sol, (3, len(burnin_times)))
    return sol.y.T


def generate_spinup_study(config: dict) -> SpinupStudyData:
    """Generate proposals once and integrate each block to a nested time ladder."""
    burnin_times = np.asarray(config["burnin_times"], dtype=float)
    if len(burnin_times) < 2 or np.any(burnin_times <= 0) or np.any(np.diff(burnin_times) <= 0):
        raise ValueError("burnin_times must be strictly increasing and positive")
    block_count = int(config["block_count"])
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    workers = int(config.get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be positive")
    batch_count = int(config.get("batch_count", 0))
    if batch_count < 0 or (batch_count and block_count % batch_count):
        raise ValueError("batch_count must be nonnegative and divide block_count")
    cfg = {"lorenz": dict(config["lorenz"]), "solver": dict(config["solver"])}

    proposal_names = []
    block_ids_by_proposal = []
    raw_by_proposal = []
    keys_by_proposal = []
    metadata = []
    all_ids = set()
    for proposal_config in config["proposals"]:
        name = str(proposal_config["name"])
        if name in proposal_names:
            raise ValueError("proposal names must be unique")
        first_id = int(proposal_config["block_id_start"])
        block_ids = tuple(range(first_id, first_id + block_count))
        if all_ids.intersection(block_ids):
            raise ValueError("block IDs must be unique across proposal families")
        all_ids.update(block_ids)
        proposal = SymmetricXYUniformProposal(
            x_half_width=proposal_config["x_half_width"],
            y_half_width=proposal_config["y_half_width"],
            z_bounds=tuple(proposal_config["z_bounds"]),
        )
        blocks = generate_initial_state_blocks(
            block_ids,
            config["root_entropy"],
            proposal,
            spinup_time=0.0,
            cfg=cfg,
        )
        proposal_names.append(name)
        block_ids_by_proposal.append(blocks.block_ids)
        raw_by_proposal.append(blocks.raw_proposals)
        keys_by_proposal.append(blocks.child_spawn_keys)
        metadata.append(
            {
                "proposal_name": name,
                "proposal": {
                    "x_half_width": proposal.x_half_width,
                    "y_half_width": proposal.y_half_width,
                    "z_bounds": proposal.z_bounds,
                },
                "root_entropy": blocks.root_entropy,
                "root_spawn_key": blocks.root_spawn_key,
                "bit_generator": blocks.bit_generator,
                "block_ids": blocks.block_ids,
                "child_spawn_keys": blocks.child_spawn_keys,
            }
        )

    raw_array = np.stack(raw_by_proposal)
    tasks = [
        (raw_array[p, b], burnin_times, cfg)
        for p in range(len(proposal_names))
        for b in range(block_count)
    ]
    if workers == 1:
        integrated = list(map(_integrate_nested_endpoints, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            integrated = list(executor.map(_integrate_nested_endpoints, tasks))
    endpoints = np.asarray(integrated).reshape(
        len(proposal_names), block_count, len(burnin_times), 3
    ).transpose(0, 2, 1, 3)
    return SpinupStudyData(
        proposal_names=tuple(proposal_names),
        burnin_times=burnin_times,
        block_ids=np.asarray(block_ids_by_proposal, dtype=np.uint32),
        raw_proposals=raw_array,
        endpoint_states=endpoints,
        child_spawn_keys=np.asarray(keys_by_proposal, dtype=np.uint32),
        generation_metadata=tuple(metadata),
        batch_ids=(
            np.tile(
                np.repeat(np.arange(batch_count), block_count // batch_count),
                (len(proposal_names), 1),
            )
            if batch_count
            else None
        ),
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


def persist_spinup_study(
    output_dir,
    data: SpinupStudyData,
    derived: dict,
    config: dict,
    provenance: dict,
) -> dict:
    """Persist raw arrays, derived diagnostics, config, and a hashed manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_endpoints.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    _write_json_atomic(config_path, config)
    temporary_raw = raw_path.with_suffix(".npz.tmp")
    arrays = {
        "proposal_names": np.asarray(data.proposal_names),
        "burnin_times": data.burnin_times,
        "block_ids": data.block_ids,
        "raw_proposals": data.raw_proposals,
        "endpoint_states": data.endpoint_states,
        "child_spawn_keys": data.child_spawn_keys,
    }
    if data.batch_ids is not None:
        arrays["batch_ids"] = data.batch_ids
    with temporary_raw.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary_raw, raw_path)
    _write_json_atomic(derived_path, derived)
    manifest = {
        "schema_version": 1,
        "classification": "exploratory",
        "study_id": config["study_id"],
        "files": {
            "config_snapshot.json": f"sha256:{_file_sha256(config_path)}",
            "raw_endpoints.npz": f"sha256:{_file_sha256(raw_path)}",
            "derived_diagnostics.json": f"sha256:{_file_sha256(derived_path)}",
        },
        "array_semantics": {
            "block_ids": "proposal, block",
            "raw_proposals": "proposal, block, state",
            "endpoint_states": "proposal, burnin_time, block, state",
            "child_spawn_keys": "proposal, block, spawn_key_component",
            "batch_ids": "proposal, block; present for replicated-batch studies",
            "state_coordinate_order": STATE_NAMES,
        },
        "generation": data.generation_metadata,
        "lorenz": config["lorenz"],
        "solver": config["solver"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


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


def run_spinup_study(config_path) -> tuple[Path, dict, float]:
    """Run and persist one configured exploratory spinup study."""
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_spinup_study(config)
    derived = analyze_spinup_study(data, config["lorenz"])
    source_paths = [
        config_path,
        Path(__file__).resolve(),
        repo_root / "src/lorenz/core.py",
        repo_root / "src/lorenz/ensemble.py",
        repo_root / "experiments/run_unforced_spinup_pilot.py",
    ]
    config_identifier = f"sha256:{_file_sha256(config_path)}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = repo_root / config["output_root"] / f"{timestamp}_{config_identifier[7:19]}"
    runtime = time.perf_counter() - start
    provenance = {
        "config_identifier": config_identifier,
        "code_identifiers": {
            str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path): f"sha256:{_file_sha256(path)}"
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
    manifest = persist_spinup_study(
        output_dir, data, derived, config, provenance
    )
    return output_dir, manifest, time.perf_counter() - start
