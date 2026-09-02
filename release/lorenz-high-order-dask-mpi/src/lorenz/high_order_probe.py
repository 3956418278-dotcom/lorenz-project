"""One-frequency harmonic probe with paired forcing directions and shared unforced.

Goal of the probe (NOT a perturbative-tensor measurement): probe the 1..5
omega response across a configured direction design and identify
which (direction, output, harmonic) cells show a reliable signed response.
The unforced trajectory is integrated once per block and shared by every
direction and strength.  An optional extension mode can reuse a completed
base artifact and integrate only newly requested paired forcing conditions;
the new artifact stores self-contained combined chunks without changing the
base artifact.  All outputs (x, y, z) and harmonics n=0..5 are
retained as block-level complex coefficients and analyzed under the signed
cos/sin convention (Re = cos component, Im = -(sin component)) with
Hotelling T^2 joint tests; DC is a scalar.  High amplitudes (h>=8) are
discovery settings, not perturbative coefficients.

The final result is the response map
    direction x output x harmonic x strength x {cos, sin}
plus the same objects' statistics.

Run with --dry-run to validate the configuration and report the expected
task counts without integrating anything.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from .artifacts import (
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    write_json_atomic,
    write_npz_atomic,
)
from .retention import (
    chunk_file_name,
    chunk_ranges,
    condition_axis_permutation,
    load_spectrum_conditions,
    mirrored_phase_pair_conditions,
    paired_condition_indices,
    paired_direction_strength_vectors,
    paired_condition_vectors,
    parse_retention_policy,
    write_block_level_chunk,
)
from .response import harmonic_order_contrast, paired_order_contrasts
from .response_plots import (
    figure_probe_detection_overview,
    figure_probe_signed_responses,
    figure_probe_spectrum_noise,
    save_figure,
)
from .statistics import benjamini_hochberg, cos_sin_statistics
from .direction_design import direction_identity
from .strength_study import (
    STATE_NAMES,
    CycleFourierSampling,
    validate_strengths,
    integrate_cycle_fourier_conditions,
    integrate_phase_and_dense_conditions,
    minimum_duration_cycle_count,
)
from .strength_series import power_series_operator
from .ensemble import generate_configured_initial_state_blocks
from .parallel import close_execution, initialize_execution

BLOCK_LEVEL_DIRECTORY = "block_level"
MIXED_BOOTSTRAP_REPLICATES = 300
MIXED_BOOTSTRAP_SEED = 20260902
UNFORCED_ALIGNMENT_RTOL = 1e-10
UNFORCED_ALIGNMENT_ATOL = 1e-11
EXTENSION_VALIDATION_ARRAYS = {
    "mean": "extension_unforced_condition_means_max_abs_difference",
    "spectrum": "extension_unforced_spectrum_max_abs_difference",
    "cycle": "extension_unforced_cycle_fourier_max_abs_difference",
    "counts": "extension_unforced_spectrum_segment_counts_exact",
}


def validate_probe_config(config: dict) -> dict:
    """Validate the probe-specific configuration surface."""
    omega = float(config["omega"])
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    block_count = int(config["block_count"])
    if isinstance(block_count, bool) or block_count < 3:
        raise ValueError("block_count must be at least three")
    directions = [np.asarray(value, dtype=float) for value in config["protocol"]["directions"]]
    if not directions:
        raise ValueError("the probe requires at least one forcing direction")
    for direction in directions:
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
            raise ValueError("each direction must be a finite nonzero vector")
    direction_labels = [direction_identity(direction)[0] for direction in directions]
    mixed_config = config.get("mixed_phase_pairs")
    configured_by_direction = config.get("strengths_by_direction")
    if mixed_config is not None and configured_by_direction is not None:
        raise ValueError("mixed_phase_pairs and strengths_by_direction are mutually exclusive")
    condition_plan = None
    if mixed_config is not None:
        if not isinstance(mixed_config, dict):
            raise ValueError("mixed_phase_pairs must be a mapping")
        names = config["protocol"].get("direction_names")
        if not isinstance(names, list) or len(names) != len(directions):
            raise ValueError("mixed probes require protocol.direction_names")
        base_amplitudes = mixed_config.get("base_amplitudes")
        if not isinstance(base_amplitudes, dict) or set(base_amplitudes) != set(names):
            raise ValueError("mixed base_amplitudes must match protocol.direction_names")
        scales = validate_strengths(mixed_config["scales"], minimum_count=3)
        phase_offset = float(mixed_config["phase_offset"])
        condition_plan = mirrored_phase_pair_conditions(
            directions,
            names,
            scales,
            [base_amplitudes[name] for name in names],
            reference_phase=float(config["protocol"].get("phase", 0.0)),
            phase_offset=phase_offset,
        )
        strengths = scales
        strengths_by_direction = None
    else:
        strengths = validate_strengths(config["strengths"], minimum_count=2)
        if configured_by_direction is None:
            strengths_by_direction = [strengths.copy() for _ in directions]
        else:
            if not isinstance(configured_by_direction, dict):
                raise ValueError("strengths_by_direction must map direction labels to strengths")
            if set(configured_by_direction) != set(direction_labels):
                raise ValueError(
                    "strengths_by_direction keys must exactly match configured direction labels"
                )
            strengths_by_direction = [
                validate_strengths(configured_by_direction[label], minimum_count=1)
                for label in direction_labels
            ]
            strength_union = np.unique(np.concatenate(strengths_by_direction))
            if not np.array_equal(strength_union, strengths):
                raise ValueError(
                    "strengths must be the sorted union of strengths_by_direction"
                )
    minimum_cycles = int(config["observation_rule"]["minimum_cycles"])
    minimum_time = float(config["observation_rule"]["minimum_physical_time"])
    if minimum_cycles < 1 or minimum_time <= 0:
        raise ValueError("observation_rule floors must be positive")
    cycles = minimum_duration_cycle_count(omega, minimum_cycles, minimum_time)
    harmonics = np.asarray(config["harmonics"], dtype=int)
    n_phase = int(config["n_phase"])
    if not {0, 1, 2, 3, 4, 5}.issubset(set(int(value) for value in harmonics)):
        raise ValueError("harmonics must include the complete set 0..5")
    if 2 * int(np.max(np.abs(harmonics))) >= n_phase:
        raise ValueError("configured harmonics must lie strictly below Nyquist")
    dense = config.get("dense")
    if dense is not None:
        for field in ("dt", "n_theta_bins", "welch_segment", "max_psd_bins"):
            if float(dense[field]) <= 0:
                raise ValueError(f"dense.{field} must be positive")
    inference = config.get("inference") or {}
    confidence = float(inference.get("confidence", 0.95))
    fdr_alpha = float(inference.get("fdr_alpha", 0.05))
    if not 0 < confidence < 1:
        raise ValueError("inference.confidence must lie strictly between zero and one")
    if not 0 < fdr_alpha < 1:
        raise ValueError("inference.fdr_alpha must lie strictly between zero and one")
    policy = parse_retention_policy(config, block_count)
    generate_figures = config.get("generate_figures", True)
    if not isinstance(generate_figures, bool):
        raise ValueError("generate_figures must be true or false")
    if generate_figures and policy.spectrum_blocks == 0:
        raise ValueError("figure generation requires retained spectrum blocks")
    mixed_mode = mixed_config is not None
    if mixed_mode and generate_figures:
        raise ValueError("mixed phase-pair collection requires generate_figures=false")
    if (configured_by_direction is not None or mixed_mode) and config.get("extension") is not None:
        raise ValueError("extension runs do not support strengths_by_direction")
    extension = _validate_extension_config(config, strengths)
    return {
        "omega": omega,
        "strengths": strengths,
        "strengths_by_direction": strengths_by_direction,
        "experiment_type": "mirrored_phase_pair" if mixed_mode else "paired_sign",
        "condition_plan": condition_plan,
        "directions": directions,
        "block_count": block_count,
        "cycles": cycles,
        "harmonics": [int(value) for value in harmonics],
        "retention_policy": policy,
        "confidence": confidence,
        "fdr_alpha": fdr_alpha,
        "generate_figures": generate_figures,
        "extension": extension,
    }


def _artifact_path(value) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_extension_config(config: dict, strengths: np.ndarray):
    extension = config.get("extension")
    if extension is None:
        return None
    if not isinstance(extension, dict):
        raise ValueError("extension must be an object")
    new_strengths = validate_strengths(extension["new_strengths"], minimum_count=1)
    if any(not np.any(np.isclose(strengths, value, rtol=0.0, atol=1e-12))
           for value in new_strengths):
        raise ValueError("every extension.new_strengths value must appear in strengths")
    base_strengths = np.asarray([
        value for value in strengths
        if not np.any(np.isclose(new_strengths, value, rtol=0.0, atol=1e-12))
    ])
    if not len(base_strengths):
        raise ValueError("an extension requires at least one base strength")
    base_artifact = _artifact_path(extension["base_artifact"])
    if not base_artifact.is_dir():
        raise ValueError(f"base artifact does not exist: {base_artifact}")
    output_root = _artifact_path(config["output_root"])
    if output_root == base_artifact.parent:
        raise ValueError("extension output_root must differ from the base output root")
    return {
        "base_artifact": base_artifact,
        "new_strengths": new_strengths,
        "base_strengths": base_strengths,
    }


def probe_task_counts(checked: dict) -> dict:
    """Expected integration task counts (no integration performed)."""
    if checked.get("extension") is not None:
        forced_conditions = (
            2 * len(checked["extension"]["new_strengths"])
            * len(checked["directions"])
        )
        integrated_conditions = 1 + forced_conditions
        combined_conditions = 1 + 2 * len(checked["strengths"]) * len(checked["directions"])
        return {
            "unforced_trajectories": checked["block_count"],
            "forced_trajectories": checked["block_count"] * forced_conditions,
            "total_condition_integrations": checked["block_count"] * integrated_conditions,
            "spinup_integrations": checked["block_count"],
            "conditions_per_block": integrated_conditions,
            "combined_conditions_per_block": combined_conditions,
        }
    if checked["experiment_type"] == "mirrored_phase_pair":
        conditions = len(checked["condition_plan"]["forcing_vectors"])
        forced_conditions = conditions - 1
    else:
        forced_conditions = 2 * sum(
            len(values) for values in checked["strengths_by_direction"]
        )
        conditions = 1 + forced_conditions
    return {
        "unforced_trajectories": checked["block_count"],
        "forced_trajectories": checked["block_count"] * (conditions - 1),
        "total_condition_integrations": checked["block_count"] * conditions,
        "spinup_integrations": checked["block_count"],
        "conditions_per_block": conditions,
        "combined_conditions_per_block": conditions,
    }


def _base_artifact_spec(config: dict, checked: dict, blocks) -> dict:
    extension = checked["extension"]
    artifact = extension["base_artifact"]
    snapshot_path = artifact / "config_snapshot.json"
    manifest_path = artifact / "manifest.json"
    if not snapshot_path.is_file() or not manifest_path.is_file():
        raise ValueError("base artifact requires config_snapshot.json and manifest.json")
    base_config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = base_manifest.get("files", {})
    base_checked = validate_probe_config(base_config)
    compatibility_fields = (
        "omega", "block_count", "discard_time", "n_phase", "harmonics",
        "observation_rule", "lorenz", "solver", "initial_ensemble", "protocol",
        "dense", "retention",
    )
    mismatched = [
        field for field in compatibility_fields
        if base_config.get(field) != config.get(field)
    ]
    if mismatched:
        raise ValueError(f"base and extension settings differ: {mismatched}")
    if not np.array_equal(base_checked["strengths"], extension["base_strengths"]):
        raise ValueError("base strengths do not equal final strengths minus new strengths")

    prefix = f"omega_{format(float(checked['omega']), '.12g').replace('-', 'm').replace('.', 'p')}"
    checkpoint_path = artifact / "checkpoints" / f"{prefix}.npz"
    if not checkpoint_path.is_file():
        raise ValueError("base checkpoint is missing")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        base_ids = np.asarray(checkpoint[f"{prefix}_block_ids"], dtype=np.uint32)
        base_means_shape = checkpoint[f"{prefix}_condition_means"].shape
        base_vectors = np.asarray(
            checkpoint[f"{prefix}_condition_vectors"], dtype=float
        )
        base_strengths = np.asarray(checkpoint[f"{prefix}_strengths"], dtype=float)
        base_harmonics = np.asarray(checkpoint[f"{prefix}_harmonics"], dtype=int)
    expected_vectors = paired_condition_vectors(
        base_checked["directions"], base_checked["strengths"], include_unforced=True
    )
    if not np.array_equal(base_ids, np.asarray(blocks.block_ids, dtype=np.uint32)):
        raise ValueError("base checkpoint block IDs differ from regenerated block IDs")
    if base_means_shape != (
        checked["block_count"], len(expected_vectors), 3, len(checked["harmonics"])
    ):
        raise ValueError("base checkpoint condition means have incompatible axes")
    if not np.allclose(base_vectors, expected_vectors, rtol=0.0, atol=1e-12):
        raise ValueError("base checkpoint condition vectors do not match its condition axis")
    if not np.array_equal(base_strengths, base_checked["strengths"]):
        raise ValueError("base checkpoint strengths differ from the base config")
    if not np.array_equal(base_harmonics, np.asarray(checked["harmonics"], dtype=int)):
        raise ValueError("base checkpoint harmonics differ from the extension")

    regenerated = generate_configured_initial_state_blocks(base_config)
    if not np.array_equal(regenerated.block_ids, blocks.block_ids):
        raise ValueError("base and extension regenerated block IDs differ")
    if not np.array_equal(regenerated.final_states, blocks.final_states):
        raise ValueError("base and extension regenerated initial states differ")

    checkpoint_relative = f"checkpoints/{prefix}.npz"
    required_manifest_paths = {"config_snapshot.json", checkpoint_relative}
    required_manifest_paths.update(
        f"{BLOCK_LEVEL_DIRECTORY}/{prefix}/"
        f"{chunk_file_name(prefix, start, end)}"
        for start, end in chunk_ranges(
            checked["block_count"], checked["retention_policy"].chunk_blocks
        )
    )
    missing_manifest = required_manifest_paths.difference(manifest_files)
    if missing_manifest:
        raise ValueError(
            f"base manifest lacks required files: {sorted(missing_manifest)}"
        )
    for path, relative in (
        (snapshot_path, "config_snapshot.json"),
        (checkpoint_path, checkpoint_relative),
    ):
        identifier = f"sha256:{file_sha256(path)}"
        if manifest_files[relative] != identifier:
            raise ValueError(f"base manifest hash mismatch: {relative}")

    return {
        "artifact": artifact,
        "config": base_config,
        "checked": base_checked,
        "prefix": prefix,
        "condition_vectors": base_vectors,
        "manifest_identifier": f"sha256:{file_sha256(manifest_path)}",
        "config_identifier": f"sha256:{file_sha256(snapshot_path)}",
        "manifest_files": manifest_files,
    }


def generate_probe_cell(config, checked, blocks, output_dir, policy) -> dict:
    if checked.get("extension") is not None:
        base = _base_artifact_spec(config, checked, blocks)
        return _generate_extension_probe_cell(
            config, checked, blocks, output_dir, policy, base
        )
    return _generate_full_probe_cell(config, checked, blocks, output_dir, policy)


def _generate_full_probe_cell(config, checked, blocks, output_dir, policy) -> dict:
    """Integrate one frequency cell with configured directions and one shared
    unforced trajectory per block, persisting the retained block-level
    objects and returning the per-condition cycle means."""
    omega = checked["omega"]
    directions = checked["directions"]
    block_count = checked["block_count"]
    workers = int(config.get("workers", 1))
    condition_plan = checked["condition_plan"]
    if condition_plan is None:
        forcing_vectors = paired_direction_strength_vectors(
            directions, checked["strengths_by_direction"], include_unforced=True
        )
        forcing_phases = None
        labels = None
        mixed_pairs = []
    else:
        forcing_vectors = condition_plan["forcing_vectors"]
        forcing_phases = condition_plan["forcing_phases"]
        labels = condition_plan["condition_labels"]
        mixed_pairs = condition_plan["pairs"]
    condition_count = len(forcing_vectors)
    sampling = CycleFourierSampling(
        omega=omega,
        phase=float(config["protocol"].get("phase", 0.0)),
        discard_time=float(config["discard_time"]),
        n_cycle=checked["cycles"],
        n_phase=int(config["n_phase"]),
        harmonics=np.asarray(config["harmonics"], dtype=int),
    )
    dense_config = dict(config["dense"])
    if dense_config.get("spectrum_harmonic_cap") is not None:
        cap = float(dense_config["spectrum_harmonic_cap"])
        dense_config["spectrum_max_omega"] = min(
            np.pi / float(dense_config["dt"]), cap * omega
        )
    numerical_config = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["solver"]),
    }
    prefix = f"omega_{format(float(omega), '.12g').replace('-', 'm').replace('.', 'p')}"
    means_pieces = []
    for start, end in chunk_ranges(block_count, policy.chunk_blocks):
        chunk_ids = blocks.block_ids[start:end]
        chunk_length = end - start
        need_spectrum = start < policy.spectrum_blocks
        retained_spectrum_count = max(
            0, min(policy.spectrum_blocks - start, chunk_length)
        )
        chunk_path = (
            output_dir / BLOCK_LEVEL_DIRECTORY / prefix
            / chunk_file_name(prefix, start, end)
        )
        if chunk_path.is_file():
            with np.load(chunk_path, allow_pickle=False) as retained:
                expected_ids = np.asarray(chunk_ids, dtype=np.uint32)
                if (
                    "condition_means" not in retained
                    or not np.array_equal(retained["block_ids"], expected_ids)
                    or retained["condition_means"].shape
                    != (chunk_length, condition_count, 3, len(checked["harmonics"]))
                    or need_spectrum != ("spectrum" in retained)
                    or (
                        need_spectrum
                        and retained["spectrum"].shape[0]
                        != retained_spectrum_count
                    )
                ):
                    raise ValueError(f"incomplete or incompatible probe chunk: {chunk_path}")
                means_pieces.append(np.asarray(retained["condition_means"]))
            print(
                f"[{time.strftime('%H:%M:%S')}] probe chunk {start}-{end} reused",
                flush=True,
            )
            continue
        spectra = None
        if need_spectrum:
            fourier, _, spectra, _, _, _ = integrate_phase_and_dense_conditions(
                chunk_ids,
                blocks.final_states[start:end],
                forcing_vectors,
                sampling,
                dense_config,
                numerical_config,
                forcing_phases=forcing_phases,
                workers=workers,
                raw_segment_blocks=0,
                retain_phase_samples=False,
                retain_dense_blocks=0,
                compute_summaries=False,
            )
        else:
            fourier = integrate_cycle_fourier_conditions(
                chunk_ids,
                blocks.final_states[start:end],
                forcing_vectors,
                sampling,
                numerical_config,
                forcing_phases=forcing_phases,
                workers=workers,
            )
        condition_means = fourier.mean(axis=3)  # (chunk, cond, 3, H)
        means_pieces.append(condition_means)
        retained_fourier = None
        if policy.per_cycle_fourier_blocks > start:
            retained_fourier = np.moveaxis(
                fourier[: policy.per_cycle_fourier_blocks - start], 2, 3
            )
        retained_spectra = None
        spectrum_grid = None
        spectrum_counts = None
        if need_spectrum:
            retained_spectra = np.stack(
                [
                    np.stack(
                        [spectra[b, c].coefficients for c in range(condition_count)]
                    )
                    for b in range(retained_spectrum_count)
                ]
            )
            spectrum_grid = spectra[0, 0].frequency_grid
            spectrum_counts = np.stack(
                [
                    np.asarray(
                        [spectra[b, c].segment_count for c in range(condition_count)]
                    )
                    for b in range(retained_spectrum_count)
                ]
            )
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        write_block_level_chunk(
            chunk_path,
            _chunk_arrays(
                chunk_ids, condition_means, retained_fourier, retained_spectra,
                spectrum_grid, spectrum_counts,
            ),
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] probe chunk {start}-{end} of "
            f"{block_count} integrated",
            flush=True,
        )
    means = np.concatenate(means_pieces, axis=0)  # (B, cond, 3, H)
    return {
        "means": means,
        "condition_vectors": forcing_vectors,
        "condition_phases": forcing_phases,
        "condition_labels": labels,
        "mixed_pairs": mixed_pairs,
        "directions": directions,
    }


def _generate_extension_probe_cell(
    config, checked, blocks, output_dir, policy, base
) -> dict:
    """Integrate only new paired strengths and persist self-contained chunks.

    Each retained chunk merges the immutable base conditions with the newly
    integrated conditions, then reorders every retained array onto the normal
    canonical direction/strength/sign axis.  Stored condition vectors use
    exactly that same order; all downstream lookup remains value-based.
    """
    omega = checked["omega"]
    directions = checked["directions"]
    new_strengths = checked["extension"]["new_strengths"]
    block_count = checked["block_count"]
    workers = int(config.get("workers", 1))
    extension_vectors = paired_condition_vectors(
        directions, new_strengths, include_unforced=True
    )
    source_vectors = np.concatenate(
        (base["condition_vectors"], extension_vectors[1:]), axis=0
    )
    combined_vectors = paired_condition_vectors(
        directions, checked["strengths"], include_unforced=True
    )
    condition_permutation = condition_axis_permutation(
        source_vectors, combined_vectors
    )
    integrated_condition_count = len(extension_vectors)
    new_forced_condition_count = integrated_condition_count - 1
    combined_condition_count = len(combined_vectors)
    validation = {
        "status": "accepted",
        "comparison": "redundant extension unforced versus immutable base unforced",
        "rtol": UNFORCED_ALIGNMENT_RTOL,
        "atol": UNFORCED_ALIGNMENT_ATOL,
        "condition_means_max_abs_difference": 0.0,
        "spectrum_max_abs_difference": 0.0,
        "cycle_fourier_max_abs_difference": 0.0,
        "spectrum_segment_counts_exact": True,
        "chunks_checked": 0,
    }
    sampling = CycleFourierSampling(
        omega=omega,
        phase=float(config["protocol"].get("phase", 0.0)),
        discard_time=float(config["discard_time"]),
        n_cycle=checked["cycles"],
        n_phase=int(config["n_phase"]),
        harmonics=np.asarray(config["harmonics"], dtype=int),
    )
    dense_config = dict(config["dense"])
    if dense_config.get("spectrum_harmonic_cap") is not None:
        cap = float(dense_config["spectrum_harmonic_cap"])
        dense_config["spectrum_max_omega"] = min(
            np.pi / float(dense_config["dt"]), cap * omega
        )
    numerical_config = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["solver"]),
    }
    prefix = base["prefix"]
    means_pieces = []
    for start, end in chunk_ranges(block_count, policy.chunk_blocks):
        chunk_ids = np.asarray(blocks.block_ids[start:end], dtype=np.uint32)
        chunk_length = end - start
        need_spectrum = start < policy.spectrum_blocks
        retained_spectrum_count = max(
            0, min(policy.spectrum_blocks - start, chunk_length)
        )
        filename = chunk_file_name(prefix, start, end)
        chunk_path = output_dir / BLOCK_LEVEL_DIRECTORY / prefix / filename
        if chunk_path.is_file():
            with np.load(chunk_path, allow_pickle=False) as retained:
                missing_validation = set(EXTENSION_VALIDATION_ARRAYS.values()).difference(
                    retained.files
                )
                if (
                    "condition_means" not in retained
                    or missing_validation
                    or not np.array_equal(retained["block_ids"], chunk_ids)
                    or retained["condition_means"].shape
                    != (chunk_length, combined_condition_count, 3,
                        len(checked["harmonics"]))
                    or need_spectrum != ("spectrum" in retained)
                    or (
                        need_spectrum
                        and retained["spectrum"].shape[0]
                        != retained_spectrum_count
                    )
                ):
                    raise ValueError(
                        f"incomplete or incompatible extension chunk: {chunk_path}"
                    )
                means_pieces.append(np.asarray(retained["condition_means"]))
                validation["condition_means_max_abs_difference"] = max(
                    validation["condition_means_max_abs_difference"],
                    float(retained[EXTENSION_VALIDATION_ARRAYS["mean"]]),
                )
                validation["spectrum_max_abs_difference"] = max(
                    validation["spectrum_max_abs_difference"],
                    float(retained[EXTENSION_VALIDATION_ARRAYS["spectrum"]]),
                )
                validation["cycle_fourier_max_abs_difference"] = max(
                    validation["cycle_fourier_max_abs_difference"],
                    float(retained[EXTENSION_VALIDATION_ARRAYS["cycle"]]),
                )
                validation["spectrum_segment_counts_exact"] = bool(
                    validation["spectrum_segment_counts_exact"]
                    and bool(retained[EXTENSION_VALIDATION_ARRAYS["counts"]])
                )
                validation["chunks_checked"] += 1
            print(
                f"[{time.strftime('%H:%M:%S')}] extension chunk {start}-{end} reused",
                flush=True,
            )
            continue

        base_path = (
            base["artifact"] / BLOCK_LEVEL_DIRECTORY / prefix / filename
        )
        if not base_path.is_file():
            raise ValueError(f"base chunk is missing: {base_path}")
        base_relative = str(base_path.relative_to(base["artifact"])).replace("\\", "/")
        if (
            f"sha256:{file_sha256(base_path)}"
            != base["manifest_files"][base_relative]
        ):
            raise ValueError(f"base manifest hash mismatch: {base_relative}")
        spectra = None
        if need_spectrum:
            fourier, _, spectra, _, _, _ = integrate_phase_and_dense_conditions(
                chunk_ids,
                blocks.final_states[start:end],
                extension_vectors,
                sampling,
                dense_config,
                numerical_config,
                workers=workers,
                raw_segment_blocks=0,
                retain_phase_samples=False,
                retain_dense_blocks=0,
                compute_summaries=False,
            )
        else:
            fourier = integrate_cycle_fourier_conditions(
                chunk_ids,
                blocks.final_states[start:end],
                extension_vectors,
                sampling,
                numerical_config,
                workers=workers,
            )
        new_means = fourier.mean(axis=3)
        new_cycle_fourier = None
        if policy.per_cycle_fourier_blocks > start:
            new_cycle_fourier = np.moveaxis(
                fourier[: policy.per_cycle_fourier_blocks - start], 2, 3
            )
        new_spectra = None
        new_grid = None
        new_counts = None
        if need_spectrum:
            new_spectra = np.stack([
                np.stack([
                    spectra[b, c].coefficients
                    for c in range(integrated_condition_count)
                ])
                for b in range(retained_spectrum_count)
            ])
            new_grid = np.asarray(spectra[0, 0].frequency_grid, dtype=float)
            new_counts = np.stack([
                np.asarray([
                    spectra[b, c].segment_count
                    for c in range(integrated_condition_count)
                ])
                for b in range(retained_spectrum_count)
            ])

        with np.load(base_path, allow_pickle=False) as retained:
            mean_difference = 0.0
            spectrum_difference = 0.0
            cycle_difference = 0.0
            counts_exact = True
            required = {"block_ids", "condition_means"}
            if need_spectrum:
                required.update({
                    "spectrum", "spectrum_frequency_grid", "spectrum_segment_counts"
                })
            missing = required.difference(retained.files)
            if missing:
                raise ValueError(f"base chunk lacks arrays {sorted(missing)}: {base_path}")
            if not np.array_equal(retained["block_ids"], chunk_ids):
                raise ValueError(f"base chunk block IDs differ: {base_path}")
            base_means = np.asarray(retained["condition_means"])
            if base_means.shape[1] != len(base["condition_vectors"]):
                raise ValueError(f"base chunk condition axis differs: {base_path}")
            mean_difference = float(np.max(np.abs(base_means[:, 0] - new_means[:, 0])))
            validation["condition_means_max_abs_difference"] = max(
                validation["condition_means_max_abs_difference"], mean_difference
            )
            if not np.allclose(
                base_means[:, 0], new_means[:, 0],
                rtol=UNFORCED_ALIGNMENT_RTOL, atol=UNFORCED_ALIGNMENT_ATOL,
            ):
                raise ValueError(
                    f"redundant unforced condition means fail alignment: {base_path}"
                )
            combined_means = np.concatenate(
                (base_means, new_means[:, 1:]), axis=1
            )[:, condition_permutation]
            combined_cycle = None
            if new_cycle_fourier is not None:
                if "cycle_fourier" not in retained:
                    raise ValueError(f"base cycle Fourier retention differs: {base_path}")
                base_cycle = np.asarray(retained["cycle_fourier"])
                cycle_difference = float(np.max(
                    np.abs(base_cycle[:, 0] - new_cycle_fourier[:, 0])
                ))
                validation["cycle_fourier_max_abs_difference"] = max(
                    validation["cycle_fourier_max_abs_difference"], cycle_difference
                )
                if not np.allclose(
                    base_cycle[:, 0], new_cycle_fourier[:, 0],
                    rtol=UNFORCED_ALIGNMENT_RTOL, atol=UNFORCED_ALIGNMENT_ATOL,
                ):
                    raise ValueError(
                        f"redundant unforced cycle Fourier fails alignment: {base_path}"
                    )
                combined_cycle = np.concatenate(
                    (base_cycle, new_cycle_fourier[:, 1:]), axis=1
                )[:, condition_permutation]
            elif "cycle_fourier" in retained:
                raise ValueError(f"extension cycle Fourier retention differs: {base_path}")
            combined_spectra = None
            combined_counts = None
            combined_grid = None
            if need_spectrum:
                combined_grid = np.asarray(
                    retained["spectrum_frequency_grid"], dtype=float
                )
                if not np.array_equal(combined_grid, new_grid):
                    raise ValueError(f"base and extension spectrum grids differ: {base_path}")
                base_spectrum = np.asarray(retained["spectrum"])
                base_counts = np.asarray(retained["spectrum_segment_counts"])
                spectrum_difference = float(np.max(
                    np.abs(base_spectrum[:, 0] - new_spectra[:, 0])
                ))
                validation["spectrum_max_abs_difference"] = max(
                    validation["spectrum_max_abs_difference"], spectrum_difference
                )
                if not np.allclose(
                    base_spectrum[:, 0], new_spectra[:, 0],
                    rtol=UNFORCED_ALIGNMENT_RTOL, atol=UNFORCED_ALIGNMENT_ATOL,
                ):
                    raise ValueError(
                        f"redundant unforced spectra fail alignment: {base_path}"
                    )
                counts_exact = np.array_equal(base_counts[:, 0], new_counts[:, 0])
                validation["spectrum_segment_counts_exact"] = bool(
                    validation["spectrum_segment_counts_exact"] and counts_exact
                )
                if not counts_exact:
                    raise ValueError(
                        f"redundant unforced spectrum counts fail alignment: {base_path}"
                    )
                combined_spectra = np.concatenate(
                    (base_spectrum, new_spectra[:, 1:]), axis=1
                )[:, condition_permutation]
                combined_counts = np.concatenate(
                    (base_counts, new_counts[:, 1:]),
                    axis=1,
                )[:, condition_permutation]
            validation["chunks_checked"] += 1
        means_pieces.append(combined_means)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_arrays = _chunk_arrays(
                chunk_ids, combined_means, combined_cycle, combined_spectra,
                combined_grid, combined_counts,
        )
        chunk_arrays.update({
            EXTENSION_VALIDATION_ARRAYS["mean"]: np.asarray(mean_difference),
            EXTENSION_VALIDATION_ARRAYS["spectrum"]: np.asarray(spectrum_difference),
            EXTENSION_VALIDATION_ARRAYS["cycle"]: np.asarray(cycle_difference),
            EXTENSION_VALIDATION_ARRAYS["counts"]: np.asarray(counts_exact),
        })
        write_block_level_chunk(chunk_path, chunk_arrays)
        print(
            f"[{time.strftime('%H:%M:%S')}] extension chunk {start}-{end} of "
            f"{block_count} integrated ({new_forced_condition_count} new forced "
            f"plus redundant unforced, "
            f"{combined_condition_count} combined conditions)",
            flush=True,
        )
    expected_chunks = len(list(chunk_ranges(block_count, policy.chunk_blocks)))
    if validation["chunks_checked"] != expected_chunks:
        raise ValueError("extension unforced audit did not cover every retained chunk")
    if not validation["spectrum_segment_counts_exact"]:
        raise ValueError("extension unforced audit did not pass exactly")
    return {
        "means": np.concatenate(means_pieces, axis=0),
        "condition_vectors": combined_vectors,
        "directions": directions,
        "base_artifact": base,
        "unforced_validation": validation,
    }


def _chunk_arrays(chunk_ids, condition_means, cycle_fourier, spectra, grid, counts):
    arrays = {
        "block_ids": np.asarray(chunk_ids, dtype=np.uint32),
        "condition_means": np.asarray(condition_means),
    }
    if cycle_fourier is not None:
        arrays["cycle_fourier"] = cycle_fourier
    if spectra is not None:
        arrays["spectrum"] = spectra
        arrays["spectrum_frequency_grid"] = np.asarray(grid, dtype=float)
        arrays["spectrum_segment_counts"] = np.asarray(counts, dtype=np.int64)
    return arrays


def build_response_map(cell: dict, checked: dict) -> list:
    """direction x output x harmonic x strength x {cos, sin} response map."""
    means = cell["means"]
    directions = checked["directions"]
    strengths_by_direction = checked["strengths_by_direction"]
    harmonics = [int(value) for value in checked["harmonics"]]
    idx = {value: index for index, value in enumerate(harmonics)}
    unforced_index = 0
    unforced = means[:, unforced_index]
    entries = []
    for direction_index, (direction, strengths) in enumerate(
        zip(directions, strengths_by_direction)
    ):
        direction_label, _ = direction_identity(direction)
        for strength in strengths:
            strength = float(strength)
            plus, minus = paired_condition_indices(
                cell["condition_vectors"], direction, strength
            )
            odd, even = paired_order_contrasts(
                means[:, plus], means[:, minus], unforced
            )
            for harmonic in harmonics:
                for output, name in enumerate(STATE_NAMES):
                    if harmonic == 0:
                        dc = even[:, output, idx[0]].real
                        mean = float(dc.mean())
                        se = float(dc.std(ddof=1) / np.sqrt(len(dc)))
                        entries.append({
                            "direction": direction_index,
                            "direction_label": direction_label,
                            "output": name,
                            "harmonic": harmonic,
                            "strength": strength,
                            "kind": "dc_scalar",
                            "mean": mean,
                            "se": se,
                        })
                        continue
                    values = harmonic_order_contrast(odd, even, harmonic)[
                        :, output, idx[harmonic]
                    ]
                    stats = cos_sin_statistics(
                        values, confidence=float(checked.get("confidence", 0.95))
                    )
                    entries.append({
                        "direction": direction_index,
                        "direction_label": direction_label,
                        "output": name,
                        "harmonic": harmonic,
                        "strength": strength,
                        "kind": "odd_contrast" if harmonic % 2 == 1 else "even_contrast",
                        "mean_cos": stats.mean_cos,
                        "se_cos": stats.se_cos,
                        "ci_cos": list(stats.ci_cos),
                        "mean_sin": stats.mean_sin,
                        "se_sin": stats.se_sin,
                        "ci_sin": list(stats.ci_sin),
                        "hotelling_t2": stats.hotelling_t2,
                        "hotelling_f": stats.hotelling_f,
                        "hotelling_p": stats.hotelling_p_value,
                        "block_count": int(len(values)),
                        "used_pseudoinverse": stats.used_pseudoinverse,
                        "magnitude_derived": stats.magnitude,
                        "phase_derived": stats.phase,
                    })
    return entries


def add_probe_q_values(entries: list, alpha: float = 0.05) -> None:
    """Add BH q-values over the complete non-DC probe family in ``entries``."""
    tested = [entry for entry in entries if entry["harmonic"] != 0]
    q_values = benjamini_hochberg(
        [entry["hotelling_p"] for entry in tested]
    )
    for entry, q_value in zip(tested, q_values):
        entry["hotelling_q_bh"] = float(q_value)
        entry["detected_q05"] = bool(q_value <= alpha)


def attach_background_scales(entries, output_dir, prefix, omega):
    """Attach direct-response / single-block chaotic-spectrum display ratios."""
    unforced, frequency_grid = load_spectrum_conditions(
        output_dir / BLOCK_LEVEL_DIRECTORY, prefix, [0]
    )
    background = unforced[:, 0]  # block, output, frequency
    for entry in entries:
        if entry["harmonic"] == 0:
            continue
        output_index = STATE_NAMES.index(entry["output"])
        target = float(entry["harmonic"]) * float(omega)
        bin_index = int(np.argmin(np.abs(frequency_grid - target)))
        single_block = np.abs(background[:, output_index, bin_index])
        median = float(np.median(single_block))
        low, high = np.quantile(single_block, (0.05, 0.95))
        entry["background_frequency_bin"] = float(frequency_grid[bin_index])
        entry["background_single_block_median"] = median
        entry["background_single_block_q05_q95"] = [float(low), float(high)]
        entry["response_to_background_median"] = (
            float(entry["magnitude_derived"] / median) if median > 0 else None
        )
    return background, frequency_grid


def mark_background_scales_unavailable(entries):
    """Keep numerical output schemas stable when spectra are not retained."""
    for entry in entries:
        if entry["harmonic"] == 0:
            continue
        entry["background_frequency_bin"] = None
        entry["background_single_block_median"] = None
        entry["background_single_block_q05_q95"] = [None, None]
        entry["response_to_background_median"] = None


def _significance_rows(entries):
    keys = (
        "direction", "direction_label", "output", "harmonic", "strength",
        "kind", "mean_cos", "ci_cos", "mean_sin", "ci_sin",
        "hotelling_t2", "hotelling_f", "hotelling_p", "hotelling_q_bh",
        "detected_q05", "magnitude_derived",
        "background_frequency_bin", "background_single_block_median",
        "background_single_block_q05_q95", "response_to_background_median",
    )
    return [
        {key: entry[key] for key in keys}
        for entry in entries if entry["harmonic"] != 0
    ]


def _write_significance_csv(path, rows):
    flattened = []
    for row in rows:
        value = dict(row)
        value["ci_cos_low"], value["ci_cos_high"] = value.pop("ci_cos")
        value["ci_sin_low"], value["ci_sin_high"] = value.pop("ci_sin")
        value["background_q05"], value["background_q95"] = value.pop(
            "background_single_block_q05_q95"
        )
        flattened.append(value)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
    writer.writeheader()
    writer.writerows(flattened)
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")


def build_detection_summary(entries, alpha=0.05):
    tested = [entry for entry in entries if entry["harmonic"] != 0]
    harmonics = sorted({int(entry["harmonic"]) for entry in tested})
    by_strength = []
    for strength in sorted({float(entry["strength"]) for entry in tested}):
        detected = [
            entry for entry in tested
            if float(entry["strength"]) == strength and entry["detected_q05"]
        ]
        detected.sort(key=lambda entry: (entry["hotelling_q_bh"], -entry["hotelling_t2"]))
        by_strength.append({
            "strength": strength,
            "detected_cell_count": len(detected),
            "detected_harmonics": sorted({entry["harmonic"] for entry in detected}),
            "cells": [
                {
                    "direction": entry["direction"],
                    "direction_label": entry["direction_label"],
                    "output": entry["output"],
                    "harmonic": entry["harmonic"],
                    "hotelling_p": entry["hotelling_p"],
                    "hotelling_q_bh": entry["hotelling_q_bh"],
                    "mean_cos": entry["mean_cos"],
                    "ci_cos": entry["ci_cos"],
                    "mean_sin": entry["mean_sin"],
                    "ci_sin": entry["ci_sin"],
                    "response_to_background_median": entry["response_to_background_median"],
                }
                for entry in detected
            ],
        })
    by_harmonic = []
    for harmonic in harmonics:
        cells = [entry for entry in tested if entry["harmonic"] == harmonic]
        strongest = min(cells, key=lambda entry: (entry["hotelling_q_bh"], -entry["hotelling_t2"]))
        by_harmonic.append({
            "harmonic": harmonic,
            "detected": any(entry["detected_q05"] for entry in cells),
            "detected_cell_count": sum(entry["detected_q05"] for entry in cells),
            "strongest_cell": {
                key: strongest[key] for key in (
                    "direction", "direction_label", "output", "strength",
                    "mean_cos", "ci_cos", "mean_sin", "ci_sin",
                    "hotelling_t2", "hotelling_p", "hotelling_q_bh",
                    "magnitude_derived", "background_single_block_median",
                    "response_to_background_median",
                )
            },
        })
    return {
        "detection_rule": f"joint signed cos/sin Hotelling T2, BH q <= {alpha:g}",
        "alpha": alpha,
        "multiplicity_method": "Benjamini-Hochberg FDR",
        "probe_family": (
            "all direction x output x harmonic"
            f"{harmonics} x strength cells"
        ),
        "family_size": len(tested),
        "background_scale": (
            "median |S_b(Omega)| of single-block unforced retained spectra at "
            "the nearest display bin; scientific response numerator is the "
            "direct known-frequency coefficient"
            if any(
                entry["background_single_block_median"] is not None
                for entry in tested
            )
            else "not computed because physical-frequency spectra were not retained"
        ),
        "by_strength": by_strength,
        "by_harmonic": by_harmonic,
    }


def _signed_components(values):
    values = np.asarray(values)
    return np.stack((values.real, -values.imag), axis=-1)


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty mixed-analysis table")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")


def _paired_mirror_bootstrap_means(plus, minus, *, resamples: int, seed: int):
    """Resample whole blocks jointly and combine both mirrors per draw."""
    plus = np.asarray(plus)
    minus = np.asarray(minus)
    if plus.shape != minus.shape or plus.ndim < 2 or plus.shape[0] < 2:
        raise ValueError("mirror arrays must share a block axis with at least two blocks")
    cross = np.empty((resamples, *plus.shape[1:]), dtype=complex)
    difference = np.empty_like(cross)
    seed_sequence = np.random.SeedSequence(seed)
    rng = np.random.Generator(np.random.PCG64DXSM(seed_sequence))
    for replicate in range(resamples):
        indices = rng.integers(0, plus.shape[0], size=plus.shape[0])
        plus_mean = plus[indices].mean(axis=0)
        minus_mean = minus[indices].mean(axis=0)
        cross[replicate] = (plus_mean + minus_mean) / 2
        difference[replicate] = (plus_mean - minus_mean) / 2
    return cross, difference


def analyze_mixed_artifact(
    artifact,
    *,
    bootstrap_replicates: int = MIXED_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = MIXED_BOOTSTRAP_SEED,
) -> Path:
    """Read a completed mixed checkpoint and estimate zero-scale coefficients."""
    artifact = Path(artifact).resolve()
    config = json.loads((artifact / "config_snapshot.json").read_text(encoding="utf-8"))
    checked = validate_probe_config(config)
    if checked["experiment_type"] != "mirrored_phase_pair":
        raise ValueError("artifact is not a mixed phase-pair experiment")
    if isinstance(bootstrap_replicates, bool) or int(bootstrap_replicates) < 20:
        raise ValueError("bootstrap_replicates must be at least 20")
    bootstrap_replicates = int(bootstrap_replicates)
    prefix = f"omega_{format(float(checked['omega']), '.12g').replace('-', 'm').replace('.', 'p')}"
    checkpoint_path = artifact / "checkpoints" / f"{prefix}.npz"
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        block_ids = np.asarray(checkpoint[f"{prefix}_block_ids"], dtype=np.uint32)
        means = np.asarray(checkpoint[f"{prefix}_condition_means"])
        vectors = np.asarray(checkpoint[f"{prefix}_condition_vectors"], dtype=float)
        phases = np.asarray(checkpoint[f"{prefix}_condition_phases"], dtype=float)
        harmonics = np.asarray(checkpoint[f"{prefix}_harmonics"], dtype=int)
    plan = checked["condition_plan"]
    if not np.array_equal(vectors, plan["forcing_vectors"]):
        raise ValueError("checkpoint component amplitudes differ from config metadata")
    if not np.array_equal(phases, plan["forcing_phases"]):
        raise ValueError("checkpoint component phases differ from config metadata")
    expected = (len(block_ids), len(vectors), 3, len(harmonics))
    if means.shape != expected or not np.iscomplexobj(means):
        raise ValueError(f"condition means must have complex axes {expected}")
    harmonic_indices = np.flatnonzero(harmonics == 2)
    if len(harmonic_indices) != 1:
        raise ValueError("mixed analysis requires harmonic n=2 exactly once")

    names = tuple(config["protocol"]["direction_names"])
    scales = np.asarray(config["mixed_phase_pairs"]["scales"], dtype=float)
    records_by_name = {
        name: [record for record in plan["pairs"] if record["name"] == name]
        for name in names
    }
    if any(
        [record["scale"] for record in records_by_name[name]] != scales.tolist()
        for name in names
    ):
        raise ValueError("mixed condition records are not ordered by configured scale")
    harmonic = int(harmonic_indices[0])
    plus = np.stack([
        np.stack([
            means[:, int(record["mirror_plus_index"]), :, harmonic]
            for record in records_by_name[name]
        ], axis=1)
        for name in names
    ], axis=1)
    minus = np.stack([
        np.stack([
            means[:, int(record["mirror_minus_index"]), :, harmonic]
            for record in records_by_name[name]
        ], axis=1)
        for name in names
    ], axis=1)
    cross_blocks = (plus + minus) / 2
    difference_blocks = (plus - minus) / 2

    models = {
        "lambda2": (2,),
        "lambda2_lambda4": (2, 4),
        "lambda2_lambda4_lambda6": (2, 4, 6),
    }
    windows = (
        ("low3", np.arange(3)),
        ("low4", np.arange(4)),
        ("full", np.arange(5)),
        ("drop_smallest", np.arange(1, 5)),
    )
    operators = {
        (window_name, model): power_series_operator(scales[indices], orders)
        for window_name, indices in windows
        for model, orders in models.items()
    }
    cross_mean = cross_blocks.mean(axis=0)
    difference_mean = difference_blocks.mean(axis=0)
    fitted = {}
    for pair_index in range(len(names)):
        for window_name, indices in windows:
            for model in models:
                design, operator = operators[window_name, model]
                coefficients = operator @ cross_mean[pair_index, indices]
                residuals = cross_mean[pair_index, indices] - design @ coefficients
                fitted[pair_index, window_name, model] = (coefficients, residuals)

    bootstrap_cross, bootstrap_difference = _paired_mirror_bootstrap_means(
        plus, minus, resamples=bootstrap_replicates, seed=bootstrap_seed
    )
    bootstrap_leading = {
        key: np.empty((bootstrap_replicates, len(names), 3), dtype=complex)
        for key in operators
    }
    for replicate in range(bootstrap_replicates):
        replicate_cross = bootstrap_cross[replicate]
        for (window_name, model), (_, operator) in operators.items():
            window_indices = dict(windows)[window_name]
            for pair_index in range(len(names)):
                bootstrap_leading[window_name, model][replicate, pair_index] = (
                    operator @ replicate_cross[pair_index, window_indices]
                )[0]

    raw_ci = np.percentile(_signed_components(bootstrap_cross), (2.5, 97.5), axis=0)
    difference_ci = np.percentile(
        _signed_components(bootstrap_difference), (2.5, 97.5), axis=0
    )
    raw_rows = []
    for pair_index, name in enumerate(names):
        for scale_index, scale in enumerate(scales):
            record = records_by_name[name][scale_index]
            for output_index, output in enumerate(STATE_NAMES):
                for component_index, component in enumerate(("cos", "sin")):
                    raw_rows.append({
                        "pair": name,
                        "scale": float(scale),
                        "first_amplitude": record["amplitudes"][0],
                        "second_amplitude": record["amplitudes"][1],
                        "output": output,
                        "component": component,
                        "Z_cross": _signed_components(cross_mean)[pair_index, scale_index, output_index, component_index],
                        "Z_cross_ci95_low": raw_ci[0, pair_index, scale_index, output_index, component_index],
                        "Z_cross_ci95_high": raw_ci[1, pair_index, scale_index, output_index, component_index],
                        "Z_diag_difference": _signed_components(difference_mean)[pair_index, scale_index, output_index, component_index],
                        "Z_diag_difference_ci95_low": difference_ci[0, pair_index, scale_index, output_index, component_index],
                        "Z_diag_difference_ci95_high": difference_ci[1, pair_index, scale_index, output_index, component_index],
                    })

    fit_rows = []
    previous_h = {}
    previous_bootstrap_h = {}
    for pair_index, name in enumerate(names):
        base = np.asarray(records_by_name[name][0]["base_amplitudes"], dtype=float)
        base_product = float(np.prod(base))
        for model, orders in models.items():
            for window_name, indices in windows:
                coefficients, residuals = fitted[pair_index, window_name, model]
                chi2 = -2 * coefficients[0] / base_product
                hessian = 2 * chi2
                boot_chi2 = -2 * bootstrap_leading[window_name, model][:, pair_index] / base_product
                boot_hessian = 2 * boot_chi2
                chi2_ci = np.percentile(_signed_components(boot_chi2), (2.5, 97.5), axis=0)
                ci = np.percentile(_signed_components(boot_hessian), (2.5, 97.5), axis=0)
                signed_h = _signed_components(hessian)
                signed_chi2 = _signed_components(chi2)
                signed_residual = _signed_components(residuals)
                key = (pair_index, model)
                for output_index, output in enumerate(STATE_NAMES):
                    for component_index, component in enumerate(("cos", "sin")):
                        previous = previous_h.get((key, output_index, component_index))
                        previous_boot = previous_bootstrap_h.get((key, output_index, component_index))
                        current_boot = _signed_components(boot_hessian)[:, output_index, component_index]
                        drift_ci = (
                            (np.nan, np.nan) if previous_boot is None else
                            tuple(np.percentile(current_boot - previous_boot, (2.5, 97.5)))
                        )
                        fit_rows.append({
                            "pair": name,
                            "output": output,
                            "component": component,
                            "model": model,
                            "orders": ";".join(str(order) for order in orders),
                            "window": window_name,
                            "scales": ";".join(f"{scales[index]:g}" for index in indices),
                            "n_scales": len(indices),
                            "residual_degrees_of_freedom": len(indices) - len(orders),
                            "base_amplitude_product": base_product,
                            "leading_Z_coefficient": _signed_components(coefficients[0])[output_index, component_index],
                            "chi2": signed_chi2[output_index, component_index],
                            "chi2_ci95_low": chi2_ci[0, output_index, component_index],
                            "chi2_ci95_high": chi2_ci[1, output_index, component_index],
                            "H": signed_h[output_index, component_index],
                            "H_ci95_low": ci[0, output_index, component_index],
                            "H_ci95_high": ci[1, output_index, component_index],
                            "H_change_from_previous_window": (
                                np.nan if previous is None else signed_h[output_index, component_index] - previous
                            ),
                            "H_change_ci95_low": drift_ci[0],
                            "H_change_ci95_high": drift_ci[1],
                            "residual_rmse": float(np.sqrt(np.mean(signed_residual[:, output_index, component_index] ** 2))),
                        })
                        previous_h[key, output_index, component_index] = signed_h[output_index, component_index]
                        previous_bootstrap_h[key, output_index, component_index] = current_boot

    final_rows = [
        {**row, "primary": row["model"] == "lambda2_lambda4"}
        for row in fit_rows
        if row["output"] == "z" and row["window"] == "full"
    ]
    stability = []
    for name in names:
        for model in models:
            for component in ("cos", "sin"):
                values = [
                    row["H"] for row in fit_rows
                    if row["pair"] == name and row["output"] == "z"
                    and row["model"] == model and row["component"] == component
                ]
                stability.append({
                    "pair": name,
                    "component": component,
                    "model": model,
                    "H_min_across_windows": min(values),
                    "H_max_across_windows": max(values),
                    "H_range_across_windows": max(values) - min(values),
                })

    output = artifact / "mixed_analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(output / "mixed_raw_contrasts.csv", raw_rows)
    _write_rows_csv(output / "mixed_coefficient_fits.csv", fit_rows)
    _write_rows_csv(output / "mixed_final_coefficients.csv", final_rows)
    _write_rows_csv(output / "mixed_fit_stability.csv", stability)
    write_json_atomic(output / "mixed_analysis_summary.json", {
        "source_checkpoint": str(checkpoint_path),
        "block_count": len(block_ids),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": int(bootstrap_seed),
            "bit_generator": "PCG64DXSM",
            "replication_unit": "whole matched block",
            "pairing": "one index draw shared across every pair, scale, mirror, output, cos and sin; P+/P- combined before fitting",
        },
        "fourier_convention": "coefficient=cos-i*sin",
        "estimator": {
            "Z_cross": "(Z_plus+Z_minus)/2",
            "Z_diag_difference": "(Z_plus-Z_minus)/2 (diagnostic only)",
            "chi2": "-2*leading_lambda2_Z/(base_first*base_second)",
            "Taylor_H": "2*chi2",
            "primary_model": "Z_cross=b2*lambda^2+b4*lambda^4",
        },
        "final_z_coefficients": final_rows,
        "fit_window_stability": stability,
    })
    return output


def run_probe(config_path, resume_dir=None):
    config = json.loads(Path(config_path).resolve().read_text(encoding="utf-8"))
    checked = validate_probe_config(config)
    policy = checked["retention_policy"]
    if resume_dir is not None and Path(resume_dir).is_dir():
        output_dir = Path(resume_dir)
        snapshot_path = output_dir / "config_snapshot.json"
        if not snapshot_path.is_file():
            raise ValueError("resume directory has no config_snapshot.json")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot != config:
            raise ValueError("resume config does not match the artifact snapshot")
    else:
        config_identifier = f"sha256:{file_sha256(config_path)}"
        output_dir = REPO_ROOT / config["output_root"] / (
            f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{config_identifier[7:19]}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(output_dir / "config_snapshot.json", config)
    blocks = generate_configured_initial_state_blocks(config)
    print(
        f"[{time.strftime('%H:%M:%S')}] probe blocks generated "
        f"({len(blocks.block_ids)} blocks)",
        flush=True,
    )
    cell = generate_probe_cell(config, checked, blocks, output_dir, policy)
    prefix = f"omega_{format(float(checked['omega']), '.12g').replace('-', 'm').replace('.', 'p')}"
    if checked.get("extension") is not None:
        write_json_atomic(
            output_dir / "extension_validation.json",
            cell["unforced_validation"],
        )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        f"{prefix}_block_ids": np.asarray(blocks.block_ids, dtype=np.uint32),
        f"{prefix}_condition_means": cell["means"],
        f"{prefix}_condition_vectors": cell["condition_vectors"],
        f"{prefix}_strengths": checked["strengths"],
        f"{prefix}_harmonics": np.asarray(checked["harmonics"], dtype=int),
    }
    if cell.get("condition_phases") is not None:
        checkpoint[f"{prefix}_condition_phases"] = cell["condition_phases"]
        checkpoint[f"{prefix}_mixed_scales"] = checked["condition_plan"]["scales"]
    if cell.get("condition_labels") is not None:
        checkpoint[f"{prefix}_condition_labels"] = np.asarray(cell["condition_labels"])
    write_npz_atomic(checkpoint_dir / f"{prefix}.npz", checkpoint)
    if checked["experiment_type"] == "mirrored_phase_pair":
        write_json_atomic(output_dir / "derived_response_map.json", {
            "experiment_type": "mirrored_phase_pair",
            "convention": "mean(x exp(-i n theta)); cos = Re, sin = -Im",
            "contrast_definition": (
                "Z_cross_raw=(Z_plus+Z_minus)/2; "
                "Z_diag_difference=(Z_plus-Z_minus)/2; "
                "chi2_jk=-2*Z_cross_raw/(a_j*a_k); "
                "Taylor H_jk=2*chi2_jk=-4*Z_cross_raw/(a_j*a_k)"
            ),
            "condition_amplitudes": cell["condition_vectors"].tolist(),
            "condition_phases": cell["condition_phases"].tolist(),
            "condition_labels": list(cell["condition_labels"]),
            "mixed_pairs": cell["mixed_pairs"],
            "mixed_phase_pairs": config["mixed_phase_pairs"],
            "harmonics": checked["harmonics"],
        })
        analyze_mixed_artifact(output_dir)
        _write_probe_manifest(output_dir, config, config_path)
        return output_dir
    entries = build_response_map(cell, checked)
    add_probe_q_values(entries, alpha=checked["fdr_alpha"])
    background = spectrum_grid = None
    if policy.spectrum_blocks:
        background, spectrum_grid = attach_background_scales(
            entries, output_dir, prefix, checked["omega"]
        )
    else:
        mark_background_scales_unavailable(entries)
    family_size = (
        sum(len(values) for values in checked["strengths_by_direction"])
        * len(STATE_NAMES)
        * sum(harmonic != 0 for harmonic in checked["harmonics"])
    )
    base_metadata = None
    if checked.get("extension") is not None:
        base = cell["base_artifact"]
        try:
            base_path = str(base["artifact"].relative_to(REPO_ROOT))
        except ValueError:
            base_path = str(base["artifact"])
        base_metadata = {
            "artifact": base_path,
            "manifest_identifier": base["manifest_identifier"],
            "config_identifier": base["config_identifier"],
            "new_strengths": [
                float(value) for value in checked["extension"]["new_strengths"]
            ],
            "merge": (
                "base conditions plus newly integrated paired conditions, reordered "
                "onto the canonical direction/strength/sign condition axis"
            ),
            "redundant_unforced_validation": cell["unforced_validation"],
        }
    derived = {
        "interpretation": (
            "high-order probe: which (direction, output, harmonic) cells "
            "show a reliable signed response; configured amplitudes are "
            "finite-amplitude responses, not perturbative tensor estimates"
        ),
        "convention": (
            "mean(x exp(-i n theta)); cos component = Re, sin component = -Im"
        ),
        "contrast_definition": (
            "odd=(plus-minus)/2; even=(plus+minus)/2-unforced; parity selects "
            "odd for odd n and even for even n"
        ),
        "detection_rule": (
            "joint signed cos/sin Hotelling T2 with Benjamini-Hochberg "
            f"q<={checked['fdr_alpha']:g} "
            f"over the complete {family_size}-cell non-DC probe family"
        ),
        "directions": [value.tolist() for value in checked["directions"]],
        "strengths": [float(value) for value in checked["strengths"]],
        "strengths_by_direction": {
            direction_identity(direction)[0]: [float(value) for value in strengths]
            for direction, strengths in zip(
                checked["directions"], checked["strengths_by_direction"]
            )
        },
        "harmonics": [0, 1, 2, 3, 4, 5],
        "response_map": entries,
    }
    if base_metadata is not None:
        derived["extension"] = base_metadata
    write_json_atomic(output_dir / "derived_response_map.json", derived)
    significance_rows = _significance_rows(entries)
    write_json_atomic(output_dir / "significance_table.json", {
        "family_size": len(significance_rows),
        "multiplicity_method": "Benjamini-Hochberg FDR",
        "detection_alpha": checked["fdr_alpha"],
        "rows": significance_rows,
    })
    _write_significance_csv(output_dir / "significance_table.csv", significance_rows)
    summary = build_detection_summary(entries, alpha=checked["fdr_alpha"])
    write_json_atomic(output_dir / "detection_summary.json", summary)
    if checked["generate_figures"]:
        figure_dir = output_dir / "figures"
        save_figure(
            figure_probe_signed_responses(
                entries, "cos", confidence=checked["confidence"]
            ),
            figure_dir,
            "signed_cos_response",
        )
        save_figure(
            figure_probe_signed_responses(
                entries, "sin", confidence=checked["confidence"]
            ),
            figure_dir,
            "signed_sin_response",
        )
        save_figure(
            figure_probe_detection_overview(entries),
            figure_dir,
            "detection_overview",
        )
        save_figure(
            figure_probe_spectrum_noise(
                entries,
                cell["directions"],
                cell["condition_vectors"],
                output_dir / BLOCK_LEVEL_DIRECTORY,
                prefix,
                background,
                spectrum_grid,
                checked["omega"],
            ),
            figure_dir,
            "spectrum_noise",
        )
    _write_probe_manifest(output_dir, config, config_path, base_metadata)
    return output_dir


def _write_probe_manifest(output_dir, config, config_path, base_metadata=None):
    provenance = {
        "config_identifier": f"sha256:{file_sha256(config_path)}",
        "code_identifiers": active_source_identifiers(
            REPO_ROOT, "src/lorenz/high_order_probe.py"
        ),
        "git": git_provenance(REPO_ROOT),
        "environment": environment_provenance(),
    }
    if base_metadata is not None:
        provenance["base_artifact"] = base_metadata
    artifact_files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = str(path.relative_to(output_dir)).replace("\\", "/")
            artifact_files[relative] = f"sha256:{file_sha256(path)}"
    manifest = {
        "schema_version": 1,
        "classification": config["classification"],
        "study_id": config["study_id"],
        "files": artifact_files,
        "provenance": provenance,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="external experiment configuration JSON",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config and report task counts only")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--analyze-artifact", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=MIXED_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=MIXED_BOOTSTRAP_SEED)
    arguments = parser.parse_args()
    if arguments.analyze_artifact is not None:
        output = analyze_mixed_artifact(
            arguments.analyze_artifact,
            bootstrap_replicates=arguments.bootstrap_replicates,
            bootstrap_seed=arguments.bootstrap_seed,
        )
        print(json.dumps({"mixed_analysis_dir": str(output)}, indent=2))
        return
    runtime = initialize_execution()
    try:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        checked = validate_probe_config(config)
        counts = probe_task_counts(checked)
        print(json.dumps({
            "omega": checked["omega"],
            "directions": len(checked["directions"]),
            "strengths": [float(value) for value in checked["strengths"]],
            "strengths_by_direction": (
                None
                if checked["experiment_type"] == "mirrored_phase_pair"
                else config.get("strengths_by_direction")
                if config.get("strengths_by_direction") is not None
                else {
                    direction_identity(direction)[0]: [float(value) for value in strengths]
                    for direction, strengths in zip(
                        checked["directions"], checked["strengths_by_direction"]
                    )
                }
            ),
            "mixed_phase_pairs": config.get("mixed_phase_pairs"),
            "experiment_type": checked["experiment_type"],
            "block_count": checked["block_count"],
            "cycles": checked["cycles"],
            "conditions_per_block": counts["conditions_per_block"],
            "combined_conditions_per_block": counts["combined_conditions_per_block"],
            "unforced_trajectories": counts["unforced_trajectories"],
            "forced_trajectories": counts["forced_trajectories"],
            "total_condition_integrations": counts["total_condition_integrations"],
            "spinup_integrations": counts["spinup_integrations"],
        }, indent=2))
        if arguments.dry_run:
            print("dry run: no integration performed")
            return
        output_dir = run_probe(arguments.config, resume_dir=arguments.resume)
        print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    finally:
        close_execution(runtime)


if __name__ == "__main__":
    main()
