"""High-order probe: one frequency, six forcing directions, shared unforced.

Goal of the probe (NOT a perturbative-tensor measurement): probe the 1..5
omega response across the full second-order direction design and identify
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
    paired_condition_indices,
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
    integrate_phase_and_dense_conditions,
    minimum_duration_cycle_count,
)
from .ensemble import generate_configured_initial_state_blocks

BLOCK_LEVEL_DIRECTORY = "block_level"
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
    strengths = validate_strengths(config["strengths"], minimum_count=2)
    block_count = int(config["block_count"])
    if isinstance(block_count, bool) or block_count < 3:
        raise ValueError("block_count must be at least three")
    directions = [np.asarray(value, dtype=float) for value in config["protocol"]["directions"]]
    if len(directions) != 6:
        raise ValueError("the probe requires exactly six forcing directions")
    for direction in directions:
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
            raise ValueError("each direction must be a finite nonzero vector")
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
    extension = _validate_extension_config(config, strengths)
    return {
        "omega": omega,
        "strengths": strengths,
        "directions": directions,
        "block_count": block_count,
        "cycles": cycles,
        "harmonics": [int(value) for value in harmonics],
        "retention_policy": policy,
        "confidence": confidence,
        "fdr_alpha": fdr_alpha,
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
    conditions = 1 + 2 * len(checked["strengths"]) * len(checked["directions"])
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
    """Integrate one frequency cell with six directions and one shared
    unforced trajectory per block, persisting the retained block-level
    objects and returning the per-condition cycle means."""
    omega = checked["omega"]
    strengths = checked["strengths"]
    directions = checked["directions"]
    block_count = checked["block_count"]
    workers = int(config.get("workers", 1))
    forcing_vectors = paired_condition_vectors(
        directions, strengths, include_unforced=True
    )
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
    raw_segment_blocks = int(dense_config.get("raw_segment_blocks", 0))
    prefix = f"omega_{format(float(omega), '.12g').replace('-', 'm').replace('.', 'p')}"
    means_pieces = []
    for start, end in chunk_ranges(block_count, policy.chunk_blocks):
        chunk_ids = blocks.block_ids[start:end]
        chunk_length = end - start
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
                    or (policy.block_spectra and "spectrum" not in retained)
                ):
                    raise ValueError(f"incomplete or incompatible probe chunk: {chunk_path}")
                means_pieces.append(np.asarray(retained["condition_means"]))
            print(
                f"[{time.strftime('%H:%M:%S')}] probe chunk {start}-{end} reused",
                flush=True,
            )
            continue
        chunk_raw = max(0, min(raw_segment_blocks - start, chunk_length))
        fourier, summaries, spectra, _, _, _ = integrate_phase_and_dense_conditions(
            chunk_ids,
            blocks.final_states[start:end],
            forcing_vectors,
            sampling,
            dense_config,
            numerical_config,
            workers=workers,
            raw_segment_blocks=chunk_raw,
            retain_phase_samples=False,
            retain_dense_blocks=0,
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
        if policy.block_spectra:
            retained_spectra = np.stack(
                [
                    np.stack(
                        [spectra[b, c].coefficients for c in range(condition_count)]
                    )
                    for b in range(chunk_length)
                ]
            )
            spectrum_grid = spectra[0, 0].frequency_grid
            spectrum_counts = np.stack(
                [
                    np.asarray(
                        [spectra[b, c].segment_count for c in range(condition_count)]
                    )
                    for b in range(chunk_length)
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
    raw_segment_blocks = int(dense_config.get("raw_segment_blocks", 0))
    prefix = base["prefix"]
    means_pieces = []
    for start, end in chunk_ranges(block_count, policy.chunk_blocks):
        chunk_ids = np.asarray(blocks.block_ids[start:end], dtype=np.uint32)
        chunk_length = end - start
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
                    or (policy.block_spectra and "spectrum" not in retained)
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
        chunk_raw = max(0, min(raw_segment_blocks - start, chunk_length))
        fourier, summaries, spectra, _, _, _ = integrate_phase_and_dense_conditions(
            chunk_ids,
            blocks.final_states[start:end],
            extension_vectors,
            sampling,
            dense_config,
            numerical_config,
            workers=workers,
            raw_segment_blocks=chunk_raw,
            retain_phase_samples=False,
            retain_dense_blocks=0,
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
        if policy.block_spectra:
            new_spectra = np.stack([
                np.stack([
                    spectra[b, c].coefficients
                    for c in range(integrated_condition_count)
                ])
                for b in range(chunk_length)
            ])
            new_grid = np.asarray(spectra[0, 0].frequency_grid, dtype=float)
            new_counts = np.stack([
                np.asarray([
                    spectra[b, c].segment_count
                    for c in range(integrated_condition_count)
                ])
                for b in range(chunk_length)
            ])

        with np.load(base_path, allow_pickle=False) as retained:
            mean_difference = 0.0
            spectrum_difference = 0.0
            cycle_difference = 0.0
            counts_exact = True
            required = {"block_ids", "condition_means"}
            if policy.block_spectra:
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
            if policy.block_spectra:
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
    strengths = checked["strengths"]
    harmonics = [int(value) for value in checked["harmonics"]]
    idx = {value: index for index, value in enumerate(harmonics)}
    unforced_index = 0
    unforced = means[:, unforced_index]
    entries = []
    for direction_index, direction in enumerate(directions):
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
        ),
        "by_strength": by_strength,
        "by_harmonic": by_harmonic,
    }


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
    write_npz_atomic(checkpoint_dir / f"{prefix}.npz", checkpoint)
    entries = build_response_map(cell, checked)
    add_probe_q_values(entries, alpha=checked["fdr_alpha"])
    background, spectrum_grid = attach_background_scales(
        entries, output_dir, prefix, checked["omega"]
    )
    family_size = (
        len(checked["directions"])
        * len(STATE_NAMES)
        * sum(harmonic != 0 for harmonic in checked["harmonics"])
        * len(checked["strengths"])
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
            "show a reliable signed response; high amplitudes are probe "
            "settings, not perturbative coefficients; h>=8 values are discovery "
            "amplitudes and not perturbative tensor estimates"
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
    provenance = {
        "config_identifier": f"sha256:{file_sha256(config_path)}",
        "code_identifiers": active_source_identifiers(REPO_ROOT, "experiments/run_high_order_probe.py"),
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
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/production/high_order_probe_omega_6p7_v1.json",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config and report task counts only")
    parser.add_argument("--resume", type=Path, default=None)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    checked = validate_probe_config(config)
    counts = probe_task_counts(checked)
    print(json.dumps({
        "omega": checked["omega"],
        "directions": len(checked["directions"]),
        "strengths": [float(value) for value in checked["strengths"]],
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


if __name__ == "__main__":
    main()
