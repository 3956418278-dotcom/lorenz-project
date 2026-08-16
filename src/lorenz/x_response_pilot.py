"""x-direction dense frequency response pilot.

Composes the shared sampling, fitting, bootstrap, and persistence machinery
for the result-stage x-direction experiment:

- one uniform dense frequency grid for the first-order landscape (c1),
- a smaller predeclared higher-budget strength set for the second-order
  coefficients (c2) and the validity-boundary evidence,
- block-level dense spectral summaries (folded cycle mean, Welch PSD of the
  residual, small raw segments) alongside the inferential Fourier harmonics,
- the retained-data contract: condition identity, per-cycle Fourier
  coefficients, estimator phase-grid samples, per-block complex
  physical-frequency spectra, and optionally full dense trajectories are
  persisted chunk-by-chunk in ``block_level/`` so aggregation stays a later
  analysis operation instead of an integration-side discard.

Ownership: integration, block generation, Fourier sampling, power-series fits,
bootstrap mechanics, and artifact persistence remain with their existing
modules; this module owns only the pilot-specific composition and the
per-frequency detection/adequacy decision family.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .core import sampling_grids
from .direction_design import LORENZ_PARITY
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
from .retention import (
    chunk_file_name,
    chunk_ranges,
    condition_labels,
    condition_table,
    parse_retention_policy,
    write_block_level_chunk,
    write_condition_metadata,
)
from .strength_bootstrap import bootstrap_max_statistic
from .strength_series import fit_block_power_series
from .strength_study import (
    STATE_NAMES,
    CycleFourierSampling,
    StrengthStudyData,
    _checked_strengths,
    frequency_key,
    integrate_phase_and_dense_conditions,
    minimum_duration_cycle_count,
    strength_fourier_components,
)

RAW_FORCED_AXES = ("block", "strength", "state", "cycle", "harmonic")
RAW_BASELINE_AXES = ("block", "state", "cycle", "harmonic")
FOLDED_AXES = ("block", "condition", "state", "theta_bin")
PSD_AXES = ("block", "condition", "state", "frequency_bin")
CYCLE_VARIANCE_AXES = ("block", "strength", "state", "harmonic", "part")
BASELINE_VARIANCE_AXES = ("block", "state", "harmonic", "part")

TARGET_ORDERS = {
    "odd_fundamental": (1, 3),
    "even_second_harmonic": (2, 4),
    "even_dc": (2, 4),
}

HARMONIC_TARGETS = {
    "odd_fundamental": ("odd", 1),
    "even_second_harmonic": ("even", 2),
    "even_dc": ("even", 0),
}

HIGHER_HARMONICS = (
    ("odd", 3),
    ("even", 4),
    ("odd", 5),
)

BLOCK_LEVEL_DIRECTORY = "block_level"


@dataclass(frozen=True)
class XResponseCell:
    """One frequency cell with block-level Fourier and dense summaries.

    The ``study`` carries per-block cycle means (the cycle axis is a single
    slot); the full per-cycle coefficients, phase-grid samples, complex
    block spectra, and dense trajectories are retained objects that live in
    the ``block_level/`` files or in the ``retained_*`` fields after loading.
    """

    omega: float
    study: StrengthStudyData
    folded_cycle_mean: np.ndarray
    welch_psd: np.ndarray
    segment_counts: np.ndarray
    raw_segments: dict
    dense_metadata: dict
    runtime_seconds: float
    cycle_variances: dict | None = None
    sampling_metadata: dict | None = None
    retained_cycle_fourier: dict | None = None
    retained_phase_samples: dict | None = None
    retained_dense: dict | None = None
    block_spectra: dict | None = None


def validate_x_response_config(config: dict) -> dict:
    """Validate the pilot-specific configuration surface."""
    frequencies = tuple(float(value) for value in config["frequencies"])
    if (
        len(frequencies) < 1
        or any(not np.isfinite(value) or value <= 0 for value in frequencies)
        or tuple(sorted(frequencies)) != frequencies
        or len(set(frequencies)) != len(frequencies)
    ):
        raise ValueError("frequencies must be distinct, positive, and increasing")
    strengths = _checked_strengths(config["strengths"], minimum_count=2)
    block_count = int(config["block_count"])
    if isinstance(block_count, bool) or block_count < 2:
        raise ValueError("block_count must be at least two")
    minimum_cycles = int(config["observation_rule"]["minimum_cycles"])
    minimum_time = float(config["observation_rule"]["minimum_physical_time"])
    if minimum_cycles < 1 or minimum_time <= 0:
        raise ValueError("observation_rule floors must be positive")
    dense = config.get("dense")
    if dense is not None:
        for field in ("dt", "n_theta_bins", "welch_segment", "max_psd_bins"):
            if float(dense[field]) <= 0:
                raise ValueError(f"dense.{field} must be positive")
    protocol = config["protocol"]
    direction = np.asarray(protocol["direction"], dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("protocol.direction must be a finite nonzero vector")
    bootstrap = config["bootstrap"]
    if not 0 < float(bootstrap["confidence"]) < 1:
        raise ValueError("bootstrap confidence must lie strictly inside (0, 1)")
    if int(bootstrap["resamples"]) < 2:
        raise ValueError("bootstrap resamples must be at least two")
    policy = parse_retention_policy(config, block_count)
    return {
        "frequencies": frequencies,
        "strengths": strengths,
        "block_count": block_count,
        "cycles": {
            omega: minimum_duration_cycle_count(omega, minimum_cycles, minimum_time)
            for omega in frequencies
        },
        "direction": direction,
        "retention_policy": policy,
    }


def _dense_summary_arrays(summaries, dense_config, condition_count, block_count):
    """Extract persistent arrays from per-cell dense summary objects."""
    first = summaries[0, 0]
    folded = np.empty(
        (block_count, condition_count, 3, first.folded_cycle_mean.shape[1]),
        dtype=float,
    )
    psd = np.empty(
        (block_count, condition_count, 3, first.welch_psd.shape[1]), dtype=float
    )
    segment_counts = np.empty((block_count, condition_count), dtype=np.int64)
    for block_index in range(block_count):
        for condition in range(condition_count):
            summary = summaries[block_index, condition]
            folded[block_index, condition] = summary.folded_cycle_mean
            psd[block_index, condition] = summary.welch_psd
            segment_counts[block_index, condition] = summary.segment_count
    raw_samples = int(dense_config.get("raw_segment_samples", 0))
    raw_segments = {}
    if raw_samples:
        raw_blocks = int(dense_config.get("raw_segment_blocks", 0))
        first_segment = summaries[0, 0]
        raw_segments = {
            "values": np.stack(
                [
                    np.stack(
                        [
                            summaries[b, c].raw_values
                            for c in range(condition_count)
                        ]
                    )
                    for b in range(raw_blocks)
                ]
            ),
            "times": first_segment.raw_times,
        }
    metadata = {
        "dt": first.dense_dt,
        "frequency_step": first.frequency_step,
        "nyquist": first.nyquist,
        "maximum_frequency": first.maximum_frequency,
        "window": "Hann",
        "residual": "dense samples minus folded cycle mean",
        "one_sided": True,
        "units": "state^2 per (rad/time)",
    }
    return folded, psd, segment_counts, raw_segments, metadata


def _cell_sampling_metadata(
    omega, sampling: CycleFourierSampling, forcing_vectors, dense_config
) -> dict:
    """Structured per-cell sampling record including the actual grids.

    Keys starting with ``_`` hold arrays and are stripped from the JSON
    metadata file; they are persisted as arrays in the summary archive.
    """
    phase_times, dense_times = sampling_grids(
        omega,
        sampling.discard_time,
        sampling.n_cycle,
        sampling.n_phase,
        dense_config["dt"],
    )
    dense_times = np.asarray(dense_times, dtype=float)
    phase_offset = float(
        np.mod(
            sampling.omega * float(sampling.discard_time) + sampling.phase,
            2 * np.pi,
        )
    )
    table = condition_table(forcing_vectors)
    return {
        "omega": float(omega),
        "forcing_phase": float(sampling.phase),
        "discard_time": float(sampling.discard_time),
        "n_cycle": int(sampling.n_cycle),
        "n_phase": int(sampling.n_phase),
        "phase_offset": phase_offset,
        "dense_dt": float(dense_config["dt"]),
        "dense_count": int(len(dense_times)),
        "harmonics": [int(value) for value in sampling.harmonics],
        "condition_labels": list(condition_labels(forcing_vectors)),
        "_phase_sample_times": phase_times,
        "_dense_sample_times": dense_times,
        "_condition_signs": table["sign"],
        "_condition_strengths": table["signed_strength"],
        "_condition_directions": table["direction"],
    }


def _retain_count(policy_count: int, start: int, chunk_length: int) -> int:
    """Number of chunk-local leading blocks covered by a global retention count."""
    if policy_count <= start:
        return 0
    return min(policy_count - start, chunk_length)


def _write_chunk_file(
    block_level_dir,
    prefix,
    start,
    end,
    block_ids,
    cycle_fourier,
    phase_samples,
    dense_values,
    dense_times,
    spectra,
    spectrum_grid,
    spectrum_segment_counts,
    policy,
):
    """Persist the retained block-level objects of one block chunk.

    Array axes follow the declared retention contract: cycle Fourier
    ``(block, condition, cycle, state, harmonic)``; phase samples
    ``(block, condition, cycle, phase, state)``; dense trajectories
    ``(block, condition, state, time)``; spectra
    ``(block, condition, state, frequency_bin)``.
    """
    if block_level_dir is None:
        return
    arrays = {"block_ids": np.asarray(block_ids, dtype=np.uint32)}
    if cycle_fourier is not None:
        arrays["cycle_fourier"] = cycle_fourier
    if phase_samples is not None:
        arrays["phase_values"] = phase_samples
    if dense_values is not None:
        arrays["dense_values"] = dense_values
        arrays["dense_times"] = np.asarray(dense_times, dtype=float)
    if spectra is not None:
        arrays["spectrum"] = spectra
        arrays["spectrum_frequency_grid"] = np.asarray(spectrum_grid, dtype=float)
        arrays["spectrum_segment_counts"] = np.asarray(
            spectrum_segment_counts, dtype=np.int64
        )
    if len(arrays) == 1:
        return
    directory = Path(block_level_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / chunk_file_name(prefix, start, end)
    write_block_level_chunk(path, arrays)


def _proposal_from_config(config: dict):
    proposal_cfg = config["initial_ensemble"]["proposal"]
    return SymmetricXYUniformProposal(
        x_half_width=proposal_cfg["x_half_width"],
        y_half_width=proposal_cfg["y_half_width"],
        z_bounds=tuple(proposal_cfg["z_bounds"]),
    )


def generate_production_blocks(config: dict):
    """Generate the initial-state blocks of one run configuration.

    Shared by every frequency cell of the run: the blocks depend only on the
    initial-ensemble settings, so they are generated once and reused instead
    of being re-integrated per frequency.
    """
    initial = config["initial_ensemble"]
    block_id_start = int(initial["block_id_start"])
    block_count = int(config["block_count"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    numerical_config = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["solver"]),
    }
    return generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        _proposal_from_config(config),
        float(initial["spinup_time"]),
        numerical_config,
    )


def generate_x_response_cell(
    config: dict, omega: float, strengths, cycles: int, block_level_dir=None,
    precomputed_blocks=None,
) -> XResponseCell:
    """Integrate one frequency cell with paired forcing and dense summaries.

    Blocks are integrated in chunks.  Retained block-level objects (per-cycle
    Fourier, phase-grid samples, dense trajectories, complex block spectra)
    are written chunk-by-chunk into ``block_level_dir`` according to the
    configured retention policy; the in-memory cell carries only the
    block-level summaries the analysis family consumes.  Pass
    ``precomputed_blocks`` (from :func:`generate_production_blocks`) to
    share one block generation across every frequency of a run.
    """
    started = time.perf_counter()
    block_count = int(config["block_count"])
    workers = int(config.get("workers", 1))
    block_id_start = int(config["initial_ensemble"]["block_id_start"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    if precomputed_blocks is None:
        blocks = generate_production_blocks(config)
    else:
        blocks = precomputed_blocks
    proposal = _proposal_from_config(config)
    numerical_config = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["solver"]),
    }
    protocol = config["protocol"]
    direction = np.asarray(protocol["direction"], dtype=float)
    strengths = _checked_strengths(strengths, minimum_count=2)
    forcing_vectors = [np.zeros(3)]
    forcing_vectors.extend(
        sign * strength * direction
        for strength in strengths
        for sign in (1.0, -1.0)
    )
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
    sampling = CycleFourierSampling(
        omega=omega,
        phase=float(protocol.get("phase", 0.0)),
        discard_time=float(config["discard_time"]),
        n_cycle=cycles,
        n_phase=int(config["n_phase"]),
        harmonics=np.asarray(config["harmonics"], dtype=int),
    )
    dense_config = dict(config["dense"])
    # Physical per-cell spectrum cap (retention decision): the continuous
    # display spectrum is stored up to min(pi/dt, cap * omega), so a fixed
    # harmonic range costs the same physical range at every frequency.
    if dense_config.get("spectrum_harmonic_cap") is not None:
        cap = float(dense_config["spectrum_harmonic_cap"])
        if cap <= 0:
            raise ValueError("dense.spectrum_harmonic_cap must be positive")
        dense_config["spectrum_max_omega"] = min(
            np.pi / float(dense_config["dt"]), cap * omega
        )
    raw_segment_blocks = int(dense_config.get("raw_segment_blocks", 0))
    policy = parse_retention_policy(config, block_count)
    condition_count = 1 + 2 * len(strengths)

    positive_means, negative_means, unforced_means = [], [], []
    positive_variances, negative_variances, unforced_variances = [], [], []
    summary_pieces = []
    for start, end in chunk_ranges(block_count, policy.chunk_blocks):
        chunk_ids = block_ids[start:end]
        chunk_length = end - start
        chunk_raw_blocks = max(0, min(raw_segment_blocks - start, chunk_length))
        result = integrate_phase_and_dense_conditions(
            chunk_ids,
            blocks.final_states[start:end],
            forcing_vectors,
            sampling,
            dense_config,
            numerical_config,
            workers=workers,
            raw_segment_blocks=chunk_raw_blocks,
            retain_phase_samples=_retain_count(
                policy.phase_samples_blocks, start, chunk_length
            )
            > 0,
            retain_dense_blocks=_retain_count(
                policy.dense_trajectory_blocks, start, chunk_length
            ),
        )
        fourier, summaries, spectra, phase_values, _phase_times, dense = result
        unforced = fourier[:, 0]
        paired = fourier[:, 1:].reshape(
            chunk_length,
            len(strengths),
            2,
            3,
            cycles,
            len(sampling.harmonics),
        )
        positive = paired[:, :, 0]
        negative = paired[:, :, 1]
        positive_means.append(positive.mean(axis=-2))
        negative_means.append(negative.mean(axis=-2))
        unforced_means.append(unforced.mean(axis=-2))
        positive_variances.append(
            np.stack(
                [positive.real.var(axis=-2, ddof=1), positive.imag.var(axis=-2, ddof=1)],
                axis=-1,
            )
        )
        negative_variances.append(
            np.stack(
                [negative.real.var(axis=-2, ddof=1), negative.imag.var(axis=-2, ddof=1)],
                axis=-1,
            )
        )
        unforced_variances.append(
            np.stack(
                [unforced.real.var(axis=-2, ddof=1), unforced.imag.var(axis=-2, ddof=1)],
                axis=-1,
            )
        )
        summary_pieces.append(summaries)

        retained_cycle_fourier = None
        if _retain_count(policy.per_cycle_fourier_blocks, start, chunk_length):
            retained_cycle_fourier = np.moveaxis(
                fourier[
                    : _retain_count(
                        policy.per_cycle_fourier_blocks, start, chunk_length
                    )
                ],
                2,
                3,
            )
        retained_phase_samples = None
        if _retain_count(policy.phase_samples_blocks, start, chunk_length):
            retained_phase_samples = np.moveaxis(
                phase_values[
                    : _retain_count(policy.phase_samples_blocks, start, chunk_length)
                ],
                2,
                -1,
            )
        retained_spectra = None
        spectrum_grid = None
        spectrum_segment_counts = None
        if policy.block_spectra and spectra[0, 0] is not None:
            retained_spectra = np.stack(
                [
                    np.stack([spectra[b, c].coefficients for c in range(condition_count)])
                    for b in range(chunk_length)
                ]
            )
            spectrum_grid = spectra[0, 0].frequency_grid
            spectrum_segment_counts = np.stack(
                [
                    np.asarray(
                        [spectra[b, c].segment_count for c in range(condition_count)]
                    )
                    for b in range(chunk_length)
                ]
            )
        retained_dense = None
        dense_times = None
        if dense is not None:
            retained_dense = dense[0]
            dense_times = dense[1]
        _write_chunk_file(
            block_level_dir,
            f"omega_{frequency_key(omega)}",
            start,
            end,
            chunk_ids,
            retained_cycle_fourier,
            retained_phase_samples,
            retained_dense,
            dense_times,
            retained_spectra,
            spectrum_grid,
            spectrum_segment_counts,
            policy,
        )

    summaries = np.empty((block_count, condition_count), dtype=object)
    for start, end in chunk_ranges(block_count, policy.chunk_blocks):
        chunk_index = start // policy.chunk_blocks
        summaries[start:end] = summary_pieces[chunk_index]

    folded, psd, segment_counts, raw_segments, dense_metadata = (
        _dense_summary_arrays(summaries, dense_config, condition_count, block_count)
    )
    study = StrengthStudyData(
        block_ids=blocks.block_ids,
        strengths=strengths,
        harmonics=sampling.harmonics,
        omega=omega,
        n_phase=int(config["n_phase"]),
        positive_cycle_fourier=np.concatenate(positive_means, axis=0)[
            :, :, :, None, :
        ],
        negative_cycle_fourier=np.concatenate(negative_means, axis=0)[
            :, :, :, None, :
        ],
        unforced_cycle_fourier=np.concatenate(unforced_means, axis=0)[:, :, None, :],
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
    cycle_variances = {
        "positive": np.concatenate(positive_variances, axis=0),
        "negative": np.concatenate(negative_variances, axis=0),
        "unforced": np.concatenate(unforced_variances, axis=0),
    }
    return XResponseCell(
        omega=omega,
        study=study,
        folded_cycle_mean=folded,
        welch_psd=psd,
        segment_counts=segment_counts,
        raw_segments=raw_segments,
        dense_metadata=dense_metadata,
        runtime_seconds=time.perf_counter() - started,
        cycle_variances=cycle_variances,
        sampling_metadata=_cell_sampling_metadata(
            omega, sampling, forcing_vectors, dense_config
        ),
    )


def _load_cell(raw, prefix: str, omega: float, config_snapshot, n_phase: int):
    """Reconstruct one frequency cell from prefixed summary arrays.

    Used both for the full summary archive and for per-frequency resume
    checkpoints (which store the same prefixed keys).  The per-cycle axis is
    restored with a single slot; cycle variances are loaded when present.
    """
    block_ids = tuple(int(value) for value in raw[f"{prefix}_block_ids"])
    strengths = np.asarray(raw[f"{prefix}_strengths"], dtype=float)
    harmonics = np.asarray(raw[f"{prefix}_harmonics"], dtype=int)
    positive = np.asarray(raw[f"{prefix}_positive_cycle_mean"])
    negative = np.asarray(raw[f"{prefix}_negative_cycle_mean"])
    unforced = np.asarray(raw[f"{prefix}_unforced_cycle_mean"])
    study = StrengthStudyData(
        block_ids=block_ids,
        strengths=strengths,
        harmonics=harmonics,
        omega=omega,
        n_phase=_artifact_n_phase(raw, prefix, config_snapshot, n_phase),
        positive_cycle_fourier=positive[:, :, :, None, :],
        negative_cycle_fourier=negative[:, :, :, None, :],
        unforced_cycle_fourier=unforced[:, :, None, :],
        raw_proposals=np.asarray(raw[f"{prefix}_raw_proposals"]),
        initial_states=np.asarray(raw[f"{prefix}_initial_states"]),
        child_spawn_keys=tuple(
            tuple(int(item) for item in row)
            for row in raw[f"{prefix}_child_spawn_keys"]
        ),
        generation_metadata={},
    )
    cycle_variances = None
    if f"{prefix}_positive_cycle_variance" in raw.files:
        cycle_variances = {
            "positive": np.asarray(raw[f"{prefix}_positive_cycle_variance"]),
            "negative": np.asarray(raw[f"{prefix}_negative_cycle_variance"]),
            "unforced": np.asarray(raw[f"{prefix}_unforced_cycle_variance"]),
        }
    cell = XResponseCell(
        omega=omega,
        study=study,
        folded_cycle_mean=(
            np.asarray(raw[f"{prefix}_folded_cycle_mean"])
            if f"{prefix}_folded_cycle_mean" in raw.files
            else None
        ),
        welch_psd=(
            np.asarray(raw[f"{prefix}_welch_psd"])
            if f"{prefix}_welch_psd" in raw.files
            else None
        ),
        segment_counts=(
            np.asarray(raw[f"{prefix}_segment_counts"])
            if f"{prefix}_segment_counts" in raw.files
            else None
        ),
        raw_segments={
            "values": np.asarray(raw[f"{prefix}_raw_segment_values"]),
            "times": np.asarray(raw[f"{prefix}_raw_segment_times"]),
        }
        if f"{prefix}_raw_segment_values" in raw.files
        else {},
        dense_metadata=(
            json.loads(str(np.asarray(raw[f"{prefix}_dense_metadata"])))
            if f"{prefix}_dense_metadata" in raw.files
            else {}
        ),
        runtime_seconds=0.0,
        cycle_variances=cycle_variances,
        sampling_metadata=_load_sampling_metadata(
            raw, prefix, omega, config_snapshot
        ),
    )
    return _attach_legacy_subset(cell, raw, prefix, block_ids)


def load_x_response_cells(
    raw_path, n_phase: int = 32, *, load_block_level: bool = True
) -> dict:
    """Reconstruct per-frequency cells from a persisted raw summary artifact.

    Block-level cycle means are sufficient for the analysis family; the
    per-cycle axis is restored with a single slot.  Retained block-level
    objects (per-cycle Fourier, phase samples, dense trajectories, complex
    spectra) are loaded from the ``block_level/`` chunk files when present;
    legacy per-cycle ``_subset`` arrays of older artifacts are read as a
    fallback.
    """
    raw_path = Path(raw_path)
    raw = np.load(raw_path, allow_pickle=False)
    config_snapshot = None
    snapshot_path = raw_path.parent / "config_snapshot.json"
    if snapshot_path.is_file():
        config_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    try:
        cells = {}
        frequency_names = {
            name[len("omega_") :].split("_", 1)[0]
            for name in raw.files
            if name.startswith("omega_")
        }
        for name in sorted(frequency_names):
            prefix = f"omega_{name}"
            omega = float(name.replace("p", ".").replace("m", "-"))
            cells[omega] = _load_cell(
                raw, prefix, omega, config_snapshot, n_phase
            )
    finally:
        raw.close()
    if load_block_level:
        block_level_root = raw_path.parent / BLOCK_LEVEL_DIRECTORY
        if block_level_root.is_dir():
            for omega, cell in cells.items():
                prefix = f"omega_{frequency_key(omega)}"
                cell_dir = block_level_root / prefix
                if not cell_dir.is_dir():
                    continue
                cells[omega] = _attach_block_level_files(cell, cell_dir, prefix)
    return cells


def _artifact_n_phase(raw, prefix, config_snapshot, fallback: int) -> int:
    if f"{prefix}_n_phase" in raw.files:
        return int(np.asarray(raw[f"{prefix}_n_phase"]))
    if config_snapshot is not None:
        return int(config_snapshot.get("n_phase", fallback))
    return int(fallback)


def _load_sampling_metadata(raw, prefix, omega, config_snapshot) -> dict:
    """Reconstruct the per-cell sampling record, preferring stored arrays."""
    metadata = {}
    scalar_fields = (
        "n_phase",
        "n_cycle",
        "discard_time",
        "forcing_phase",
        "phase_offset",
        "dense_dt",
    )
    for field in scalar_fields:
        if f"{prefix}_{field}" in raw.files:
            metadata[field] = float(np.asarray(raw[f"{prefix}_{field}"]))
    for field in ("_phase_sample_times", "_dense_sample_times"):
        if f"{prefix}{field}" in raw.files:
            metadata[field] = np.asarray(raw[f"{prefix}{field}"])
    for field in ("_condition_signs", "_condition_strengths", "_condition_directions"):
        if f"{prefix}{field}" in raw.files:
            metadata[field] = np.asarray(raw[f"{prefix}{field}"])
    metadata["omega"] = float(omega)
    if config_snapshot is not None and (
        "n_cycle" not in metadata or "discard_time" not in metadata
    ):
        rule = config_snapshot.get("observation_rule", {})
        metadata.setdefault(
            "n_cycle",
            minimum_duration_cycle_count(
                omega,
                int(rule.get("minimum_cycles", 0)),
                float(rule.get("minimum_physical_time", 0.0)),
            )
            if rule
            else None,
        )
        metadata.setdefault("discard_time", float(config_snapshot.get("discard_time", 0.0)))
        metadata.setdefault(
            "forcing_phase",
            float(config_snapshot.get("protocol", {}).get("phase", 0.0)),
        )
        if "phase_offset" not in metadata and metadata.get("discard_time") is not None:
            metadata["phase_offset"] = float(
                np.mod(
                    omega * float(metadata["discard_time"])
                    + metadata.get("forcing_phase", 0.0),
                    2 * np.pi,
                )
            )
        if "dense_dt" not in metadata and config_snapshot.get("dense"):
            metadata["dense_dt"] = float(config_snapshot["dense"]["dt"])
    return metadata


def _attach_legacy_subset(cell: XResponseCell, raw, prefix, block_ids) -> XResponseCell:
    """Read legacy per-cycle ``_subset`` arrays of older artifacts.

    The legacy arrays are strength-indexed (positive/negative/unforced).
    They are normalized to the unified condition-indexed contract
    ``(block, condition, cycle, state, harmonic)`` with condition 0 the
    unforced reference and conditions ``1 + 2*s``/``2 + 2*s`` the +/-h pair
    of strength ``s``.
    """
    positive_field = f"{prefix}_positive_cycle_fourier_subset"
    negative_field = f"{prefix}_negative_cycle_fourier_subset"
    unforced_field = f"{prefix}_unforced_cycle_fourier_subset"
    if positive_field not in raw.files or negative_field not in raw.files:
        return cell
    positive = np.asarray(raw[positive_field])  # (block, strength, state, cycle, harmonic)
    negative = np.asarray(raw[negative_field])
    unforced = np.asarray(raw[unforced_field])  # (block, state, cycle, harmonic)
    block_count, strength_count = positive.shape[:2]
    cycle_count = positive.shape[3]
    condition_count = 1 + 2 * strength_count
    values = np.empty(
        (block_count, condition_count, cycle_count, 3, positive.shape[4]),
        dtype=complex,
    )
    values[:, 0] = unforced.transpose(0, 2, 1, 3)
    for strength_index in range(strength_count):
        values[:, 1 + 2 * strength_index] = positive[:, strength_index].transpose(
            0, 2, 1, 3
        )
        values[:, 2 + 2 * strength_index] = negative[:, strength_index].transpose(
            0, 2, 1, 3
        )
    retained = {
        "values": values,
        "block_ids": np.asarray(block_ids[:block_count], dtype=np.uint32),
    }
    return dataclass_replace_immutable(cell, retained_cycle_fourier=retained)


def _attach_block_level_files(
    cell: XResponseCell, cell_dir: Path, prefix
) -> XResponseCell:
    """Concatenate the chunk files of one frequency cell along the block axis."""
    chunk_paths = sorted(cell_dir.glob(f"{prefix}_blocks_*.npz"))
    if not chunk_paths:
        return cell
    collected = {}
    chunk_block_ids = []
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as chunk:
            for key in chunk.files:
                if key == "block_ids":
                    chunk_block_ids.append(np.asarray(chunk[key], dtype=np.uint32))
                    continue
                collected.setdefault(key, []).append(np.asarray(chunk[key]))

    def concatenate(key):
        pieces = collected[key]
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=0)

    block_ids = np.concatenate(chunk_block_ids, axis=0)
    expected = np.asarray(cell.study.block_ids[: len(block_ids)], dtype=np.uint32)
    if not np.array_equal(block_ids, expected):
        raise ValueError("block_level chunk block IDs disagree with the cell")

    retained_cycle_fourier = None
    if "cycle_fourier" in collected:
        retained_cycle_fourier = {
            "values": concatenate("cycle_fourier"),
            "block_ids": block_ids,
        }
    retained_phase_samples = None
    if "phase_values" in collected:
        times = (cell.sampling_metadata or {}).get("_phase_sample_times")
        if times is None:
            raise ValueError("phase sample times missing from the cell metadata")
        retained_phase_samples = {
            "values": concatenate("phase_values"),
            "times": np.asarray(times, dtype=float),
            "block_ids": block_ids,
        }
    retained_dense = None
    if "dense_values" in collected:
        dense_times = [np.asarray(times) for times in collected["dense_times"]]
        if not all(
            np.array_equal(dense_times[0], times) for times in dense_times
        ):
            raise ValueError("dense times differ across chunk files")
        retained_dense = {
            "values": concatenate("dense_values"),
            "times": dense_times[0],
            "block_ids": block_ids,
        }
    block_spectra = None
    if "spectrum" in collected:
        grids = [np.asarray(grid) for grid in collected["spectrum_frequency_grid"]]
        if not all(np.array_equal(grids[0], grid) for grid in grids):
            raise ValueError("spectrum frequency grids differ across chunk files")
        block_spectra = {
            "values": concatenate("spectrum"),
            "frequency_grid": grids[0],
            "segment_counts": concatenate("spectrum_segment_counts"),
            "block_ids": block_ids,
        }
    return dataclass_replace_immutable(
        cell,
        retained_cycle_fourier=retained_cycle_fourier,
        retained_phase_samples=retained_phase_samples,
        retained_dense=retained_dense,
        block_spectra=block_spectra,
    )


def dataclass_replace_immutable(instance: XResponseCell, **changes) -> XResponseCell:
    """Return a copy of one immutable cell with the given fields replaced."""
    values = {
        field.name: getattr(instance, field.name)
        for field in XResponseCell.__dataclass_fields__.values()
    }
    values.update(changes)
    return XResponseCell(**values)


def merge_cell(base: XResponseCell, extra: XResponseCell) -> XResponseCell:
    """Merge a supplementary strength cell into a base cell of the same frequency.

    Shared strengths must agree exactly (deterministic integration), so the
    merged cell simply concatenates the extra strengths onto the base study.
    Retained block-level objects are concatenated along the strength axis when
    both cells carry them; base-only retention is preserved.  Merging retained
    phase samples or dense trajectories is not yet supported.
    """
    if base.omega != extra.omega:
        raise ValueError("merge requires matching frequencies")
    if base.study.block_ids != extra.study.block_ids:
        raise ValueError("merge requires matching block IDs/order")
    if not np.array_equal(base.study.harmonics, extra.study.harmonics):
        raise ValueError("merge requires matching harmonics")
    shared = np.intersect1d(base.study.strengths, extra.study.strengths)
    for strength in shared:
        b_index = list(base.study.strengths).index(strength)
        e_index = list(extra.study.strengths).index(strength)
        if not np.allclose(
            base.study.positive_cycle_fourier[:, b_index],
            extra.study.positive_cycle_fourier[:, e_index],
        ):
            raise ValueError("merge found disagreeing shared strength cells")
    combined = _checked_strengths(
        list(base.study.strengths) + list(extra.study.strengths)
    )
    order = {float(value): index for index, value in enumerate(combined)}

    def merge_strength_axis(base_array, extra_array):
        merged_shape = (len(combined),) + np.shape(base_array)[1:]
        merged = np.empty(merged_shape, dtype=base_array.dtype)
        for strength in combined:
            index = order[float(strength)]
            if strength in base.study.strengths:
                merged[index] = base_array[
                    list(base.study.strengths).index(strength)
                ]
            else:
                merged[index] = extra_array[
                    list(extra.study.strengths).index(strength)
                ]
        return merged

    def merge_condition_axis(base_array, extra_array):
        """Merge condition-indexed arrays (condition 0 is the unforced cell).

        The unforced condition is shared, so only the forced conditions of the
        extra cell are appended, in the combined strength/sign order.
        """
        if not np.allclose(base_array[:, 0], extra_array[:, 0]):
            raise ValueError("merge found disagreeing shared unforced cells")
        merged_shape = (base_array.shape[0], 1 + 2 * len(combined)) + tuple(
            base_array.shape[2:]
        )
        merged = np.empty(merged_shape, dtype=base_array.dtype)
        merged[:, 0] = base_array[:, 0]
        for strength in combined:
            index = order[float(strength)]
            for sign_offset in (0, 1):
                if strength in base.study.strengths:
                    source = base_array[
                        :, 1 + 2 * list(base.study.strengths).index(strength) + sign_offset
                    ]
                else:
                    source = extra_array[
                        :, 1 + 2 * list(extra.study.strengths).index(strength) + sign_offset
                    ]
                merged[:, 1 + 2 * index + sign_offset] = source
        return merged

    positive = merge_strength_axis(
        np.moveaxis(base.study.positive_cycle_fourier, 1, 0),
        np.moveaxis(extra.study.positive_cycle_fourier, 1, 0),
    )
    negative = merge_strength_axis(
        np.moveaxis(base.study.negative_cycle_fourier, 1, 0),
        np.moveaxis(extra.study.negative_cycle_fourier, 1, 0),
    )
    study = StrengthStudyData(
        block_ids=base.study.block_ids,
        strengths=combined,
        harmonics=base.study.harmonics,
        omega=base.study.omega,
        n_phase=base.study.n_phase,
        positive_cycle_fourier=np.moveaxis(positive, 0, 1),
        negative_cycle_fourier=np.moveaxis(negative, 0, 1),
        unforced_cycle_fourier=base.study.unforced_cycle_fourier,
        raw_proposals=base.study.raw_proposals,
        initial_states=base.study.initial_states,
        child_spawn_keys=base.study.child_spawn_keys,
        generation_metadata=base.study.generation_metadata,
    )
    cycle_variances = None
    if base.cycle_variances is not None and extra.cycle_variances is not None:
        cycle_variances = {
            key: np.moveaxis(
                merge_strength_axis(
                    np.moveaxis(base.cycle_variances[key], 1, 0),
                    np.moveaxis(extra.cycle_variances[key], 1, 0),
                ),
                0,
                1,
            )
            for key in ("positive", "negative")
        }
        cycle_variances["unforced"] = base.cycle_variances["unforced"]
    retained_cycle_fourier = None
    if base.retained_cycle_fourier is not None:
        if extra.retained_cycle_fourier is not None:
            retained_cycle_fourier = {
                "values": merge_condition_axis(
                    base.retained_cycle_fourier["values"],
                    extra.retained_cycle_fourier["values"],
                ),
                "block_ids": base.retained_cycle_fourier["block_ids"],
            }
        else:
            retained_cycle_fourier = dict(base.retained_cycle_fourier)
    elif extra.retained_cycle_fourier is not None:
        raise ValueError(
            "merge requires base retention when the extra cell carries per-cycle arrays"
        )
    block_spectra = None
    if base.block_spectra is not None:
        if extra.block_spectra is not None:
            block_spectra = {
                "values": merge_condition_axis(
                    base.block_spectra["values"], extra.block_spectra["values"]
                ),
                "frequency_grid": base.block_spectra["frequency_grid"],
                "segment_counts": merge_condition_axis(
                    base.block_spectra["segment_counts"].astype(float),
                    extra.block_spectra["segment_counts"].astype(float),
                ).astype(np.int64),
                "block_ids": base.block_spectra["block_ids"],
            }
        else:
            block_spectra = dict(base.block_spectra)
    elif extra.block_spectra is not None:
        raise ValueError(
            "merge requires base retention when the extra cell carries spectra"
        )
    if base.retained_phase_samples is not None or extra.retained_phase_samples is not None:
        raise ValueError("merging retained phase samples is not yet supported")
    if base.retained_dense is not None or extra.retained_dense is not None:
        raise ValueError("merging retained dense trajectories is not yet supported")
    return XResponseCell(
        omega=base.omega,
        study=study,
        folded_cycle_mean=base.folded_cycle_mean,
        welch_psd=base.welch_psd,
        segment_counts=base.segment_counts,
        raw_segments=base.raw_segments,
        dense_metadata=base.dense_metadata,
        runtime_seconds=base.runtime_seconds + extra.runtime_seconds,
        cycle_variances=cycle_variances,
        sampling_metadata=base.sampling_metadata,
        retained_cycle_fourier=retained_cycle_fourier,
        retained_phase_samples=None,
        retained_dense=None,
        block_spectra=block_spectra,
    )


def _parity_nulls(direction) -> dict:
    """Structural-null masks for one direction from Lorenz parity."""
    direction = np.asarray(direction, dtype=float)
    transformed = LORENZ_PARITY * direction
    parity = next(
        (value for value in (-1, 1) if np.allclose(transformed, value * direction)),
        None,
    )
    linear_allowed = (
        LORENZ_PARITY == parity if parity is not None else np.ones(3, dtype=bool)
    )
    quadratic_allowed = (
        LORENZ_PARITY == 1 if parity is not None else np.ones(3, dtype=bool)
    )
    return {
        "odd_fundamental": ~linear_allowed,
        "even_second_harmonic": ~quadratic_allowed,
        "even_dc": ~quadratic_allowed,
    }


def _append_feature(columns, members, values, metadata, structural_null):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("each family component must have exactly one block axis")
    real_index = len(columns)
    columns.append(np.asarray(values.real, dtype=float))
    members.append({**metadata, "part": "real", "structural_null": bool(structural_null)})
    imaginary_index = None
    if np.iscomplexobj(values):
        imaginary_index = len(columns)
        columns.append(np.asarray(values.imag, dtype=float))
        members.append(
            {**metadata, "part": "imaginary", "structural_null": bool(structural_null)}
        )
    return real_index, imaginary_index, structural_null


def build_x_response_family(cells: dict, direction):
    """Build one whole-block family across frequencies, strengths, and targets."""
    columns = []
    members = []
    groups = []
    fits = {}
    nulls = _parity_nulls(direction)
    for omega in sorted(cells):
        cell = cells[omega]
        study = cell.study
        components = strength_fourier_components(study)
        index = {int(value): i for i, value in enumerate(study.harmonics)}
        odd = components.odd
        forced_even = components.forced_even
        baseline = components.unforced
        targets = {
            "odd_fundamental": odd[..., index[1]],
            "even_second_harmonic": forced_even[..., index[2]],
            "even_dc": (
                forced_even[..., index[0]] - baseline[:, None, :, index[0]]
            ).real,
        }
        normalizations = {
            "odd_fundamental": lambda value, strength: 2j * value / strength,
            "even_second_harmonic": lambda value, strength: -4 * value / strength**2,
            "even_dc": lambda value, strength: value / strength**2,
        }
        for target, values in targets.items():
            for strength_index, strength in enumerate(study.strengths):
                for observable in range(3):
                    real_index, imaginary_index, null = _append_feature(
                        columns,
                        members,
                        normalizations[target](
                            values[:, strength_index, observable], strength
                        ),
                        {
                            "omega": omega,
                            "section": "identification",
                            "target": target,
                            "strength": float(strength),
                            "observable": STATE_NAMES[observable],
                        },
                        nulls[target][observable],
                    )
                    groups.append(
                        {
                            "omega": omega,
                            "section": "identification",
                            "target": target,
                            "strength": float(strength),
                            "observable": STATE_NAMES[observable],
                            "real_index": real_index,
                            "imaginary_index": imaginary_index,
                            "structural_null": bool(null),
                        }
                    )
        fitted = {}
        for target, orders in TARGET_ORDERS.items():
            fitted[target] = fit_block_power_series(
                targets[target], study.strengths, orders
            )
        fits[omega] = fitted
        for target, orders in TARGET_ORDERS.items():
            null = nulls[target]
            for coefficient_index, coefficient_name in enumerate(
                ("leading", "higher")
            ):
                for observable in range(3):
                    values = fitted[target].coefficients[
                        :, coefficient_index, observable
                    ]
                    real_index, imaginary_index, structural_null = _append_feature(
                        columns,
                        members,
                        values,
                        {
                            "omega": omega,
                            "section": "coefficient",
                            "target": target,
                            "coefficient": coefficient_name,
                            "order": int(orders[coefficient_index]),
                            "observable": STATE_NAMES[observable],
                        },
                        null[observable],
                    )
                    groups.append(
                        {
                            "omega": omega,
                            "section": "coefficient",
                            "target": target,
                            "coefficient": coefficient_name,
                            "order": int(orders[coefficient_index]),
                            "observable": STATE_NAMES[observable],
                            "real_index": real_index,
                            "imaginary_index": imaginary_index,
                            "structural_null": bool(structural_null),
                        }
                    )
        for prefix_size in range(3, len(study.strengths) + 1):
            prefix = study.strengths[:prefix_size]
            upper = float(prefix[-1])
            for target, orders in TARGET_ORDERS.items():
                prefix_fit = fit_block_power_series(
                    targets[target][:, :prefix_size], prefix, orders
                )
                null = nulls[target]
                for contribution, coefficient_index, factor in (
                    ("leading", 0, upper ** orders[0]),
                    ("higher", 1, upper ** orders[1]),
                ):
                    for observable in range(3):
                        values = (
                            prefix_fit.coefficients[
                                :, coefficient_index, observable
                            ]
                            * factor
                        )
                        real_index, imaginary_index, structural_null = _append_feature(
                            columns,
                            members,
                            values,
                            {
                                "omega": omega,
                                "section": "adequacy",
                                "target": target,
                                "contribution": contribution,
                                "prefix_upper_strength": upper,
                                "observable": STATE_NAMES[observable],
                            },
                            null[observable],
                        )
                        groups.append(
                            {
                                "omega": omega,
                                "section": "adequacy",
                                "target": target,
                                "contribution": contribution,
                                "prefix_upper_strength": upper,
                                "observable": STATE_NAMES[observable],
                                "real_index": real_index,
                                "imaginary_index": imaginary_index,
                                "structural_null": bool(structural_null),
                            }
                        )
        for contrast, harmonic in HIGHER_HARMONICS:
            for strength_index, strength in enumerate(study.strengths):
                if contrast == "odd":
                    values = odd[:, strength_index, :, index[harmonic]]
                    null = nulls["odd_fundamental"]
                else:
                    values = forced_even[:, strength_index, :, index[harmonic]]
                    null = nulls["even_second_harmonic"]
                for observable in range(3):
                    real_index, imaginary_index, structural_null = _append_feature(
                        columns,
                        members,
                        values[:, observable],
                        {
                            "omega": omega,
                            "section": "harmonic",
                            "contrast": contrast,
                            "harmonic": int(harmonic),
                            "strength": float(strength),
                            "observable": STATE_NAMES[observable],
                        },
                        null[observable],
                    )
                    groups.append(
                        {
                            "omega": omega,
                            "section": "harmonic",
                            "contrast": contrast,
                            "harmonic": int(harmonic),
                            "strength": float(strength),
                            "observable": STATE_NAMES[observable],
                            "real_index": real_index,
                            "imaginary_index": imaginary_index,
                            "structural_null": bool(structural_null),
                        }
                    )
    return np.column_stack(columns), tuple(members), tuple(groups), fits


def _magnitude_bounds(group, lower, upper):
    def coordinate(column_index):
        lo, hi = float(lower[column_index]), float(upper[column_index])
        return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))

    real_lower, real_upper = coordinate(group["real_index"])
    if group["imaginary_index"] is None:
        return real_lower, real_upper
    imaginary_lower, imaginary_upper = coordinate(group["imaginary_index"])
    return float(np.hypot(real_lower, imaginary_lower)), float(
        np.hypot(real_upper, imaginary_upper)
    )


def analyze_x_response(cells: dict, config: dict) -> dict:
    """Run the simultaneous bootstrap and form the detection/adequacy decisions."""
    direction = np.asarray(config["protocol"]["direction"], dtype=float)
    matrix, members, groups, fits = build_x_response_family(cells, direction)
    bootstrap = config["bootstrap"]
    mean, standard_error, statistic, metadata = bootstrap_max_statistic(
        matrix,
        resamples=int(bootstrap["resamples"]),
        root_entropy=bootstrap["root_entropy"],
        batch_size=int(bootstrap.get("batch_size", 64)),
    )
    confidence = float(bootstrap["confidence"])
    critical = float(np.quantile(statistic, confidence, method="higher"))
    lower, upper = mean - critical * standard_error, mean + critical * standard_error
    fraction_limit = float(bootstrap["higher_order_fraction_limit"])

    records = []
    for group in groups:
        magnitude_lower, magnitude_upper = _magnitude_bounds(group, lower, upper)
        record = {
            key: value
            for key, value in group.items()
            if key not in ("real_index", "imaginary_index")
        }
        record.update(
            magnitude_lower=magnitude_lower,
            magnitude_upper=magnitude_upper,
            simultaneous_interval_excludes_origin=magnitude_lower > 0,
        )
        records.append(record)

    def entries(section, **filters):
        return [
            record
            for record in records
            if record["section"] == section
            and all(record.get(key) == value for key, value in filters.items())
        ]

    identification = {}
    for omega in sorted(cells):
        identification[omega] = {}
        for target in TARGET_ORDERS:
            identification[omega][target] = {}
            for strength in cells[omega].study.strengths:
                allowed = [
                    record
                    for record in entries(
                        "identification",
                        omega=omega,
                        target=target,
                        strength=float(strength),
                    )
                    if not record["structural_null"]
                ]
                identification[omega][target][str(float(strength))] = {
                    "identified": any(
                        record["simultaneous_interval_excludes_origin"]
                        for record in allowed
                    ),
                    "components": allowed,
                }

    coefficient_detection = {}
    for omega in sorted(cells):
        coefficient_detection[omega] = {}
        for target in TARGET_ORDERS:
            allowed_higher = [
                record
                for record in entries(
                    "coefficient", omega=omega, target=target, coefficient="higher"
                )
                if not record["structural_null"]
            ]
            allowed_leading = [
                record
                for record in entries(
                    "coefficient", omega=omega, target=target, coefficient="leading"
                )
                if not record["structural_null"]
            ]
            coefficient_detection[omega][target] = {
                "higher_detected": any(
                    record["simultaneous_interval_excludes_origin"]
                    for record in allowed_higher
                ),
                "leading_detected": any(
                    record["simultaneous_interval_excludes_origin"]
                    for record in allowed_leading
                ),
                "higher_components": allowed_higher,
                "leading_components": allowed_leading,
            }

    harmonic_detection = {}
    for omega in sorted(cells):
        harmonic_detection[omega] = {}
        for contrast, harmonic in HIGHER_HARMONICS:
            harmonic_detection[omega][f"{contrast}_n{harmonic}"] = {}
            for strength in cells[omega].study.strengths:
                allowed = [
                    record
                    for record in entries(
                        "harmonic",
                        omega=omega,
                        contrast=contrast,
                        harmonic=harmonic,
                        strength=float(strength),
                    )
                    if not record["structural_null"]
                ]
                harmonic_detection[omega][f"{contrast}_n{harmonic}"][
                    str(float(strength))
                ] = {
                    "detected": any(
                        record["simultaneous_interval_excludes_origin"]
                        for record in allowed
                    ),
                    "components": allowed,
                }

    adequacy = {}
    for omega in sorted(cells):
        adequacy[omega] = {}
        for target in TARGET_ORDERS:
            prefixes = sorted(
                {
                    record["prefix_upper_strength"]
                    for record in entries("adequacy", omega=omega, target=target)
                }
            )
            adequacy[omega][target] = {}
            for prefix in prefixes:
                low = [
                    record
                    for record in entries(
                        "adequacy",
                        omega=omega,
                        target=target,
                        contribution="leading",
                        prefix_upper_strength=prefix,
                    )
                    if not record["structural_null"]
                ]
                high = [
                    record
                    for record in entries(
                        "adequacy",
                        omega=omega,
                        target=target,
                        contribution="higher",
                        prefix_upper_strength=prefix,
                    )
                    if not record["structural_null"]
                ]
                dominant_low_lower = max(
                    (record["magnitude_lower"] for record in low), default=0.0
                )
                maximum_higher_upper = max(
                    (record["magnitude_upper"] for record in high), default=np.inf
                )
                denominator_status = (
                    "simultaneously_resolved" if dominant_low_lower > 0 else "unresolved"
                )
                ratio_upper = (
                    maximum_higher_upper / dominant_low_lower
                    if dominant_low_lower > 0
                    else None
                )
                adequacy[omega][target][str(prefix)] = {
                    "adequate": bool(ratio_upper <= fraction_limit)
                    if ratio_upper is not None
                    else False,
                    "denominator_status": denominator_status,
                    "dominant_low_order_magnitude_lower": dominant_low_lower,
                    "maximum_higher_order_magnitude_upper": maximum_higher_upper,
                    "higher_to_dominant_low_upper": ratio_upper,
                    "fraction_limit": fraction_limit,
                    "rule": (
                        "the simultaneous upper bound on every allowed first-higher-order "
                        "contribution at the prefix upper strength is at most fraction_limit "
                        "times the simultaneous lower bound on the dominant allowed low-order "
                        "contribution"
                    ),
                }

    interpretation_states = {}
    for omega in sorted(cells):
        interpretation_states[omega] = {}
        for target in TARGET_ORDERS:
            leading_detected = coefficient_detection[omega][target]["leading_detected"]
            higher_detected = coefficient_detection[omega][target]["higher_detected"]
            adequate_at_any = any(
                decision["adequate"]
                for decision in adequacy[omega][target].values()
            )
            bound_small_at_any = any(
                decision["higher_to_dominant_low_upper"] is not None
                and decision["higher_to_dominant_low_upper"] <= fraction_limit
                for decision in adequacy[omega][target].values()
            )
            if not leading_detected:
                state = "leading_not_resolved"
            elif bound_small_at_any:
                state = "bounded_small_for_truncation"
            elif higher_detected:
                state = "higher_order_detected_above_tolerance"
            else:
                state = "negligibility_not_established"
            interpretation_states[omega][target] = {
                "state": state,
                "leading_detected": leading_detected,
                "higher_detected": higher_detected,
                "adequate_at_any_prefix": adequate_at_any,
                "note": (
                    "detection of the fitted higher-order coefficient is separate from "
                    "the provisional negligibility bound; a failed bound alone does not "
                    "establish significance"
                ),
            }

    coefficient_summaries = {}
    for omega in sorted(cells):
        coefficient_summaries[omega] = {}
        for target in TARGET_ORDERS:
            coefficient_summaries[omega][target] = {}
            for coefficient in ("leading", "higher"):
                matches = entries(
                    "coefficient",
                    omega=omega,
                    target=target,
                    coefficient=coefficient,
                )
                coefficient_summaries[omega][target][coefficient] = matches

    return {
        "interpretation": (
            "x-direction dense frequency response pilot. Detection of fitted "
            "higher-order coefficients and directly observed higher harmonics is "
            "reported separately from the provisional higher/leading negligibility "
            "bound; a failed bound alone does not establish significance."
        ),
        "coverage": {
            "confidence": confidence,
            "resamples": int(bootstrap["resamples"]),
            "critical_value": critical,
            "quantile_method": "higher",
            "resampling_unit": "one whole initial-state block row shared across frequencies, strengths, targets, coefficients, harmonics, and prefixes",
            "bootstrap": metadata,
        },
        "family": {
            "scalar_coordinate_count": matrix.shape[1],
            "members": members,
            "mean_realified": mean,
            "standard_error_realified": standard_error,
            "simultaneous_lower_realified": lower,
            "simultaneous_upper_realified": upper,
            "row_block_ids": cells[next(iter(cells))].study.block_ids,
            "frequency_major_order": sorted(cells),
        },
        "identification": identification,
        "coefficient_detection": coefficient_detection,
        "harmonic_detection": harmonic_detection,
        "adequacy": adequacy,
        "interpretation_states": interpretation_states,
        "coefficient_summaries": coefficient_summaries,
        "frequencies": sorted(cells),
        "strengths_by_frequency": {
            omega: [float(value) for value in cells[omega].study.strengths]
            for omega in sorted(cells)
        },
    }


def _cell_summary_arrays(cell: XResponseCell, prefix: str) -> dict:
    """Per-frequency summary arrays for the archive and checkpoints.

    The arrays are the derived summaries plus the condition identity and
    sampling grids of one frequency cell; the block-level raw objects live
    in the chunk files.
    """
    arrays = {}
    study = cell.study
    arrays[f"{prefix}_block_ids"] = np.asarray(study.block_ids, dtype=np.uint32)
    arrays[f"{prefix}_strengths"] = study.strengths
    arrays[f"{prefix}_harmonics"] = study.harmonics
    arrays[f"{prefix}_raw_proposals"] = study.raw_proposals
    arrays[f"{prefix}_initial_states"] = study.initial_states
    arrays[f"{prefix}_child_spawn_keys"] = np.asarray(
        study.child_spawn_keys, dtype=np.uint32
    )
    positive_mean = study.positive_cycle_fourier.mean(axis=-2)
    negative_mean = study.negative_cycle_fourier.mean(axis=-2)
    unforced_mean = study.unforced_cycle_fourier.mean(axis=-2)
    arrays[f"{prefix}_positive_cycle_mean"] = positive_mean
    arrays[f"{prefix}_negative_cycle_mean"] = negative_mean
    arrays[f"{prefix}_unforced_cycle_mean"] = unforced_mean
    if cell.cycle_variances is not None:
        arrays[f"{prefix}_positive_cycle_variance"] = cell.cycle_variances["positive"]
        arrays[f"{prefix}_negative_cycle_variance"] = cell.cycle_variances["negative"]
        arrays[f"{prefix}_unforced_cycle_variance"] = cell.cycle_variances["unforced"]
    arrays[f"{prefix}_folded_cycle_mean"] = cell.folded_cycle_mean
    arrays[f"{prefix}_welch_psd"] = cell.welch_psd
    arrays[f"{prefix}_segment_counts"] = cell.segment_counts
    if cell.raw_segments:
        arrays[f"{prefix}_raw_segment_values"] = cell.raw_segments["values"]
        arrays[f"{prefix}_raw_segment_times"] = cell.raw_segments["times"]
    arrays[f"{prefix}_dense_metadata"] = np.asarray(
        json.dumps(cell.dense_metadata)
    )
    record = dict(cell.sampling_metadata or {})
    for key, value in record.items():
        if key.startswith("_"):
            arrays[f"{prefix}{key}"] = np.asarray(value)
    arrays[f"{prefix}_omega"] = np.asarray(float(record.get("omega", cell.omega)))
    arrays[f"{prefix}_forcing_phase"] = np.asarray(record["forcing_phase"])
    arrays[f"{prefix}_discard_time"] = np.asarray(record["discard_time"])
    arrays[f"{prefix}_n_cycle"] = np.asarray(int(record["n_cycle"]))
    arrays[f"{prefix}_n_phase"] = np.asarray(int(record["n_phase"]))
    arrays[f"{prefix}_phase_offset"] = np.asarray(record["phase_offset"])
    arrays[f"{prefix}_dense_dt"] = np.asarray(record["dense_dt"])
    return arrays


def persist_x_response(output_dir, cells: dict, derived: dict, config: dict, provenance):
    """Persist the x-response pilot artifact contract.

    The block is the replication unit.  Derived block-level summaries (cycle
    means, cycle variances, folded cycle means, Welch PSDs, raw segments)
    go into the summary archive; the retained raw objects (per-cycle Fourier
    coefficients, phase-grid samples, complex block spectra, dense
    trajectories) live in the ``block_level/`` chunk files written during
    integration.  Aggregation and contrasts are computed later from the
    retained objects.
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_x_response_summaries.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    metadata_path = output_dir / "condition_metadata.json"
    policy = parse_retention_policy(config, int(config["block_count"]))
    arrays = {}
    sampling_records = {}
    for omega in sorted(cells):
        prefix = f"omega_{frequency_key(omega)}"
        cell_arrays = _cell_summary_arrays(cells[omega], prefix)
        arrays.update(cell_arrays)
        sampling_records[str(float(omega))] = dict(
            cells[omega].sampling_metadata or {}
        )
    write_npz_atomic(raw_path, arrays)
    write_condition_metadata(
        metadata_path,
        config,
        cells[next(iter(cells))].study.block_ids,
        sorted(cells),
        sampling_records,
        policy,
    )
    write_json_atomic(derived_path, derived)
    first_cell = cells[next(iter(cells))]
    manifest = {
        "schema_version": 2,
        "classification": config["classification"],
        "study_id": config["study_id"],
        "files": {
            path.name: f"sha256:{file_sha256(path)}"
            for path in (config_path, raw_path, derived_path, metadata_path)
        },
        "retention_policy": {
            "per_cycle_fourier_blocks": policy.per_cycle_fourier_blocks,
            "phase_samples_blocks": policy.phase_samples_blocks,
            "dense_trajectory_blocks": policy.dense_trajectory_blocks,
            "block_spectra": policy.block_spectra,
            "chunk_blocks": policy.chunk_blocks,
        },
        "array_semantics": {
            "forced_axes": RAW_FORCED_AXES,
            "unforced_axes": RAW_BASELINE_AXES,
            "folded_axes": FOLDED_AXES,
            "psd_axes": PSD_AXES,
            "cycle_variance_axes": CYCLE_VARIANCE_AXES,
            "baseline_variance_axes": BASELINE_VARIANCE_AXES,
            "fourier": "mean over phase times exp(-i*n*theta)",
            "folded_cycle_mean": "phase-conditioned mean on a dense forcing-phase grid from fixed-dt samples",
            "welch_psd": "one-sided Hann-windowed Welch PSD of the residual after removing the folded cycle mean; state^2 per (rad/time)",
            "cycle_variance": "across-cycle sample variance (ddof=1) of real and imaginary cycle Fourier coefficients, part axis [real, imaginary]",
            "condition_order": ["unforced"] + [
                f"{sign}{strength}"
                for strength in first_cell.study.strengths
                for sign in ("+", "-")
            ],
            "raw_contract": (
                "block_level/ holds the retained raw objects: per-cycle Fourier "
                "coefficients, estimator phase-grid samples, complex per-block "
                "physical-frequency spectra S_b(Omega) with the explicit frequency "
                "grid, and optionally complete dense trajectories, chunked by "
                "block.  All arrays in this summary archive are derived from "
                "them and are reproducible from them."
            ),
        },
        "coverage": derived["coverage"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    block_level_root = output_dir / BLOCK_LEVEL_DIRECTORY
    if block_level_root.is_dir():
        manifest["files"].update(
            {
                f"{BLOCK_LEVEL_DIRECTORY}/{path.name}": f"sha256:{file_sha256(path)}"
                for path in sorted(block_level_root.rglob("*.npz"))
            }
        )
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def frequency_chunks_complete(
    block_level_dir, prefix: str, block_count: int, chunk_blocks: int,
    block_id_start: int,
) -> bool:
    """Technical completeness check for one frequency cell's chunk files.

    Returns true only if every expected chunk file exists, loads, and
    carries the exact expected block-id range.  Chunk writes are atomic, so
    existence plus a successful load is the integrity criterion used for
    resuming.  No scientific quantities are inspected.
    """
    directory = Path(block_level_dir)
    if not directory.is_dir():
        return False
    expected = list(chunk_ranges(int(block_count), int(chunk_blocks)))
    for start, end in expected:
        path = directory / chunk_file_name(prefix, start, end)
        if not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as chunk:
                if "block_ids" not in chunk.files:
                    return False
                identifiers = np.asarray(chunk["block_ids"])
                expected_ids = np.arange(
                    int(block_id_start) + start, int(block_id_start) + end,
                    dtype=np.uint32,
                )
                if not np.array_equal(identifiers, expected_ids):
                    return False
                if any(
                    not np.isfinite(np.asarray(chunk[key])).all()
                    for key in chunk.files
                    if np.asarray(chunk[key]).dtype.kind in "fc"
                ):
                    return False
        except (OSError, ValueError):
            return False
    return True


def verify_frequency_chunks(
    block_level_dir, prefix: str, block_count: int, chunk_blocks: int,
    block_id_start: int,
) -> None:
    """Raise unless the technical completeness check passes."""
    if not frequency_chunks_complete(
        block_level_dir, prefix, block_count, chunk_blocks, block_id_start
    ):
        raise ValueError(
            f"chunk integrity check failed for {prefix} in {block_level_dir}"
        )


def run_x_response_pilot(config_path, resume_dir=None):
    """Run the configured integration, analysis, and persistence workflow.

    With ``resume_dir`` pointing at an existing partially written artifact
    directory, frequencies whose chunk files already pass the technical
    completeness check are reused as production data and skipped; the
    interrupted frequency and all remaining frequencies are integrated
    normally, and the summary/derived artifacts are rewritten at the end.
    """
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_x_response_config(config)
    config_identifier = f"sha256:{file_sha256(config_path)}"
    if resume_dir is not None and Path(resume_dir).is_dir():
        output_dir = Path(resume_dir)
        print(f"[{time.strftime('%H:%M:%S')}] resuming artifact {output_dir}")
    else:
        output_dir = Path(__file__).resolve().parents[2] / config["output_root"] / (
            f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{config_identifier[7:19]}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(output_dir / "config_snapshot.json", config)
    policy = checked["retention_policy"]
    block_id_start = int(config["initial_ensemble"]["block_id_start"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    blocks = generate_production_blocks(config)
    print(
        f"[{time.strftime('%H:%M:%S')}] initial-state blocks generated "
        f"({len(blocks.block_ids)} blocks)",
        flush=True,
    )
    cells = {}
    for omega in checked["frequencies"]:
        prefix = f"omega_{frequency_key(omega)}"
        chunk_dir = output_dir / BLOCK_LEVEL_DIRECTORY / prefix
        if resume_dir is not None and frequency_chunks_complete(
            chunk_dir, prefix, checked["block_count"], policy.chunk_blocks,
            block_id_start,
        ):
            checkpoint_path = checkpoint_dir / f"{prefix}.npz"
            if not checkpoint_path.is_file():
                raise ValueError(
                    f"resume found complete chunks for omega={omega} but no "
                    f"checkpoint at {checkpoint_path}"
                )
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                cells[omega] = _load_cell(
                    checkpoint, prefix, omega, config, int(config["n_phase"])
                )
            print(
                f"[{time.strftime('%H:%M:%S')}] omega={omega} chunks already "
                "complete and verified; reusing production data",
                flush=True,
            )
            continue
        cell = generate_x_response_cell(
            config,
            omega,
            checked["strengths"],
            checked["cycles"][omega],
            block_level_dir=chunk_dir,
            precomputed_blocks=blocks,
        )
        verify_frequency_chunks(
            chunk_dir, prefix, checked["block_count"], policy.chunk_blocks,
            block_id_start,
        )
        write_npz_atomic(
            checkpoint_dir / f"{prefix}.npz", _cell_summary_arrays(cell, prefix)
        )
        cells[omega] = cell
        print(
            f"[{time.strftime('%H:%M:%S')}] omega={omega} integrated in "
            f"{cell.runtime_seconds:.1f}s; chunk integrity verified",
            flush=True,
        )
    derived = analyze_x_response(cells, config)
    provenance = {
        "config_identifier": config_identifier,
        "code_identifiers": active_source_identifiers(
            Path(__file__).resolve().parents[2], config["runner_path"]
        ),
        "git": git_provenance(Path(__file__).resolve().parents[2]),
        "environment": environment_provenance(),
        "runtime_seconds_before_persistence": time.perf_counter() - started,
        "resumed_from": str(resume_dir) if resume_dir is not None else None,
    }
    manifest = persist_x_response(output_dir, cells, derived, config, provenance)
    return output_dir, manifest, time.perf_counter() - started
