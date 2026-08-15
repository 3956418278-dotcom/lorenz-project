"""Block-level retention contract for response experiments.

The integration and persistence paths produce raw block-level objects
(phase-grid samples, per-cycle Fourier coefficients, per-block complex
physical-frequency spectra, dense trajectories) plus derived summaries.
This module owns the policy that decides which block-level objects are
retained, the condition-identity records that let the same block be paired
across ``+h``/``-h``/strength/frequency/direction/unforced conditions, and
the chunked on-disk layout of the retained objects.

Scientific array meanings and experiment interpretation stay with the
experiment modules; this module owns only the retention decisions and the
serialization layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .artifacts import write_json_atomic, write_npz_atomic

OBSERVABLE_LABELS = ("x", "y", "z")

# Axes of each retained block-level object.
PHASE_SAMPLE_AXES = ("block", "condition", "cycle", "phase", "state")
CYCLE_FOURIER_AXES = ("block", "condition", "cycle", "state", "harmonic")
BLOCK_SPECTRUM_AXES = ("block", "condition", "state", "frequency_bin")
DENSE_TRAJECTORY_AXES = ("block", "condition", "state", "time")


@dataclass(frozen=True)
class RetentionPolicy:
    """Resolved retention decisions for one experiment run.

    Each ``*_blocks`` field is a concrete block count (``0`` means nothing is
    retained; ``block_count`` means every block).  ``block_spectra`` controls
    the complex per-block physical-frequency spectra, which have no legacy
    equivalent.  ``chunk_blocks`` is the block-chunk size used for the
    chunked on-disk layout.
    """

    per_cycle_fourier_blocks: int
    phase_samples_blocks: int
    dense_trajectory_blocks: int
    block_spectra: bool
    chunk_blocks: int


def _resolved_blocks(value, block_count: int, *, default: int) -> int:
    """Resolve an explicit retention block count, ``"all"``, or ``None``."""
    if value is None:
        return int(default)
    if isinstance(value, str):
        if value == "all":
            return int(block_count)
        raise ValueError("retention block counts must be 'all', an integer, or null")
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("retention block counts must be 'all', an integer, or null")
    resolved = int(value)
    if resolved < 0 or resolved > block_count:
        raise ValueError("retention block counts must lie within the block axis")
    return resolved


def parse_retention_policy(config: dict, block_count: int) -> RetentionPolicy:
    """Parse the ``retention`` configuration surface into concrete decisions.

    Defaults implement the retained-data contract: per-cycle Fourier
    coefficients and the estimator's phase-grid samples are retained for
    every block unless the configuration explicitly reduces them.  Dense
    trajectory retention defaults to nothing and must be set explicitly by a
    production configuration that has seen the storage-scaling report.
    """
    retention = config.get("retention") or {}
    if not isinstance(retention, dict):
        raise ValueError("retention must be a mapping")
    block_count = int(block_count)
    chunk_blocks = int(retention.get("chunk_blocks", 8))
    if isinstance(chunk_blocks, bool) or chunk_blocks < 1:
        raise ValueError("retention.chunk_blocks must be a positive integer")
    return RetentionPolicy(
        per_cycle_fourier_blocks=_resolved_blocks(
            retention.get("per_cycle_fourier_blocks", "all"),
            block_count,
            default=block_count,
        ),
        phase_samples_blocks=_resolved_blocks(
            retention.get("phase_samples_blocks", "all"),
            block_count,
            default=block_count,
        ),
        dense_trajectory_blocks=_resolved_blocks(
            retention.get("dense_trajectory_blocks"), block_count, default=0
        ),
        block_spectra=bool(retention.get("block_spectra", True)),
        chunk_blocks=chunk_blocks,
    )


def condition_table(forcing_vectors) -> dict[str, np.ndarray]:
    """Build identity arrays aligned with a condition axis.

    ``forcing_vectors`` has axes ``condition, state``.  The first condition
    is conventionally the unforced reference.  The returned arrays allow
    machine pairing of the same block across sign, strength, frequency,
    direction, and unforced reference without reconstructing identity from
    filenames.
    """
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
    if (
        forcing_vectors.ndim != 2
        or forcing_vectors.shape[0] < 1
        or forcing_vectors.shape[1] != 3
        or not np.isfinite(forcing_vectors).all()
    ):
        raise ValueError("forcing_vectors must have finite axes condition,state")
    norms = np.linalg.norm(forcing_vectors, axis=1)
    reference = None
    for forcing in forcing_vectors[1:]:
        if float(np.linalg.norm(forcing)) > 0:
            reference = forcing
            break
    if reference is None:
        sign = np.zeros(len(forcing_vectors), dtype=np.int8)
        signed_strength = np.zeros(len(forcing_vectors), dtype=float)
    else:
        projection = forcing_vectors @ reference / float(
            np.linalg.norm(reference)
        )
        collinear = np.isclose(np.abs(projection), norms)
        sign = np.where(
            collinear & (norms > 0), np.sign(projection), 0
        ).astype(np.int8)
        # Forced conditions not collinear with the reference direction carry
        # a canonical positive strength; the direction array identifies them.
        signed_strength = np.where(
            sign == 0, np.where(norms > 0, norms, 0.0), sign * norms
        )
    directions = np.zeros_like(forcing_vectors)
    nonzero = norms > 0
    directions[nonzero] = forcing_vectors[nonzero] / norms[nonzero, None]
    return {
        "sign": sign,
        "signed_strength": signed_strength,
        "direction": directions,
    }


def condition_labels(forcing_vectors) -> tuple[str, ...]:
    """Human-readable labels for one condition axis."""
    table = condition_table(forcing_vectors)
    labels = ["unforced"]
    for index in range(1, len(forcing_vectors)):
        signed = float(table["signed_strength"][index])
        labels.append(f"{'+' if signed > 0 else '-'}{abs(signed):g}")
    return tuple(labels)


def chunk_ranges(block_count: int, chunk_blocks: int):
    """Yield ``(start, end)`` block index ranges covering the block axis."""
    block_count = int(block_count)
    chunk_blocks = int(chunk_blocks)
    if block_count < 1 or chunk_blocks < 1:
        raise ValueError("block counts must be positive")
    start = 0
    while start < block_count:
        end = min(start + chunk_blocks, block_count)
        yield start, end
        start = end


def chunk_file_name(prefix: str, start: int, end: int) -> str:
    """Stable chunk file name for one block range of one frequency cell."""
    return f"{prefix}_blocks_{start:06d}_{end:06d}.npz"


def write_block_level_chunk(path: str | os.PathLike[str], arrays) -> None:
    """Write one block-chunk archive of retained block-level objects."""
    write_npz_atomic(path, arrays)


def write_condition_metadata(
    path: str | os.PathLike[str],
    config: dict,
    block_ids,
    frequencies,
    sampling_records: dict,
    policy: RetentionPolicy,
) -> None:
    """Persist the structured condition-identity and sampling record.

    This file is the machine-readable answer to "which condition is which":
    observable labels, the condition table, the sampling grids, the solver
    and spinup settings, and the retention decisions that were applied.
    """
    records = []
    for omega in sorted(frequencies):
        record = {
            key: value
            for key, value in sampling_records[str(float(omega))].items()
            if not key.startswith("_")
        }
        record["omega"] = float(omega)
        records.append(record)
    write_json_atomic(
        path,
        {
            "observable_labels": list(OBSERVABLE_LABELS),
            "block_ids": [int(value) for value in block_ids],
            "condition_order": (
                "condition axis index 0 is the unforced reference; remaining "
                "indices are the forced conditions in the order built by the "
                "runner"
            ),
            "lorenz": config["lorenz"],
            "solver": config["solver"],
            "initial_ensemble": config["initial_ensemble"],
            "protocol": config["protocol"],
            "sampling_records": records,
            "retention_policy": {
                "per_cycle_fourier_blocks": policy.per_cycle_fourier_blocks,
                "phase_samples_blocks": policy.phase_samples_blocks,
                "dense_trajectory_blocks": policy.dense_trajectory_blocks,
                "block_spectra": policy.block_spectra,
                "chunk_blocks": policy.chunk_blocks,
            },
            "axes": {
                "phase_samples": PHASE_SAMPLE_AXES,
                "cycle_fourier": CYCLE_FOURIER_AXES,
                "block_spectrum": BLOCK_SPECTRUM_AXES,
                "dense_trajectory": DENSE_TRAJECTORY_AXES,
            },
        },
    )
