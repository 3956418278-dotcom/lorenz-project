"""x-direction dense frequency response pilot.

Composes the shared sampling, fitting, bootstrap, and persistence machinery
for the result-stage x-direction experiment:

- one uniform dense frequency grid for the first-order landscape (c1),
- a smaller predeclared higher-budget strength set for the second-order
  coefficients (c2) and the validity-boundary evidence,
- block-level dense spectral summaries (folded cycle mean, Welch PSD of the
  residual, small raw segments) alongside the inferential Fourier harmonics.

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
from .direction_design import LORENZ_PARITY
from .ensemble import SymmetricXYUniformProposal, generate_initial_state_blocks
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


@dataclass(frozen=True)
class XResponseCell:
    """One frequency cell with block-level Fourier and dense summaries."""

    omega: float
    study: StrengthStudyData
    folded_cycle_mean: np.ndarray
    welch_psd: np.ndarray
    segment_counts: np.ndarray
    raw_segments: dict
    dense_metadata: dict
    runtime_seconds: float


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
    return {
        "frequencies": frequencies,
        "strengths": strengths,
        "block_count": block_count,
        "cycles": {
            omega: minimum_duration_cycle_count(omega, minimum_cycles, minimum_time)
            for omega in frequencies
        },
        "direction": direction,
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


def generate_x_response_cell(
    config: dict, omega: float, strengths, cycles: int
) -> XResponseCell:
    """Integrate one frequency cell with paired forcing and dense summaries."""
    started = time.perf_counter()
    block_count = int(config["block_count"])
    workers = int(config.get("workers", 1))
    initial = config["initial_ensemble"]
    block_id_start = int(initial["block_id_start"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    proposal_cfg = initial["proposal"]
    proposal = SymmetricXYUniformProposal(
        x_half_width=proposal_cfg["x_half_width"],
        y_half_width=proposal_cfg["y_half_width"],
        z_bounds=tuple(proposal_cfg["z_bounds"]),
    )
    numerical_config = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["solver"]),
    }
    blocks = generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        proposal,
        float(initial["spinup_time"]),
        numerical_config,
    )
    protocol = config["protocol"]
    direction = np.asarray(protocol["direction"], dtype=float)
    strengths = _checked_strengths(strengths, minimum_count=2)
    forcing_vectors = [np.zeros(3)]
    forcing_vectors.extend(
        sign * strength * direction
        for strength in strengths
        for sign in (1.0, -1.0)
    )
    sampling = CycleFourierSampling(
        omega=omega,
        phase=float(protocol.get("phase", 0.0)),
        discard_time=float(config["discard_time"]),
        n_cycle=cycles,
        n_phase=int(config["n_phase"]),
        harmonics=np.asarray(config["harmonics"], dtype=int),
    )
    dense_config = dict(config["dense"])
    fourier, summaries = integrate_phase_and_dense_conditions(
        blocks.block_ids,
        blocks.final_states,
        forcing_vectors,
        sampling,
        dense_config,
        numerical_config,
        workers=workers,
        raw_segment_blocks=int(dense_config.get("raw_segment_blocks", 0)),
    )
    condition_count = 1 + 2 * len(strengths)
    unforced = fourier[:, 0]
    paired = fourier[:, 1:].reshape(
        block_count,
        len(strengths),
        2,
        3,
        cycles,
        len(sampling.harmonics),
    )
    folded, psd, segment_counts, raw_segments, dense_metadata = (
        _dense_summary_arrays(summaries, dense_config, condition_count, block_count)
    )
    study = StrengthStudyData(
        block_ids=blocks.block_ids,
        strengths=strengths,
        harmonics=sampling.harmonics,
        omega=omega,
        n_phase=int(config["n_phase"]),
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
    return XResponseCell(
        omega=omega,
        study=study,
        folded_cycle_mean=folded,
        welch_psd=psd,
        segment_counts=segment_counts,
        raw_segments=raw_segments,
        dense_metadata=dense_metadata,
        runtime_seconds=time.perf_counter() - started,
    )


def load_x_response_cells(raw_path, n_phase: int = 32) -> dict:
    """Reconstruct per-frequency cells from a persisted raw summary artifact.

    Block-level cycle means are sufficient for the analysis family; the
    per-cycle axis is restored with a single slot.
    """
    raw = np.load(raw_path, allow_pickle=False)
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
                n_phase=n_phase,
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
            folded = (
                np.asarray(raw[f"{prefix}_folded_cycle_mean"])
                if f"{prefix}_folded_cycle_mean" in raw.files
                else None
            )
            cells[omega] = XResponseCell(
                omega=omega,
                study=study,
                folded_cycle_mean=folded,
                welch_psd=np.asarray(raw[f"{prefix}_welch_psd"])
                if f"{prefix}_welch_psd" in raw.files
                else None,
                segment_counts=np.asarray(raw[f"{prefix}_segment_counts"])
                if f"{prefix}_segment_counts" in raw.files
                else None,
                raw_segments={},
                dense_metadata={},
                runtime_seconds=0.0,
            )
    finally:
        raw.close()
    return cells


def merge_cell(base: XResponseCell, extra: XResponseCell) -> XResponseCell:
    """Merge a supplementary strength cell into a base cell of the same frequency.

    Shared strengths must agree exactly (deterministic integration), so the
    merged cell simply concatenates the extra strengths onto the base study.
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
    base_shape = base.study.positive_cycle_fourier.shape
    merged_shape = (len(combined),) + base_shape[0:1] + base_shape[2:]
    positive = np.empty(merged_shape, dtype=base.study.positive_cycle_fourier.dtype)
    negative = np.empty_like(positive)
    for strength in combined:
        index = order[float(strength)]
        if strength in base.study.strengths:
            b_index = list(base.study.strengths).index(strength)
            positive[index] = base.study.positive_cycle_fourier[:, b_index]
            negative[index] = base.study.negative_cycle_fourier[:, b_index]
        else:
            e_index = list(extra.study.strengths).index(strength)
            positive[index] = extra.study.positive_cycle_fourier[:, e_index]
            negative[index] = extra.study.negative_cycle_fourier[:, e_index]
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
    return XResponseCell(
        omega=base.omega,
        study=study,
        folded_cycle_mean=base.folded_cycle_mean,
        welch_psd=base.welch_psd,
        segment_counts=base.segment_counts,
        raw_segments=base.raw_segments,
        dense_metadata=base.dense_metadata,
        runtime_seconds=base.runtime_seconds + extra.runtime_seconds,
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


def persist_x_response(output_dir, cells: dict, derived: dict, config: dict, provenance):
    """Persist the x-response pilot artifact contract.

    The block is the replication unit: the full grid stores block-level
    cycle-mean Fourier coefficients and across-cycle variance summaries.
    Full per-cycle arrays are retained only for the configured small
    representative subset (cycle-resolved diagnostics).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_x_response_summaries.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    write_json_atomic(config_path, config)
    subset = config.get("retention", {}).get("per_cycle_subset")
    subset_omega = float(subset["omega"]) if subset else None
    subset_blocks = int(subset.get("block_count", 0)) if subset else 0
    arrays = {}
    for omega in sorted(cells):
        cell = cells[omega]
        study = cell.study
        prefix = f"omega_{frequency_key(omega)}"
        arrays[f"{prefix}_block_ids"] = np.asarray(study.block_ids, dtype=np.uint32)
        arrays[f"{prefix}_strengths"] = study.strengths
        arrays[f"{prefix}_harmonics"] = study.harmonics
        arrays[f"{prefix}_raw_proposals"] = study.raw_proposals
        arrays[f"{prefix}_initial_states"] = study.initial_states
        arrays[f"{prefix}_child_spawn_keys"] = np.asarray(
            study.child_spawn_keys, dtype=np.uint32
        )
        if subset_omega is not None and float(omega) == subset_omega:
            arrays[f"{prefix}_positive_cycle_fourier_subset"] = (
                study.positive_cycle_fourier[:subset_blocks]
            )
            arrays[f"{prefix}_negative_cycle_fourier_subset"] = (
                study.negative_cycle_fourier[:subset_blocks]
            )
            arrays[f"{prefix}_unforced_cycle_fourier_subset"] = (
                study.unforced_cycle_fourier[:subset_blocks]
            )
        positive_mean = study.positive_cycle_fourier.mean(axis=-2)
        negative_mean = study.negative_cycle_fourier.mean(axis=-2)
        unforced_mean = study.unforced_cycle_fourier.mean(axis=-2)
        arrays[f"{prefix}_positive_cycle_mean"] = positive_mean
        arrays[f"{prefix}_negative_cycle_mean"] = negative_mean
        arrays[f"{prefix}_unforced_cycle_mean"] = unforced_mean
        arrays[f"{prefix}_positive_cycle_variance"] = np.stack(
            [
                study.positive_cycle_fourier.real.var(axis=-2, ddof=1),
                study.positive_cycle_fourier.imag.var(axis=-2, ddof=1),
            ],
            axis=-1,
        )
        arrays[f"{prefix}_negative_cycle_variance"] = np.stack(
            [
                study.negative_cycle_fourier.real.var(axis=-2, ddof=1),
                study.negative_cycle_fourier.imag.var(axis=-2, ddof=1),
            ],
            axis=-1,
        )
        arrays[f"{prefix}_unforced_cycle_variance"] = np.stack(
            [
                study.unforced_cycle_fourier.real.var(axis=-2, ddof=1),
                study.unforced_cycle_fourier.imag.var(axis=-2, ddof=1),
            ],
            axis=-1,
        )
        arrays[f"{prefix}_folded_cycle_mean"] = cell.folded_cycle_mean
        arrays[f"{prefix}_welch_psd"] = cell.welch_psd
        arrays[f"{prefix}_segment_counts"] = cell.segment_counts
        if cell.raw_segments:
            arrays[f"{prefix}_raw_segment_values"] = cell.raw_segments["values"]
            arrays[f"{prefix}_raw_segment_times"] = cell.raw_segments["times"]
        arrays[f"{prefix}_dense_metadata"] = np.asarray(
            json.dumps(cell.dense_metadata)
        )
    write_npz_atomic(raw_path, arrays)
    write_json_atomic(derived_path, derived)
    manifest = {
        "schema_version": 1,
        "classification": config["classification"],
        "study_id": config["study_id"],
        "files": {
            path.name: f"sha256:{file_sha256(path)}"
            for path in (config_path, raw_path, derived_path)
        },
        "array_semantics": {
            "forced_axes": RAW_FORCED_AXES,
            "unforced_axes": RAW_BASELINE_AXES,
            "folded_axes": FOLDED_AXES,
            "psd_axes": PSD_AXES,
            "fourier": "mean over phase times exp(-i*n*theta)",
            "folded_cycle_mean": "phase-conditioned mean on a dense forcing-phase grid from fixed-dt samples",
            "welch_psd": "one-sided Hann-windowed Welch PSD of the residual after removing the folded cycle mean; state^2 per (rad/time)",
            "condition_order": ["unforced"] + [
                f"{sign}{strength}"
                for strength in cells[next(iter(cells))].study.strengths
                for sign in ("+", "-")
            ],
        },
        "coverage": derived["coverage"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def run_x_response_pilot(config_path):
    """Run the configured integration, analysis, and persistence workflow."""
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_x_response_config(config)
    cells = {}
    for omega in checked["frequencies"]:
        cell = generate_x_response_cell(
            config, omega, checked["strengths"], checked["cycles"][omega]
        )
        cells[omega] = cell
        print(
            f"[{time.strftime('%H:%M:%S')}] omega={omega} integrated in "
            f"{cell.runtime_seconds:.1f}s",
            flush=True,
        )
    derived = analyze_x_response(cells, config)
    config_identifier = f"sha256:{file_sha256(config_path)}"
    output_dir = Path(__file__).resolve().parents[2] / config["output_root"] / (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{config_identifier[7:19]}"
    )
    provenance = {
        "config_identifier": config_identifier,
        "code_identifiers": active_source_identifiers(
            Path(__file__).resolve().parents[2], config["runner_path"]
        ),
        "git": git_provenance(Path(__file__).resolve().parents[2]),
        "environment": environment_provenance(),
        "runtime_seconds_before_persistence": time.perf_counter() - started,
    }
    manifest = persist_x_response(output_dir, cells, derived, config, provenance)
    return output_dir, manifest, time.perf_counter() - started
