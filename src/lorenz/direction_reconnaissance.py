"""Minimal crossed forcing-direction and frequency reconnaissance.

The experiment reuses one validated held-out forcing-direction row and integrates
only the missing design rows. Numerical sampling, block generation, Fourier
summaries, strength fits, bootstrap mechanics, and persistence remain owned by
their existing public modules.
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
    verify_file_identifiers,
    write_json_atomic,
    write_npz_atomic,
)
from .direction_design import (
    INPUT_PAIR_ORDER,
    LORENZ_PARITY,
    DirectionDesign,
    build_direction_design,
    lorenz_effective_quadratic_sector_mask,
    lorenz_linear_sector_mask,
)
from .statistics import BlockFrequencyResponses, reconstruct_block_tensors
from .strength_bootstrap import bootstrap_max_statistic
from .strength_series import TARGET_ORDERS, fit_block_power_series
from .strength_study import (
    CycleFourierSampling,
    STATE_NAMES,
    frequency_key,
    integrate_cycle_fourier_conditions,
    validate_frequency_sampling_config,
)


RAW_FORCED_AXES = ("block", "direction", "strength", "state", "cycle", "harmonic")
RAW_BASELINE_AXES = ("block", "state", "cycle", "harmonic")


@dataclass(frozen=True)
class DirectionFrequencyData:
    """Crossed direction summaries with one unforced baseline per frequency."""

    block_ids: tuple[int, ...]
    strengths: np.ndarray
    harmonics: np.ndarray
    frequencies: tuple[float, ...]
    design: DirectionDesign
    positive_cycle_fourier: dict[float, np.ndarray]
    negative_cycle_fourier: dict[float, np.ndarray]
    unforced_cycle_fourier: dict[float, np.ndarray]
    raw_proposals: np.ndarray
    initial_states: np.ndarray
    child_spawn_keys: tuple[tuple[int, ...], ...]
    source: dict[float, dict]

    def __post_init__(self):
        n_block = len(self.block_ids)
        expected_leading = (
            n_block,
            len(self.design.direction_names),
            len(self.strengths),
            3,
        )
        if n_block < 2 or len(set(self.block_ids)) != n_block:
            raise ValueError("block_ids must contain at least two unique blocks")
        if set(self.positive_cycle_fourier) != set(self.frequencies):
            raise ValueError("positive summaries must match frequencies")
        if set(self.negative_cycle_fourier) != set(self.frequencies):
            raise ValueError("negative summaries must match frequencies")
        if set(self.unforced_cycle_fourier) != set(self.frequencies):
            raise ValueError("unforced summaries must match frequencies")
        if np.asarray(self.raw_proposals).shape != (n_block, 3):
            raise ValueError("raw proposals must have axes block,state")
        if np.asarray(self.initial_states).shape != (n_block, 3):
            raise ValueError("initial states must have axes block,state")
        if len(self.child_spawn_keys) != n_block:
            raise ValueError("spawn keys must match the block axis")
        for omega in self.frequencies:
            positive = np.asarray(self.positive_cycle_fourier[omega])
            negative = np.asarray(self.negative_cycle_fourier[omega])
            baseline = np.asarray(self.unforced_cycle_fourier[omega])
            if positive.shape != negative.shape or positive.shape[:4] != expected_leading:
                raise ValueError("forced summaries require block,direction,strength,state axes")
            if positive.ndim != 6 or positive.shape[-1] != len(self.harmonics):
                raise ValueError("forced summaries require cycle,harmonic trailing axes")
            if baseline.shape != (
                n_block,
                3,
                positive.shape[-2],
                len(self.harmonics),
            ):
                raise ValueError("one baseline must have block,state,cycle,harmonic axes")
            if not all(np.isfinite(value).all() for value in (positive, negative, baseline)):
                raise ValueError("Fourier summaries must be finite")


@dataclass(frozen=True)
class DirectionTensorFits:
    """Leading and first higher-order response tensors fitted within blocks."""

    leading: BlockFrequencyResponses
    higher_coefficient: BlockFrequencyResponses
    higher_at_max_strength: BlockFrequencyResponses
    directional_targets: dict[str, np.ndarray]
    unforced_second_harmonic: np.ndarray


def validate_direction_reconnaissance_config(config: dict) -> dict:
    """Validate shared sampling fields and the configured direction design."""
    checked = validate_frequency_sampling_config(config)
    design_config = config["direction_design"]
    if not isinstance(design_config, dict):
        raise ValueError("direction_design must contain name, names, and vectors")
    design = build_direction_design(
        design_config["name"],
        design_config["direction_names"],
        design_config["directions"],
    )
    if design.linear_rank != 3 or design.quadratic_rank != 6:
        raise ValueError("direction design must have full linear and quadratic rank")
    parent = config["reuse_heldout_direction"]
    index = parent.get("direction_index")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError("reused direction index must be an integer")
    if not 0 <= int(index) < len(design.direction_names):
        raise ValueError("reused direction index is outside the configured design")
    if parent.get("direction_name") != design.direction_names[int(index)]:
        raise ValueError("reused direction name does not match its configured design row")
    return {**checked, "design": design, "reused_direction_index": int(index)}


def _validated_parent(repo_root: Path, config: dict, checked: dict):
    parent_spec = config["reuse_heldout_direction"]
    parent_dir = repo_root / parent_spec["artifact"]
    manifest_path = parent_dir / "manifest.json"
    if file_sha256(manifest_path) != parent_spec["manifest_sha256"]:
        raise ValueError("held-out parent manifest hash does not match")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_file_identifiers(parent_dir, manifest["files"])
    parent_config = json.loads(
        (parent_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    for field in ("lorenz", "solver", "initial_ensemble", "discard_time", "n_phase"):
        if parent_config[field] != config[field]:
            raise ValueError(f"held-out parent {field} does not match")
    np.testing.assert_array_equal(parent_config["strengths"], checked["strengths"])
    np.testing.assert_array_equal(parent_config["harmonics"], checked["harmonics"])
    expected_direction = checked["design"].directions[
        checked["reused_direction_index"]
    ]
    if not np.array_equal(
        np.asarray(parent_config["protocol"]["direction"], dtype=float),
        expected_direction,
    ):
        raise ValueError("held-out parent direction does not match the configured design row")
    if float(parent_config["protocol"].get("phase", 0.0)) != float(
        config["protocol"].get("phase", 0.0)
    ):
        raise ValueError("held-out parent forcing phase does not match")
    if int(parent_config["block_count"]) != checked["block_count"]:
        raise ValueError("held-out parent block count does not match")
    raw_path = parent_dir / "raw_frequency_summaries.npz"
    raw = np.load(raw_path, allow_pickle=False)
    required = {"block_ids", "strengths", "harmonics", "frequencies"}
    for omega in checked["frequencies"]:
        prefix = f"omega_{frequency_key(omega)}"
        required.update(
            f"{prefix}_{suffix}"
            for suffix in (
                "positive_cycle_fourier",
                "negative_cycle_fourier",
                "unforced_cycle_fourier",
                "raw_proposals",
                "initial_states",
                "child_spawn_keys",
            )
        )
    if not required.issubset(raw.files):
        raw.close()
        raise ValueError("held-out parent lacks required raw arrays")
    try:
        np.testing.assert_array_equal(raw["strengths"], checked["strengths"])
        np.testing.assert_array_equal(raw["harmonics"], checked["harmonics"])
        if not set(checked["frequencies"]).issubset(set(raw["frequencies"])):
            raise ValueError("held-out parent lacks a configured frequency")
    except Exception:
        raw.close()
        raise
    return manifest, raw


def generate_direction_reconnaissance(config: dict, repo_root: Path) -> DirectionFrequencyData:
    """Reuse one parent row and integrate all other configured directions."""
    checked = validate_direction_reconnaissance_config(config)
    design = checked["design"]
    reused_index = checked["reused_direction_index"]
    missing_indices = tuple(
        index for index in range(len(design.directions)) if index != reused_index
    )
    manifest, raw = _validated_parent(repo_root, config, checked)
    positive = {}
    negative = {}
    unforced = {}
    source = {}
    reference = None
    try:
        for omega in checked["frequencies"]:
            prefix = f"omega_{frequency_key(omega)}"
            block_ids = tuple(int(value) for value in raw["block_ids"])
            proposals = np.asarray(raw[f"{prefix}_raw_proposals"])
            states = np.asarray(raw[f"{prefix}_initial_states"])
            spawn_keys = tuple(
                tuple(int(item) for item in row)
                for row in raw[f"{prefix}_child_spawn_keys"]
            )
            initial = config["initial_ensemble"]
            expected_block_ids = tuple(
                range(
                    int(initial["block_id_start"]),
                    int(initial["block_id_start"]) + checked["block_count"],
                )
            )
            if block_ids != expected_block_ids:
                raise ValueError("held-out parent block IDs/order do not match the design")
            identity = (block_ids, proposals, states, spawn_keys)
            if reference is None:
                reference = identity
            else:
                if block_ids != reference[0] or spawn_keys != reference[3]:
                    raise ValueError("held-out parent block IDs/order or spawn keys differ by frequency")
                if not np.array_equal(proposals, reference[1]):
                    raise ValueError("held-out parent proposals differ by frequency")
                if not np.array_equal(states, reference[2]):
                    raise ValueError("held-out parent post-spinup states differ by frequency")

            parent_positive = np.asarray(raw[f"{prefix}_positive_cycle_fourier"])
            parent_negative = np.asarray(raw[f"{prefix}_negative_cycle_fourier"])
            baseline = np.asarray(raw[f"{prefix}_unforced_cycle_fourier"])
            cycles = checked["cycles"][omega]
            expected = (
                checked["block_count"],
                len(checked["strengths"]),
                3,
                cycles,
                len(checked["harmonics"]),
            )
            if parent_positive.shape != expected or parent_negative.shape != expected:
                raise ValueError("held-out parent forced axes do not match the design")
            if baseline.shape != (expected[0], expected[2], expected[3], expected[4]):
                raise ValueError("held-out baseline axes do not match the design")

            missing_directions = design.directions[list(missing_indices)]
            forcing_vectors = np.asarray(
                [
                    sign * strength * direction
                    for direction in missing_directions
                    for strength in checked["strengths"]
                    for sign in (1.0, -1.0)
                ]
            )
            sampling = CycleFourierSampling(
                omega=omega,
                phase=float(config["protocol"].get("phase", 0.0)),
                discard_time=float(config["discard_time"]),
                n_cycle=cycles,
                n_phase=int(config["n_phase"]),
                harmonics=checked["harmonics"],
            )
            started = time.perf_counter()
            integrated = integrate_cycle_fourier_conditions(
                block_ids,
                states,
                forcing_vectors,
                sampling,
                {"lorenz": dict(config["lorenz"]), "solver": dict(config["solver"])},
                workers=int(config.get("workers", 1)),
            )
            elapsed = time.perf_counter() - started
            missing = integrated.reshape(
                len(block_ids),
                len(missing_directions),
                len(checked["strengths"]),
                2,
                3,
                cycles,
                len(checked["harmonics"]),
            )
            assembled_shape = (
                len(block_ids),
                len(design.directions),
                len(checked["strengths"]),
                3,
                cycles,
                len(checked["harmonics"]),
            )
            positive[omega] = np.empty(assembled_shape, dtype=integrated.dtype)
            negative[omega] = np.empty(assembled_shape, dtype=integrated.dtype)
            positive[omega][:, reused_index] = parent_positive
            negative[omega][:, reused_index] = parent_negative
            positive[omega][:, missing_indices] = missing[:, :, :, 0]
            negative[omega][:, missing_indices] = missing[:, :, :, 1]
            unforced[omega] = baseline.copy()
            source[omega] = {
                "reused_direction": {
                    "index": reused_index,
                    "name": design.direction_names[reused_index],
                    "mode": "validated_heldout_parent",
                },
                "other_directions": "new_integration",
                "baseline": "validated_heldout_parent_reused_once",
                "new_runtime_seconds": elapsed,
                "parent_artifact": config["reuse_heldout_direction"]["artifact"],
                "parent_raw_identifier": manifest["files"]["raw_frequency_summaries.npz"],
            }
    finally:
        raw.close()
    assert reference is not None
    return DirectionFrequencyData(
        block_ids=reference[0],
        strengths=checked["strengths"],
        harmonics=checked["harmonics"],
        frequencies=checked["frequencies"],
        design=design,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=reference[1].copy(),
        initial_states=reference[2].copy(),
        child_spawn_keys=reference[3],
        source=source,
    )


def direction_parity_null_masks(directions) -> dict[str, np.ndarray]:
    """Return target null-output masks implied for parity-eigenvector directions."""
    directions = np.asarray(directions, dtype=float)
    result = {}
    for index, direction in enumerate(directions):
        transformed = LORENZ_PARITY * direction
        parity = next(
            (value for value in (-1, 1) if np.allclose(transformed, value * direction)),
            None,
        )
        linear_allowed = LORENZ_PARITY == parity if parity is not None else np.ones(3, dtype=bool)
        quadratic_allowed = LORENZ_PARITY == 1 if parity is not None else np.ones(3, dtype=bool)
        result[index] = {
            "direction_parity": parity,
            "odd_fundamental": ~linear_allowed,
            "even_second_harmonic": ~quadratic_allowed,
            "even_dc": ~quadratic_allowed,
        }
    return result


def direction_target_contrasts(data: DirectionFrequencyData, omega: float) -> tuple[dict, np.ndarray]:
    """Form direction targets using A2, while retaining U2 only as a diagnostic."""
    positive = np.asarray(data.positive_cycle_fourier[omega]).mean(axis=-2)
    negative = np.asarray(data.negative_cycle_fourier[omega]).mean(axis=-2)
    baseline = np.asarray(data.unforced_cycle_fourier[omega]).mean(axis=-2)
    harmonic = {int(value): index for index, value in enumerate(data.harmonics)}
    odd = (positive - negative) / 2
    forced_even = (positive + negative) / 2
    return {
        "odd_fundamental": odd[..., harmonic[1]],
        "even_second_harmonic": forced_even[..., harmonic[2]],
        "even_dc": (
            forced_even[..., harmonic[0]] - baseline[:, None, None, :, harmonic[0]]
        ).real,
    }, baseline[..., harmonic[2]]


def reconstruct_strength_tensors(
    block_ids, directions, strengths, targets, unforced_second_harmonic
) -> DirectionTensorFits:
    """Fit strength series by block and reconstruct response tensors by block."""
    block_ids = tuple(block_ids)
    fitted = {}
    for target, orders in TARGET_ORDERS.items():
        values = np.asarray(targets[target])
        if values.ndim != 4:
            raise ValueError("direction targets require block,direction,strength,state axes")
        fitted[target] = fit_block_power_series(
            np.moveaxis(values, 2, 1), strengths, orders
        )

    def tensors(coefficient_index: int, strength_factor: float):
        directional = BlockFrequencyResponses(
            block_ids=block_ids,
            first_order=2j * fitted["odd_fundamental"].coefficients[:, coefficient_index] * strength_factor,
            second_harmonic=-4 * fitted["even_second_harmonic"].coefficients[:, coefficient_index] * strength_factor,
            rectification=fitted["even_dc"].coefficients[:, coefficient_index] * strength_factor,
        )
        return reconstruct_block_tensors(directions, directional)

    maximum = float(np.asarray(strengths)[-1])
    leading = tensors(0, 1.0)
    higher = tensors(1, 1.0)
    higher_at_maximum = tensors(1, maximum**2)
    return DirectionTensorFits(
        leading=leading,
        higher_coefficient=higher,
        higher_at_max_strength=higher_at_maximum,
        directional_targets={key: np.asarray(value) for key, value in targets.items()},
        unforced_second_harmonic=np.asarray(unforced_second_harmonic),
    )


def _tensor_components(target: str, values: np.ndarray):
    if target == "odd_fundamental":
        mask = lorenz_linear_sector_mask()
        for observable in range(3):
            for input_index in range(3):
                yield (
                    (observable, STATE_NAMES[input_index]),
                    values[:, observable, input_index],
                    bool(mask[observable, input_index]),
                )
    else:
        mask = lorenz_effective_quadratic_sector_mask()
        for observable in range(3):
            for pair_index, (first, second) in enumerate(INPUT_PAIR_ORDER):
                yield (
                    (observable, f"{STATE_NAMES[first]}{STATE_NAMES[second]}"),
                    values[:, observable, first, second],
                    bool(mask[observable, pair_index]),
                )


def _append_feature(columns, members, groups, values, metadata, structural_null):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("each family component must have exactly one block axis")
    real_index = len(columns)
    columns.append(np.asarray(values.real, dtype=float))
    members.append({**metadata, "part": "real", "structural_null": structural_null})
    imaginary_index = None
    if np.iscomplexobj(values):
        imaginary_index = len(columns)
        columns.append(np.asarray(values.imag, dtype=float))
        members.append({**metadata, "part": "imaginary", "structural_null": structural_null})
    groups.append(
        {
            **metadata,
            "real_index": real_index,
            "imaginary_index": imaginary_index,
            "structural_null": structural_null,
        }
    )


def build_joint_direction_family(data: DirectionFrequencyData):
    """Build one block-row family across every frequency and direction."""
    columns, members, groups = [], [], []
    fits = {}
    direction_nulls = direction_parity_null_masks(data.design.directions)
    for omega in data.frequencies:
        targets, u2 = direction_target_contrasts(data, omega)
        fit = reconstruct_strength_tensors(
            data.block_ids, data.design.directions, data.strengths, targets, u2
        )
        fits[omega] = fit
        for target, values in targets.items():
            normalization = {
                "odd_fundamental": lambda value, strength: 2j * value / strength,
                "even_second_harmonic": lambda value, strength: -4 * value / strength**2,
                "even_dc": lambda value, strength: value / strength**2,
            }[target]
            for direction_index, direction_name in enumerate(data.design.direction_names):
                for strength_index, strength in enumerate(data.strengths):
                    for observable in range(3):
                        _append_feature(
                            columns,
                            members,
                            groups,
                            normalization(values[:, direction_index, strength_index, observable], strength),
                            {
                                "omega": omega,
                                "section": "directional_identification",
                                "target": target,
                                "direction": direction_name,
                                "strength": float(strength),
                                "observable": STATE_NAMES[observable],
                            },
                            bool(direction_nulls[direction_index][target][observable]),
                        )
        for observable in range(3):
            _append_feature(
                columns,
                members,
                groups,
                u2[:, observable],
                {
                    "omega": omega,
                    "section": "unforced_U2_diagnostic",
                    "target": "unforced_U2",
                    "direction": None,
                    "strength": None,
                    "observable": STATE_NAMES[observable],
                },
                True,
            )
        tensor_sets = (
            ("tensor_leading", fit.leading),
            ("tensor_higher_at_max_strength", fit.higher_at_max_strength),
        )
        for section, tensors in tensor_sets:
            tensor_targets = {
                "odd_fundamental": tensors.first_order,
                "even_second_harmonic": tensors.second_harmonic,
                "even_dc": tensors.rectification,
            }
            for target, tensor_values in tensor_targets.items():
                for component, block_values, allowed in _tensor_components(target, tensor_values):
                    observable, input_component = component
                    _append_feature(
                        columns,
                        members,
                        groups,
                        block_values,
                        {
                            "omega": omega,
                            "section": section,
                            "target": target,
                            "direction": None,
                            "strength": float(data.strengths[-1]) if section.endswith("max_strength") else None,
                            "observable": STATE_NAMES[observable],
                            "input_component": input_component,
                        },
                        not allowed,
                    )
    return np.column_stack(columns), tuple(members), tuple(groups), fits


def _magnitude_bounds(group, lower, upper):
    def coordinate(index):
        lo, hi = float(lower[index]), float(upper[index])
        return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))

    rlo, rhi = coordinate(group["real_index"])
    if group["imaginary_index"] is None:
        return rlo, rhi
    ilo, ihi = coordinate(group["imaginary_index"])
    return float(np.hypot(rlo, ilo)), float(np.hypot(rhi, ihi))


def analyze_direction_reconnaissance(data: DirectionFrequencyData, config: dict) -> dict:
    """Fit/reconstruct and apply one simultaneous whole-block max bootstrap."""
    matrix, members, groups, _ = build_joint_direction_family(data)
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
    sector_entries = []
    null_exclusions = []
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
        if group["section"].startswith("tensor_"):
            record["sector"] = "forbidden" if group["structural_null"] else "allowed"
            sector_entries.append(record)
        if group["structural_null"] and magnitude_lower > 0:
            null_exclusions.append(record)
    leading_by_component = {
        (
            entry["omega"],
            entry["target"],
            entry["observable"],
            entry.get("input_component"),
        ): entry
        for entry in sector_entries
        if entry["section"] == "tensor_leading" and entry["sector"] == "allowed"
    }
    higher_by_component = {
        (
            entry["omega"],
            entry["target"],
            entry["observable"],
            entry.get("input_component"),
        ): entry
        for entry in sector_entries
        if entry["section"] == "tensor_higher_at_max_strength"
        and entry["sector"] == "allowed"
    }
    fraction_limit = float(bootstrap["higher_order_fraction_limit"])
    adequacy = []
    for key, leading in leading_by_component.items():
        higher = higher_by_component[key]
        denominator = leading["magnitude_lower"]
        ratio = higher["magnitude_upper"] / denominator if denominator > 0 else None
        adequacy.append(
            {
                "omega": key[0],
                "target": key[1],
                "observable": key[2],
                "input_component": key[3],
                "leading_magnitude_lower": denominator,
                "higher_at_max_strength_magnitude_upper": higher["magnitude_upper"],
                "higher_to_leading_upper": ratio,
                "denominator_status": "simultaneously_resolved" if denominator > 0 else "unresolved",
                "adequate": bool(ratio <= fraction_limit) if ratio is not None else False,
                "fraction_limit": fraction_limit,
            }
        )
    quadratic_residual_df = data.design.quadratic_residual_degrees_of_freedom
    if quadratic_residual_df:
        quadratic_residual_note = (
            f"{data.design.name} provides {quadratic_residual_df} quadratic "
            "residual degrees of freedom."
        )
    else:
        quadratic_residual_note = (
            f"{data.design.name} is exactly identified for the quadratic design; "
            "no quadratic residual diagnostic is available."
        )
    return {
        "interpretation": (
            f"Minimal {data.design.name} forcing-direction reconnaissance across "
            f"{len(data.frequencies)} configured frequencies. It diagnoses whether "
            "the next blocker is observation length, variance reduction, or broader "
            "frequency/direction design; it is not a formal scan or frequency selection."
        ),
        "design": {
            "name": data.design.name,
            "direction_names": data.design.direction_names,
            "directions": data.design.directions,
            "linear_residual_degrees_of_freedom": data.design.linear_residual_degrees_of_freedom,
            "quadratic_residual_degrees_of_freedom": quadratic_residual_df,
            "quadratic_residual_diagnostic_available": quadratic_residual_df > 0,
            "quadratic_residual_note": quadratic_residual_note,
        },
        "estimator": {
            "chi1": "2i times fitted h coefficient of the odd fundamental",
            "chi2": "-4 times fitted h^2 coefficient of forced-even A2; U2 is not subtracted",
            "Qdc": "fitted h^2 coefficient of forced-even A0 minus the shared U0 baseline",
            "U2": "retained as a separate structural-null diagnostic",
        },
        "coverage": {
            "confidence": confidence,
            "resamples": int(bootstrap["resamples"]),
            "critical_value": critical,
            "quantile_method": "higher",
            "resampling_unit": "one whole initial-state block row shared across all directions, strengths, targets, tensor sectors, and frequencies",
            "bootstrap": metadata,
        },
        "family": {
            "scalar_coordinate_count": matrix.shape[1],
            "members": members,
            "mean_realified": mean,
            "standard_error_realified": standard_error,
            "simultaneous_lower_realified": lower,
            "simultaneous_upper_realified": upper,
            "row_block_ids": data.block_ids,
            "frequency_major_order": data.frequencies,
            "direction_order": data.design.direction_names,
        },
        "tensor_sector_diagnosis": sector_entries,
        "tensor_component_adequacy": adequacy,
        "structural_null_exclusions": null_exclusions,
        "direction_parity_nulls": direction_parity_null_masks(data.design.directions),
        "higher_order_fraction_limit": {
            "value": fraction_limit,
            "status": "provisional_design_criterion_not_scientific_null_standard",
        },
        "source": data.source,
    }


def _raw_arrays(data: DirectionFrequencyData) -> dict:
    arrays = {
        "block_ids": np.asarray(data.block_ids, dtype=np.uint32),
        "strengths": data.strengths,
        "harmonics": data.harmonics,
        "frequencies": np.asarray(data.frequencies),
        "directions": data.design.directions,
        "direction_names": np.asarray(data.design.direction_names),
        "raw_proposals": data.raw_proposals,
        "initial_states": data.initial_states,
        "child_spawn_keys": np.asarray(data.child_spawn_keys, dtype=np.uint32),
    }
    for omega in data.frequencies:
        prefix = f"omega_{frequency_key(omega)}"
        arrays[f"{prefix}_positive_cycle_fourier"] = data.positive_cycle_fourier[omega]
        arrays[f"{prefix}_negative_cycle_fourier"] = data.negative_cycle_fourier[omega]
        arrays[f"{prefix}_unforced_cycle_fourier"] = data.unforced_cycle_fourier[omega]
    return arrays


def persist_direction_reconnaissance(output_dir, data, derived, config, provenance):
    """Persist the local direction-reconnaissance artifact contract."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config_snapshot.json"
    raw_path = output_dir / "raw_direction_summaries.npz"
    derived_path = output_dir / "derived_diagnostics.json"
    write_json_atomic(config_path, config)
    write_npz_atomic(raw_path, _raw_arrays(data))
    write_json_atomic(derived_path, derived)
    manifest = {
        "schema_version": 1,
        "classification": "exploratory_direction_reconnaissance",
        "study_id": config["study_id"],
        "files": {
            path.name: f"sha256:{file_sha256(path)}"
            for path in (config_path, raw_path, derived_path)
        },
        "array_semantics": {
            "forced_axes": RAW_FORCED_AXES,
            "unforced_axes": RAW_BASELINE_AXES,
            "baseline_storage": "one unforced block,state,cycle,harmonic array per frequency; never repeated over direction or strength",
            "fourier": "mean over phase times exp(-i*n*theta)",
        },
        "design": derived["design"],
        "estimator": derived["estimator"],
        "coverage": derived["coverage"],
        "source": derived["source"],
        "provenance": provenance,
        "interpretation": derived["interpretation"],
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def run_direction_reconnaissance(config_path):
    """Run the configured integration, analysis, and persistence workflow."""
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data = generate_direction_reconnaissance(config, repo_root)
    derived = analyze_direction_reconnaissance(data, config)
    config_identifier = f"sha256:{file_sha256(config_path)}"
    output_dir = repo_root / config["output_root"] / (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{config_identifier[7:19]}"
    )
    provenance = {
        "config_identifier": config_identifier,
        "code_identifiers": active_source_identifiers(repo_root, config["runner_path"]),
        "git": git_provenance(repo_root),
        "environment": environment_provenance(),
        "runtime_seconds_before_persistence": time.perf_counter() - started,
    }
    manifest = persist_direction_reconnaissance(output_dir, data, derived, config, provenance)
    return output_dir, manifest, time.perf_counter() - started
