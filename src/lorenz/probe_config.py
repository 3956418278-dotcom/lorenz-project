"""Validate the current one-frequency experiment design and sampling parameters."""
from __future__ import annotations
import numpy as np
from .conditions import direction_identity, mirrored_phase_pair_conditions, paired_direction_strength_vectors
from .sampling import STATE_NAMES, validate_strengths, minimum_duration_cycle_count

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
        axis_strengths = mixed_config.get("axis_strengths")
        if axis_strengths is not None:
            if not isinstance(axis_strengths, dict):
                raise ValueError("mixed axis_strengths must be a mapping")
            if not set(axis_strengths) <= set(STATE_NAMES):
                raise ValueError("mixed axis_strengths keys must be x, y, or z")
        target_outputs = mixed_config.get("target_outputs")
        if target_outputs is not None:
            if not isinstance(target_outputs, dict):
                raise ValueError("mixed target_outputs must be a mapping")
            if not set(target_outputs) <= set(names):
                raise ValueError(
                    "mixed target_outputs keys must be configured direction names"
                )
            if any(value not in STATE_NAMES for value in target_outputs.values()):
                raise ValueError("mixed target_outputs values must be x, y, or z")
        condition_plan = mirrored_phase_pair_conditions(
            directions,
            names,
            scales,
            [base_amplitudes[name] for name in names],
            reference_phase=float(config["protocol"].get("phase", 0.0)),
            phase_offset=phase_offset,
            axis_strengths=axis_strengths,
        )
        strengths = scales
        strengths_by_direction = None
    else:
        # Explicit per-direction instances already permit one strength, e.g.
        # a post-selection measurability pilot; their sorted union may be one.
        strengths = validate_strengths(config["strengths"], minimum_count=1 if configured_by_direction is not None else 2)
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
    workers = config.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    execution = config.get("execution", {})
    if not isinstance(execution, dict) or set(execution) - {"max_in_flight"}:
        raise ValueError("execution only supports max_in_flight")
    in_flight = execution.get("max_in_flight")
    if in_flight is not None and (isinstance(in_flight, bool) or not isinstance(in_flight, int) or in_flight < 1):
        raise ValueError("execution.max_in_flight must be a positive integer")
    return {
        "omega": omega,
        "strengths": strengths,
        "strengths_by_direction": strengths_by_direction,
        "experiment_type": "mirrored_phase_pair" if mixed_config is not None else "paired_sign",
        "condition_plan": condition_plan if condition_plan is not None else {
            "forcing_vectors": paired_direction_strength_vectors(directions, strengths_by_direction),
            "forcing_phases": None,
            "condition_labels": None,
            "pairs": [],
        },
        "directions": directions,
        "block_count": block_count,
        "cycles": cycles,
        "harmonics": [int(value) for value in harmonics],
    }


def probe_task_counts(checked: dict) -> dict:
    """Expected integration task counts (no integration performed)."""
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
