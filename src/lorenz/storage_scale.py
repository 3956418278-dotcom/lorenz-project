"""Storage-scaling calculator for the retained-data contract.

The production retention decision is made from this report, not by
hard-coding.  For a planned experiment the calculator reports the storage of
each retained object class

- dense physical-time trajectories,
- estimator phase-grid samples,
- per-cycle complex Fourier coefficients,
- block-level physical-frequency complex spectra,
- derived summaries,

as functions of the block count ``B``, the direction/strength design, the
observation length, ``dt``, the phase resolution, the harmonic count, and the
spectrum bin count.

All sizes are reported uncompressed (float64 = 8 bytes, complex128 = 16
bytes).  Compression of chaotic trajectories is weak: the executed dense
pilot stored ``raw_x_response_summaries.npz`` at a measured uncompressed/
compressed ratio of about 0.95, which is offered as an optional scaling
factor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retention import parse_retention_policy
from .strength_study import minimum_duration_cycle_count

FLOAT64 = 8
COMPLEX128 = 16
STATE_COUNT = 3

# Reference ratio measured on the executed x-direction dense pilot artifact
# (sum of array nbytes over the stored npz file size).  Chaotic data does not
# compress well.
REFERENCE_COMPRESSION_RATIO = 0.9532


@dataclass(frozen=True)
class StoragePlan:
    """Planned production parameters; every field is an experiment input."""

    block_count: int
    direction_count: int
    frequencies: tuple[float, ...]
    strengths: tuple[float, ...]
    minimum_cycles: int
    minimum_physical_time: float
    discard_time: float
    dt: float
    n_phase: int
    harmonic_count: int
    spectrum_bins: int
    n_theta_bins: int
    welch_segment: int
    dense_trajectory_blocks: int | None
    per_cycle_fourier_blocks: int | None
    phase_samples_blocks: int | None
    raw_segment_blocks: int = 0
    raw_segment_samples: int = 0


def resolved_retention_blocks(value, block_count: int) -> int:
    """Resolve ``"all"``/count/``None`` retention inputs to block counts."""
    if value is None:
        return 0
    if value == "all":
        return int(block_count)
    return int(value)


def cycles_for(omega: float, minimum_cycles: int, minimum_physical_time: float) -> int:
    """Observation cycle count of one frequency cell."""
    return minimum_duration_cycle_count(omega, minimum_cycles, minimum_physical_time)


def dense_sample_count(omega: float, n_cycle: int, dt: float) -> int:
    """Dense samples spanning ``[discard, last phase sample]`` at spacing dt."""
    period = 2 * np.pi / float(omega)
    return int(np.floor(n_cycle * period / dt)) + 1


def condition_count(strength_count: int, direction_count: int) -> int:
    """Conditions per frequency: one shared unforced cell plus forced pairs."""
    return 1 + 2 * strength_count * direction_count


def storage_report(plan: StoragePlan) -> dict:
    """Return per-frequency and per-object-class storage sizes in bytes.

    The unforced condition is integrated once per block per frequency and is
    shared by every strength and direction at that frequency.
    """
    block_count = int(plan.block_count)
    strengths = tuple(float(value) for value in plan.strengths)
    frequencies = tuple(float(value) for value in plan.frequencies)
    n_harm = int(plan.harmonic_count)
    n_bins = int(plan.spectrum_bins)
    n_theta = int(plan.n_theta_bins)
    n_phase = int(plan.n_phase)
    dt = float(plan.dt)
    dense_blocks = resolved_retention_blocks(plan.dense_trajectory_blocks, block_count)
    fourier_blocks = resolved_retention_blocks(
        plan.per_cycle_fourier_blocks, block_count
    )
    phase_blocks = resolved_retention_blocks(plan.phase_samples_blocks, block_count)
    n_cond = condition_count(len(strengths), int(plan.direction_count))
    if n_phase < 2 or n_harm < 1 or n_bins < 1 or n_theta < 1:
        raise ValueError("phase/harmonic/bin counts must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    frequencies_rows = []
    totals = {
        "dense_trajectories": 0,
        "phase_samples": 0,
        "per_cycle_fourier": 0,
        "block_spectra": 0,
        "derived_summaries": 0,
        "identity_and_grids": 0,
    }
    for omega in frequencies:
        n_cycle = cycles_for(omega, int(plan.minimum_cycles), float(plan.minimum_physical_time))
        n_dense = dense_sample_count(omega, n_cycle, dt)
        blocks = block_count
        dense_size = dense_blocks * n_cond * STATE_COUNT * n_dense * FLOAT64
        phase_size = phase_blocks * n_cond * n_cycle * n_phase * STATE_COUNT * FLOAT64
        fourier_size = fourier_blocks * n_cond * n_cycle * STATE_COUNT * n_harm * COMPLEX128
        spectrum_size = blocks * n_cond * STATE_COUNT * n_bins * COMPLEX128
        raw_segments = min(blocks, int(plan.raw_segment_blocks))
        derived_size = (
            blocks * n_cond * STATE_COUNT * n_harm * COMPLEX128  # cycle means
            + blocks * n_cond * STATE_COUNT * n_harm * 2 * FLOAT64  # cycle variances
            + blocks * n_cond * STATE_COUNT * n_theta * FLOAT64  # folded cycle means
            + blocks * n_cond * STATE_COUNT * n_bins * FLOAT64  # Welch PSDs
            + blocks * n_cond * FLOAT64  # segment counts
            + raw_segments * n_cond * STATE_COUNT * int(plan.raw_segment_samples) * FLOAT64
        )
        grids = (
            n_cycle * n_phase * FLOAT64  # phase sample times
            + n_dense * FLOAT64  # dense sample times
            + n_bins * FLOAT64  # spectrum frequency grid
            + n_cond * (1 + 1 + 3) * FLOAT64  # condition signs/strengths/directions
        )
        frequencies_rows.append(
            {
                "omega": omega,
                "cycles": n_cycle,
                "dense_samples": n_dense,
                "conditions": n_cond,
                "dense_trajectories": dense_size,
                "phase_samples": phase_size,
                "per_cycle_fourier": fourier_size,
                "block_spectra": spectrum_size,
                "derived_summaries": derived_size,
                "identity_and_grids": grids,
                "total": dense_size + phase_size + fourier_size + spectrum_size + derived_size + grids,
            }
        )
        totals["dense_trajectories"] += dense_size
        totals["phase_samples"] += phase_size
        totals["per_cycle_fourier"] += fourier_size
        totals["block_spectra"] += spectrum_size
        totals["derived_summaries"] += derived_size
        totals["identity_and_grids"] += grids
    totals["total"] = sum(totals.values())
    return {
        "plan": {
            "block_count": block_count,
            "direction_count": int(plan.direction_count),
            "frequencies": list(frequencies),
            "strengths": list(strengths),
            "minimum_cycles": int(plan.minimum_cycles),
            "minimum_physical_time": float(plan.minimum_physical_time),
            "discard_time": float(plan.discard_time),
            "dt": dt,
            "n_phase": n_phase,
            "harmonic_count": n_harm,
            "spectrum_bins": n_bins,
            "n_theta_bins": n_theta,
            "welch_segment": int(plan.welch_segment),
            "conditions_per_frequency": n_cond,
            "retention_blocks": {
                "dense_trajectories": dense_blocks,
                "per_cycle_fourier": fourier_blocks,
                "phase_samples": phase_blocks,
            },
        },
        "frequencies": frequencies_rows,
        "totals": totals,
        "formulas": {
            "cycles": "max(minimum_cycles, ceil(minimum_physical_time * omega / 2*pi))",
            "dense_samples": "floor(cycles * 2*pi/omega / dt) + 1",
            "dense_trajectories": "B_retained * conditions * 3 states * dense_samples * 8",
            "phase_samples": "B_retained * conditions * cycles * n_phase * 3 states * 8",
            "per_cycle_fourier": "B_retained * conditions * cycles * 3 states * harmonics * 16",
            "block_spectra": "B * conditions * 3 states * spectrum_bins * 16",
            "derived_summaries": (
                "cycle means B*C*3*harmonics*16 + cycle variances B*C*3*harmonics*2*8 "
                "+ folded B*C*3*n_theta*8 + Welch B*C*3*bins*8 + segment counts B*C*8 "
                "+ retained raw segments"
            ),
            "uncompressed": "float64 = 8 bytes, complex128 = 16 bytes",
        },
    }


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    scaled = float(value)
    for unit in units:
        if scaled < 1024 or unit == units[-1]:
            return f"{scaled:.2f} {unit}"
        scaled /= 1024
    raise AssertionError("unreachable")


def print_report(report: dict, compression: float | None = None) -> None:
    """Print the storage report as a readable table."""
    factor = compression if compression else 1.0
    print("Storage-scaling report (uncompressed float64/complex128)")
    if compression:
        print(f"scaled by observed reference compression ratio {compression:.4f}")
    plan = report["plan"]
    print(
        f"B={plan['block_count']}, directions={plan['direction_count']}, "
        f"strengths={plan['strengths']}, frequencies={len(plan['frequencies'])}, "
        f"conditions/frequency={plan['conditions_per_frequency']}, "
        f"n_phase={plan['n_phase']}, harmonics={plan['harmonic_count']}, "
        f"spectrum_bins={plan['spectrum_bins']}, dt={plan['dt']}"
    )
    print(
        f"retention blocks: dense={plan['retention_blocks']['dense_trajectories']}, "
        f"per_cycle_fourier={plan['retention_blocks']['per_cycle_fourier']}, "
        f"phase_samples={plan['retention_blocks']['phase_samples']}"
    )
    header = (
        f"{'omega':>8} {'cycles':>7} {'dense':>7} "
        f"{'trajectories':>14} {'phase':>14} {'fourier':>14} "
        f"{'spectra':>14} {'derived':>14} {'grids':>14} {'total':>14}"
    )
    print(header)
    for row in report["frequencies"]:
        print(
            f"{row['omega']:>8} {row['cycles']:>7} {row['dense_samples']:>7} "
            f"{_format_bytes(factor * row['dense_trajectories']):>14} "
            f"{_format_bytes(factor * row['phase_samples']):>14} "
            f"{_format_bytes(factor * row['per_cycle_fourier']):>14} "
            f"{_format_bytes(factor * row['block_spectra']):>14} "
            f"{_format_bytes(factor * row['derived_summaries']):>14} "
            f"{_format_bytes(factor * row['identity_and_grids']):>14} "
            f"{_format_bytes(factor * row['total']):>14}"
        )
    totals = report["totals"]
    print(
        f"{'TOTAL':>8} {'':>7} {'':>7} "
        f"{_format_bytes(factor * totals['dense_trajectories']):>14} "
        f"{_format_bytes(factor * totals['phase_samples']):>14} "
        f"{_format_bytes(factor * totals['per_cycle_fourier']):>14} "
        f"{_format_bytes(factor * totals['block_spectra']):>14} "
        f"{_format_bytes(factor * totals['derived_summaries']):>14} "
        f"{_format_bytes(factor * totals['identity_and_grids']):>14} "
        f"{_format_bytes(factor * totals['total']):>14}"
    )


def plan_from_config(config: dict) -> StoragePlan:
    """Build a :class:`StoragePlan` from an experiment configuration."""
    retention = parse_retention_policy(config, int(config["block_count"]))

    def blocks(value):
        return None if value == 0 else value

    return StoragePlan(
        block_count=int(config["block_count"]),
        direction_count=1,
        frequencies=tuple(float(value) for value in config["frequencies"]),
        strengths=tuple(float(value) for value in config["strengths"]),
        minimum_cycles=int(config["observation_rule"]["minimum_cycles"]),
        minimum_physical_time=float(config["observation_rule"]["minimum_physical_time"]),
        discard_time=float(config["discard_time"]),
        dt=float(config["dense"]["dt"]),
        n_phase=int(config["n_phase"]),
        harmonic_count=len(config["harmonics"]),
        spectrum_bins=int(config["dense"]["max_psd_bins"]),
        n_theta_bins=int(config["dense"]["n_theta_bins"]),
        welch_segment=int(config["dense"]["welch_segment"]),
        dense_trajectory_blocks=blocks(retention.dense_trajectory_blocks),
        per_cycle_fourier_blocks=blocks(retention.per_cycle_fourier_blocks),
        phase_samples_blocks=blocks(retention.phase_samples_blocks),
        raw_segment_blocks=int(config["dense"].get("raw_segment_blocks", 0)),
        raw_segment_samples=int(config["dense"].get("raw_segment_samples", 0)),
    )


def _float_list(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value.strip())


def main(argv=None) -> None:
    """Command-line calculator: ``python -m lorenz.storage_scale``."""
    parser = argparse.ArgumentParser(
        description="Storage-scaling calculator for the retained-data contract"
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="experiment configuration JSON")
    parser.add_argument("--block-count", type=int, default=None)
    parser.add_argument("--directions", type=int, default=1)
    parser.add_argument("--frequencies", type=_float_list, default=None)
    parser.add_argument("--strengths", type=_float_list, default=None)
    parser.add_argument("--minimum-cycles", type=int, default=None)
    parser.add_argument("--minimum-time", type=float, default=None)
    parser.add_argument("--discard-time", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--n-phase", type=int, default=None)
    parser.add_argument("--harmonics", type=int, default=None)
    parser.add_argument("--spectrum-bins", type=int, default=None)
    parser.add_argument("--theta-bins", type=int, default=None)
    parser.add_argument("--welch-segment", type=int, default=None)
    parser.add_argument("--dense-trajectory-blocks", default=None)
    parser.add_argument("--per-cycle-fourier-blocks", default=None)
    parser.add_argument("--phase-samples-blocks", default=None)
    parser.add_argument("--compression", type=float, default=None,
                        help="optional compression factor (reference: "
                        f"{REFERENCE_COMPRESSION_RATIO:.4f} observed on the dense pilot)")
    arguments = parser.parse_args(argv)

    if arguments.config is not None:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        plan = plan_from_config(config)
        overrides = {}
        if arguments.directions != 1:
            overrides["direction_count"] = arguments.directions
        for field, value in (
            ("dense_trajectory_blocks", arguments.dense_trajectory_blocks),
            ("per_cycle_fourier_blocks", arguments.per_cycle_fourier_blocks),
            ("phase_samples_blocks", arguments.phase_samples_blocks),
        ):
            if value is not None:
                overrides[field] = 0 if value in ("none", "0") else value
        if overrides:
            from dataclasses import replace

            plan = replace(plan, **overrides)
    else:
        required = (
            arguments.block_count, arguments.frequencies, arguments.strengths,
            arguments.minimum_cycles, arguments.minimum_time, arguments.dt,
            arguments.n_phase, arguments.harmonics, arguments.spectrum_bins,
            arguments.theta_bins, arguments.welch_segment,
        )
        if any(value is None for value in required):
            parser.error(
                "either --config or all of --block-count, --frequencies, "
                "--strengths, --minimum-cycles, --minimum-time, --dt, "
                "--n-phase, --harmonics, --spectrum-bins, --theta-bins, "
                "--welch-segment are required"
            )
        def resolve_blocks(value):
            if value is None or value == "none":
                return 0
            return value

        plan = StoragePlan(
            block_count=arguments.block_count,
            direction_count=arguments.directions,
            frequencies=arguments.frequencies,
            strengths=arguments.strengths,
            minimum_cycles=arguments.minimum_cycles,
            minimum_physical_time=arguments.minimum_time,
            discard_time=arguments.discard_time if arguments.discard_time is not None else 0.0,
            dt=arguments.dt,
            n_phase=arguments.n_phase,
            harmonic_count=arguments.harmonics,
            spectrum_bins=arguments.spectrum_bins,
            n_theta_bins=arguments.theta_bins,
            welch_segment=arguments.welch_segment,
            dense_trajectory_blocks=resolve_blocks(arguments.dense_trajectory_blocks),
            per_cycle_fourier_blocks=resolve_blocks(arguments.per_cycle_fourier_blocks),
            phase_samples_blocks=resolve_blocks(arguments.phase_samples_blocks),
        )
    report = storage_report(plan)
    print_report(report, arguments.compression)


if __name__ == "__main__":
    main()
