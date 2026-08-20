#!/usr/bin/env python3
"""High-order probe: one frequency, six forcing directions, shared unforced.

Goal of the probe (NOT a perturbative-tensor measurement): probe the 1..5
omega response across the full second-order direction design and identify
which (direction, output, harmonic) cells show a reliable signed response.
The unforced trajectory is integrated once per block and shared by every
direction and strength.  All outputs (x, y, z) and harmonics n=0..5 are
retained as block-level complex coefficients and analyzed under the signed
cos/sin convention (Re = cos component, Im = -(sin component)) with
Hotelling T^2 joint tests; DC is a scalar.  High amplitudes (h=8) are
probe settings, not perturbative coefficients.

The final result is the response map
    direction x output x harmonic x strength x {cos, sin}
plus the same objects' statistics.

Run with --dry-run to validate the configuration and report the expected
task counts without integrating anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.artifacts import (  # noqa: E402
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    write_json_atomic,
    write_npz_atomic,
)
from lorenz.retention import (  # noqa: E402
    chunk_file_name,
    chunk_ranges,
    condition_table,
    parse_retention_policy,
    write_block_level_chunk,
)
from lorenz.statistics import cos_sin_statistics  # noqa: E402
from lorenz.strength_study import (  # noqa: E402
    STATE_NAMES,
    CycleFourierSampling,
    _checked_strengths,
    integrate_phase_and_dense_conditions,
    minimum_duration_cycle_count,
)
from lorenz.x_response_pilot import (  # noqa: E402
    generate_production_blocks,
)

BLOCK_LEVEL_DIRECTORY = "block_level"


def validate_probe_config(config: dict) -> dict:
    """Validate the probe-specific configuration surface."""
    omega = float(config["omega"])
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    strengths = _checked_strengths(config["strengths"], minimum_count=2)
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
    policy = parse_retention_policy(config, block_count)
    return {
        "omega": omega,
        "strengths": strengths,
        "directions": directions,
        "block_count": block_count,
        "cycles": cycles,
        "harmonics": [int(value) for value in harmonics],
        "retention_policy": policy,
    }


def probe_task_counts(checked: dict) -> dict:
    """Expected integration task counts (no integration performed)."""
    conditions = 1 + 2 * len(checked["strengths"]) * len(checked["directions"])
    return {
        "unforced_trajectories": checked["block_count"],
        "forced_trajectories": checked["block_count"] * (conditions - 1),
        "total_condition_integrations": checked["block_count"] * conditions,
        "spinup_integrations": checked["block_count"],
        "conditions_per_block": conditions,
    }


def generate_probe_cell(config, checked, blocks, output_dir, policy) -> dict:
    """Integrate one frequency cell with six directions and one shared
    unforced trajectory per block, persisting the retained block-level
    objects and returning the per-condition cycle means."""
    omega = checked["omega"]
    strengths = checked["strengths"]
    directions = checked["directions"]
    block_count = checked["block_count"]
    workers = int(config.get("workers", 1))
    forcing_vectors = [np.zeros(3)]
    forcing_vectors.extend(
        sign * strength * direction
        for direction in directions
        for strength in strengths
        for sign in (1.0, -1.0)
    )
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
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
        chunk_raw = max(0, min(raw_segment_blocks - start, chunk_length))
        fourier, summaries, spectra, _, _ = integrate_phase_and_dense_conditions(
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
        means_pieces.append(fourier.mean(axis=3))  # (chunk, cond, 3, H)
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
        chunk_path = (
            output_dir / BLOCK_LEVEL_DIRECTORY / prefix
            / chunk_file_name(prefix, start, end)
        )
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        write_block_level_chunk(
            chunk_path,
            _chunk_arrays(
                chunk_ids, retained_fourier, retained_spectra, spectrum_grid,
                spectrum_counts,
            ),
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] probe chunk {start}-{end} of "
            f"{block_count} integrated",
            flush=True,
        )
    means = np.concatenate(means_pieces, axis=0)  # (B, cond, 3, H)
    table = condition_table(forcing_vectors)
    return {
        "means": means,
        "condition_signs": table["sign"],
        "condition_strengths": table["signed_strength"],
        "condition_directions": table["direction"],
    }


def _chunk_arrays(chunk_ids, cycle_fourier, spectra, grid, counts):
    arrays = {"block_ids": np.asarray(chunk_ids, dtype=np.uint32)}
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
        direction_label = "".join(
            f"{direction[i]:+g}{name}"
            for i, name in enumerate(STATE_NAMES)
            if direction[i] != 0
        )
        for strength in strengths:
            strength = float(strength)
            plus = _condition_for(cell, direction, +strength)
            minus = _condition_for(cell, direction, -strength)
            odd = (means[:, plus] - means[:, minus]) / 2
            even = (means[:, plus] + means[:, minus]) / 2
            for harmonic in harmonics:
                for output, name in enumerate(STATE_NAMES):
                    if harmonic == 0:
                        dc = (even[:, output, idx[0]] - unforced[:, output, idx[0]]).real
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
                    values = (
                        odd[:, output, idx[harmonic]]
                        if harmonic % 2 == 1
                        else even[:, output, idx[harmonic]]
                    )
                    stats = cos_sin_statistics(values)
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
                        "hotelling_p": stats.hotelling_p_value,
                        "used_pseudoinverse": stats.used_pseudoinverse,
                        "magnitude_derived": stats.magnitude,
                        "phase_derived": stats.phase,
                    })
    return entries


def _condition_for(cell, direction, signed_strength):
    direction = np.asarray(direction, dtype=float)
    for index in range(len(cell["condition_strengths"])):
        if (
            abs(float(cell["condition_strengths"][index]) - signed_strength) < 1e-12
            and np.allclose(
                cell["condition_directions"][index],
                direction * np.sign(signed_strength),
                atol=1e-9,
            )
        ):
            return index
    raise ValueError(
        f"condition direction={direction} signed={signed_strength} not found"
    )


def run_probe(config_path, resume_dir=None):
    config = json.loads(Path(config_path).resolve().read_text(encoding="utf-8"))
    checked = validate_probe_config(config)
    policy = checked["retention_policy"]
    if resume_dir is not None and Path(resume_dir).is_dir():
        output_dir = Path(resume_dir)
    else:
        config_identifier = f"sha256:{file_sha256(config_path)}"
        output_dir = REPO_ROOT / config["output_root"] / (
            f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{config_identifier[7:19]}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(output_dir / "config_snapshot.json", config)
    blocks = generate_production_blocks(config)
    print(
        f"[{time.strftime('%H:%M:%S')}] probe blocks generated "
        f"({len(blocks.block_ids)} blocks)",
        flush=True,
    )
    cell = generate_probe_cell(config, checked, blocks, output_dir, policy)
    prefix = f"omega_{format(float(checked['omega']), '.12g').replace('-', 'm').replace('.', 'p')}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        f"{prefix}_block_ids": np.asarray(blocks.block_ids, dtype=np.uint32),
        f"{prefix}_condition_means": cell["means"],
        f"{prefix}_condition_signs": cell["condition_signs"],
        f"{prefix}_condition_strengths": cell["condition_strengths"],
        f"{prefix}_condition_directions": cell["condition_directions"],
        f"{prefix}_strengths": checked["strengths"],
        f"{prefix}_harmonics": np.asarray(checked["harmonics"], dtype=int),
    }
    write_npz_atomic(checkpoint_dir / f"{prefix}.npz", checkpoint)
    entries = build_response_map(cell, checked)
    derived = {
        "interpretation": (
            "high-order probe: which (direction, output, harmonic) cells "
            "show a reliable signed response; high amplitudes are probe "
            "settings, not perturbative coefficients"
        ),
        "convention": (
            "mean(x exp(-i n theta)); cos component = Re, sin component = -Im"
        ),
        "directions": [value.tolist() for value in checked["directions"]],
        "strengths": [float(value) for value in checked["strengths"]],
        "harmonics": [0, 1, 2, 3, 4, 5],
        "response_map": entries,
    }
    write_json_atomic(output_dir / "derived_response_map.json", derived)
    provenance = {
        "config_identifier": f"sha256:{file_sha256(config_path)}",
        "code_identifiers": active_source_identifiers(REPO_ROOT, "experiments/run_high_order_probe.py"),
        "git": git_provenance(REPO_ROOT),
        "environment": environment_provenance(),
    }
    manifest = {
        "schema_version": 1,
        "classification": config["classification"],
        "study_id": config["study_id"],
        "files": {
            "config_snapshot.json": f"sha256:{file_sha256(output_dir / 'config_snapshot.json')}",
            "derived_response_map.json": f"sha256:{file_sha256(output_dir / 'derived_response_map.json')}",
        },
        "provenance": provenance,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/production/high_order_probe_v1.json",
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
