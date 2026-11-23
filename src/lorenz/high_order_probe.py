"""Collect one-frequency matched-block Fourier coefficients.

Config -> ensemble -> conditions -> phase integration/Fourier -> chunks and
checkpoint. Read-only inference is performed offline via the analysis package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .ensemble import generate_configured_initial_state_blocks
from .parallel import close_execution, initialize_execution, execution_worker_count
from .persistence import prepare_artifact, read_chunk, write_checkpoint, finish_artifact
from .probe_config import validate_probe_config, probe_task_counts
from .retention import parse_retention_policy, chunk_ranges, chunk_file_name, write_block_level_chunk
from .sampling import CycleFourierSampling, frequency_key, iter_cycle_fourier_condition_means

REPO_ROOT = Path(__file__).resolve().parents[2]


def generate_probe_cell(config, checked, blocks, output_dir, policy):
    """Collect one complex Fourier coefficient per block/condition/harmonic.

    Workers average ``Z_n`` over physical cycles before transferring it.
    Averaging reduces within-trajectory noise; it does not create a
    new replicate.  Statistical replication remains the matched initial-state
    block.  The returned ``means`` are still Fourier coefficients and are not
    converted to chi1/chi2 inside the production collector.
    """
    plan = checked["condition_plan"]
    vectors, phases = plan["forcing_vectors"], plan["forcing_phases"]
    sampling = CycleFourierSampling(
        omega=checked["omega"], phase=float(config["protocol"].get("phase", 0.0)),
        discard_time=float(config["discard_time"]), n_cycle=checked["cycles"],
        n_phase=int(config["n_phase"]), harmonics=np.asarray(checked["harmonics"]),
    )
    numerical = {"lorenz": dict(config["lorenz"]), "solver": dict(config["solver"])}
    prefix = f"omega_{frequency_key(checked['omega'])}"
    directory = output_dir / "block_level" / prefix
    directory.mkdir(parents=True, exist_ok=True)
    means = np.empty((checked["block_count"], len(vectors), 3, len(checked["harmonics"])), dtype=complex)
    pending_chunks = {}
    missing_indices = []
    for start, end in chunk_ranges(checked["block_count"], policy.chunk_blocks):
        ids = blocks.block_ids[start:end]
        path = directory / chunk_file_name(prefix, start, end)
        if path.is_file():
            means[start:end] = read_chunk(path, ids, len(vectors), len(checked["harmonics"]))
            print(f"[{time.strftime('%H:%M:%S')}] probe chunk {start}-{end} reused", flush=True)
        else:
            pending_chunks[start] = [end, path, end - start]
            missing_indices.extend(range(start, end))
    workers = int(config.get("workers", 1))
    active_workers = execution_worker_count(workers)
    max_in_flight = config.get("execution", {}).get("max_in_flight")
    print(
        f"probe scheduling: missing_blocks={len(missing_indices)} workers={active_workers} "
        f"chunk_blocks={policy.chunk_blocks} max_in_flight={max_in_flight or 2 * active_workers}",
        flush=True,
    )
    if missing_indices:
        results = iter_cycle_fourier_condition_means(
            [blocks.block_ids[index] for index in missing_indices],
            blocks.final_states[missing_indices], vectors, sampling, numerical,
            forcing_phases=phases, workers=workers, max_in_flight=max_in_flight,
        )
        for task_index, block_mean in results:
            index = missing_indices[task_index]
            means[index] = block_mean
            start = (index // policy.chunk_blocks) * policy.chunk_blocks
            chunk = pending_chunks[start]
            chunk[2] -= 1
            if chunk[2] != 0:
                continue
            end, path, _ = pending_chunks.pop(start)
            ids = blocks.block_ids[start:end]
            write_block_level_chunk(path, {
                "block_ids": np.asarray(ids, dtype=np.uint32), "condition_means": means[start:end],
            })
            print(f"[{time.strftime('%H:%M:%S')}] probe chunk {start}-{end} integrated", flush=True)
    return {"means": means, "condition_vectors": vectors}


def run_probe(config_path, resume_dir=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_probe_config(config)
    policy = parse_retention_policy(config, checked["block_count"])
    output = prepare_artifact(config, config_path, REPO_ROOT, resume_dir)
    blocks = generate_configured_initial_state_blocks(config)
    print(f"probe blocks generated ({len(blocks.block_ids)} blocks)", flush=True)
    cell = generate_probe_cell(config, checked, blocks, output, policy)
    write_checkpoint(output, checked, blocks, cell["means"])
    finish_artifact(output, config, config_path, checked, blocks, REPO_ROOT)
    return output


def dry_run_summary(config):
    checked = validate_probe_config(config)
    policy = parse_retention_policy(config, checked["block_count"])
    return {
        "omega": checked["omega"], "directions": len(checked["directions"]),
        "strengths": checked["strengths"].tolist(),
        "strengths_by_direction": config.get("strengths_by_direction"),
        "mixed_phase_pairs": config.get("mixed_phase_pairs"),
        "experiment_type": checked["experiment_type"], "block_count": checked["block_count"],
        "cycles": checked["cycles"], "harmonics": checked["harmonics"],
        "n_phase": config["n_phase"], "discard_time": config["discard_time"],
        "phase": config["protocol"].get("phase", 0.0),
        "chunk_blocks": policy.chunk_blocks,
        "scheduling": "rolling matched blocks across persistence chunks",
        "configured_workers": int(config.get("workers", 1)),
        "max_in_flight": config.get("execution", {}).get("max_in_flight", "2 * active workers (cap 2048)"),
        **probe_task_counts(checked),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Dry-run never starts Dask/MPI, spins up blocks, or writes artifacts.
    print(json.dumps(dry_run_summary(config), indent=2))
    if args.dry_run:
        print("dry run: no integration performed")
        return
    runtime = initialize_execution(workers=int(config.get("workers", 1)))
    try:
        output = run_probe(args.config, resume_dir=args.resume)
        print(json.dumps({"output_dir": str(output)}, indent=2))
    finally:
        close_execution(runtime)


if __name__ == "__main__":
    main()
