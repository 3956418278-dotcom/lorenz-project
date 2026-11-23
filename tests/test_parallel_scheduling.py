"""Regression tests for rolling task scheduling independent of persistence chunks."""
from __future__ import annotations

import os
import time

import numpy as np

from lorenz import high_order_probe as runner, parallel
from lorenz.ensemble import generate_configured_initial_state_blocks
from lorenz.probe_config import validate_probe_config
from lorenz.retention import parse_retention_policy


def _example_config() -> dict:
    return {
        "schema_version": 1,
        "study_id": "scheduling_test",
        "classification": "test",
        "output_root": "outputs/test_run",
        "runner_path": "src/lorenz/high_order_probe.py",
        "omega": 2.0,
        "strengths": [0.5, 1.0],
        "block_count": 3,
        "workers": 1,
        "discard_time": 0.1,
        "n_phase": 32,
        "harmonics": [0, 1, 2, 3, 4, 5],
        "observation_rule": {
            "minimum_cycles": 2,
            "minimum_physical_time": 1.0,
        },
        "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "solver": {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11},
        "initial_ensemble": {
            "root_entropy": 42,
            "block_id_start": 0,
            "spinup_time": 0.2,
            "proposal": {
                "x_half_width": 20.0,
                "y_half_width": 30.0,
                "z_bounds": [0.0, 50.0],
            },
        },
        "protocol": {
            "status": "test",
            "directions": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "phase": 0.0,
        },
        "retention": {
            "per_cycle_fourier_blocks": 0,
            "phase_samples_blocks": 0,
            "dense_trajectory_blocks": 0,
            "spectrum_blocks": 0,
            "chunk_blocks": 1,
        },
        "generate_figures": False,
    }


def _timed_task(index):
    start = time.perf_counter()
    time.sleep(0.12 if index == 0 else 0.025)
    return os.getpid(), start, time.perf_counter(), index


def test_process_pool_is_reused_and_completions_keep_input_indices():
    runtime = parallel.initialize_execution(workers=2)
    try:
        first = dict(parallel.iter_task_results(_timed_task, range(6), workers=2))
        executor = runtime.executor
        second = dict(parallel.iter_task_results(_timed_task, range(6), workers=2))
        assert runtime.executor is executor
        assert {row[0] for row in first.values()} == {row[0] for row in second.values()}
        assert len({row[0] for row in second.values()}) == 2
        assert [second[index][3] for index in range(6)] == list(range(6))
        assert second[1][2] < second[0][2]
    finally:
        parallel.close_execution(runtime)


def test_collector_submits_across_chunks_and_persists_completed_chunk_immediately(monkeypatch, tmp_path):
    config = _example_config()
    config["block_count"] = 5
    config["initial_ensemble"]["spinup_time"] = 0
    config["retention"]["chunk_blocks"] = 2
    checked = validate_probe_config(config)
    blocks = generate_configured_initial_state_blocks(config)
    shape = (len(checked["condition_plan"]["forcing_vectors"]), 3, 6)

    def unordered(block_ids, *_args, **_kwargs):
        assert tuple(block_ids) == blocks.block_ids
        for index in (2, 3):
            yield index, np.full(shape, index, dtype=complex)
        assert len(list(tmp_path.glob("block_level/*/*.npz"))) == 1
        yield 4, np.full(shape, 4, dtype=complex)
        assert len(list(tmp_path.glob("block_level/*/*.npz"))) == 2
        for index in (1, 0):
            yield index, np.full(shape, index, dtype=complex)

    monkeypatch.setattr(runner, "iter_cycle_fourier_condition_means", unordered)
    cell = runner.generate_probe_cell(config, checked, blocks, tmp_path, parse_retention_policy(config, 5))
    for index in range(5):
        np.testing.assert_array_equal(cell["means"][index], np.full(shape, index, dtype=complex))
    assert len(list(tmp_path.glob("block_level/*/*.npz"))) == 3


def test_spinup_uses_configured_process_workers_and_is_numerically_identical():
    config = _example_config()
    serial = generate_configured_initial_state_blocks(config)
    config["workers"] = 2
    process = generate_configured_initial_state_blocks(config)
    np.testing.assert_array_equal(process.final_states, serial.final_states)


def test_single_strength_paired_design_has_all_three_sign_pairs():
    config = _example_config()
    config["protocol"]["directions"] = np.eye(3).tolist()
    config["strengths"] = [1.0]
    config["strengths_by_direction"] = {name: [1.0] for name in ("x", "y", "z")}
    checked = validate_probe_config(config)
    vectors = checked["condition_plan"]["forcing_vectors"]
    assert len(vectors) == 7
    for axis in np.eye(3):
        for sign in (-1, 1):
            assert any(np.array_equal(vector, sign * axis) for vector in vectors)
