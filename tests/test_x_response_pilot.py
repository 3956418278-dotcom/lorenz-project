from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

import lorenz.x_response_pilot as pilot


def _config():
    return {
        "schema_version": 1,
        "study_id": "test_retention",
        "classification": "test",
        "output_root": "outputs/test",
        "runner_path": "experiments/run_x_response_pilot.py",
        "frequencies": [1.0],
        "strengths": [0.5, 1.0],
        "block_count": 2,
        "workers": 1,
        "discard_time": 0.5,
        "n_phase": 16,
        "harmonics": [0, 1, 2, 3, 4, 5],
        "observation_rule": {"minimum_cycles": 2, "minimum_physical_time": 1.0},
        "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "solver": {"method": "DOP853", "rtol": 1e-8, "atol": 1e-10},
        "initial_ensemble": {
            "root_entropy": 1,
            "block_id_start": 5000,
            "spinup_time": 0.5,
            "proposal": {"x_half_width": 20.0, "y_half_width": 30.0,
                         "z_bounds": [0.0, 50.0]},
        },
        "protocol": {"direction": [1.0, 0.0, 0.0], "phase": 0.0},
        "dense": {
            "dt": 0.25,
            "n_theta_bins": 32,
            "welch_segment": 16,
            "max_psd_bins": 8,
            "raw_segment_blocks": 1,
            "raw_segment_samples": 32,
        },
        "retention": {
            "per_cycle_fourier_blocks": "all",
            "phase_samples_blocks": "all",
            "dense_trajectory_blocks": "all",
            "block_spectra": True,
            "chunk_blocks": 1,
        },
        "bootstrap": {
            "confidence": 0.95,
            "resamples": 3,
            "root_entropy": 1,
            "batch_size": 8,
            "higher_order_fraction_limit": 0.2,
        },
    }


def _fake_blocks(config):
    initial = config["initial_ensemble"]
    start = int(initial["block_id_start"])
    block_ids = tuple(range(start, start + int(config["block_count"])))
    rng = np.random.default_rng(12345)
    return SimpleNamespace(
        block_ids=block_ids,
        final_states=rng.uniform(-10, 10, size=(len(block_ids), 3)),
        raw_proposals=rng.uniform(-10, 10, size=(len(block_ids), 3)),
        child_spawn_keys=tuple((value,) for value in block_ids),
        root_entropy=initial["root_entropy"],
        root_spawn_key=(0,),
        bit_generator="PCG64DXSM",
        spinup_time=float(initial["spinup_time"]),
    )


def test_chunked_generation_persists_and_reloads_block_level_objects(
    tmp_path, monkeypatch
):
    config = _config()
    monkeypatch.setattr(
        pilot, "generate_initial_state_blocks",
        lambda *args, **kwargs: _fake_blocks(config),
    )
    output_dir = tmp_path / "artifact"
    output_dir.mkdir()
    block_level_dir = output_dir / "block_level" / "omega_1"
    cell = pilot.generate_x_response_cell(
        config, 1.0, config["strengths"], 2, block_level_dir=block_level_dir
    )
    assert cell.study.positive_cycle_fourier.shape == (2, 2, 3, 1, 6)
    assert cell.cycle_variances["positive"].shape == (2, 2, 3, 6, 2)
    metadata = cell.sampling_metadata
    assert metadata["n_cycle"] == 2
    assert metadata["n_phase"] == 16
    assert metadata["dense_dt"] == 0.25
    assert metadata["_phase_sample_times"].shape == (2, 16)
    assert metadata["condition_labels"] == ["unforced", "+0.5", "-0.5", "+1", "-1"]

    chunk_paths = sorted(block_level_dir.glob("*.npz"))
    assert len(chunk_paths) == 2  # chunk_blocks=1 over 2 blocks
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as chunk:
            assert set(chunk.files) == {
                "block_ids", "cycle_fourier", "phase_values",
                "dense_values", "dense_times", "spectrum",
                "spectrum_frequency_grid", "spectrum_segment_counts",
            }
            assert chunk["cycle_fourier"].shape[1:] == (5, 2, 3, 6)
            assert chunk["phase_values"].shape[1:] == (5, 2, 16, 3)
            assert chunk["dense_values"].shape[1:] == (5, 3, chunk["dense_times"].shape[0])
            assert chunk["spectrum"].shape[1:] == (5, 3, 8)

    derived = pilot.analyze_x_response({1.0: cell}, config)
    assert len(derived["frequencies"]) == 1
    pilot.write_json_atomic(output_dir / "config_snapshot.json", config)
    provenance = {
        "config_identifier": "sha256:test",
        "code_identifiers": {},
        "git": {"head": None, "worktree_dirty": None},
        "environment": {},
    }
    manifest = pilot.persist_x_response(
        output_dir, {1.0: cell}, derived, config, provenance
    )
    assert manifest["schema_version"] == 2
    assert "condition_metadata.json" in manifest["files"]
    assert any(
        name.startswith("block_level/omega_1")
        for name in manifest["files"]
    )

    loaded = pilot.load_x_response_cells(output_dir / "raw_x_response_summaries.npz")
    reloaded = loaded[1.0]
    assert reloaded.study.n_phase == 16
    assert reloaded.sampling_metadata["n_cycle"] == 2
    spectra = reloaded.block_spectra
    assert spectra is not None
    assert spectra["values"].shape == (2, 5, 3, 8)
    assert spectra["frequency_grid"].shape == (8,)
    retained = reloaded.retained_cycle_fourier
    assert retained["values"].shape == (2, 5, 2, 3, 6)
    np.testing.assert_allclose(
        retained["values"].mean(axis=2),
        np.concatenate(
            [
                reloaded.study.unforced_cycle_fourier[:, :, 0, :][:, None],
                reloaded.study.positive_cycle_fourier[:, 0, :, 0, :][:, None],
                reloaded.study.negative_cycle_fourier[:, 0, :, 0, :][:, None],
                reloaded.study.positive_cycle_fourier[:, 1, :, 0, :][:, None],
                reloaded.study.negative_cycle_fourier[:, 1, :, 0, :][:, None],
            ],
            axis=1,
        ),
        rtol=1e-12,
        atol=1e-14,
    )
    assert reloaded.retained_phase_samples["values"].shape == (2, 5, 2, 16, 3)
    assert reloaded.retained_dense["values"].shape[0] == 2
    assert reloaded.retained_dense["times"].shape == (
        reloaded.retained_dense["values"].shape[3],
    )


def test_validate_config_parses_retention_policy():
    config = _config()
    checked = pilot.validate_x_response_config(config)
    policy = checked["retention_policy"]
    assert policy.per_cycle_fourier_blocks == 2
    assert policy.phase_samples_blocks == 2
    assert policy.dense_trajectory_blocks == 2
    assert policy.block_spectra is True
    assert policy.chunk_blocks == 1


def test_merge_cell_merges_spectra_along_the_condition_axis():
    def cell(strengths, fourier_values):
        study = pilot.StrengthStudyData(
            block_ids=(0, 1),
            strengths=np.asarray(strengths, float),
            harmonics=np.asarray([0, 1, 2]),
            omega=2.0,
            n_phase=8,
            positive_cycle_fourier=fourier_values,
            negative_cycle_fourier=np.zeros_like(fourier_values),
            unforced_cycle_fourier=np.zeros((2, 3, 1, 3)),
            raw_proposals=np.zeros((2, 3)),
            initial_states=np.ones((2, 3)),
            child_spawn_keys=((0,), (1,)),
            generation_metadata={},
        )
        base = pilot.XResponseCell(
            omega=2.0, study=study,
            folded_cycle_mean=None, welch_psd=None, segment_counts=None,
            raw_segments={}, dense_metadata={}, runtime_seconds=0.0,
            block_spectra={
                "values": np.arange(2 * (1 + 2 * len(strengths)) * 3 * 4).reshape(
                    2, 1 + 2 * len(strengths), 3, 4
                ) * (1 + 1j),
                "frequency_grid": np.arange(4, dtype=float),
                "segment_counts": np.ones((2, 1 + 2 * len(strengths)), dtype=np.int64),
                "block_ids": np.asarray([0, 1], dtype=np.uint32),
            },
        )
        return base

    base = cell([0.5, 1.0], np.ones((2, 2, 3, 1, 3)))
    extra = cell([3.0, 4.0], np.full((2, 2, 3, 1, 3), 7.0))
    merged = pilot.merge_cell(base, extra)
    assert tuple(merged.study.strengths) == (0.5, 1.0, 3.0, 4.0)
    assert merged.study.positive_cycle_fourier.shape == (2, 4, 3, 1, 3)
    spectra = merged.block_spectra
    assert spectra["values"].shape == (2, 9, 3, 4)
    assert spectra["segment_counts"].shape == (2, 9)
    # unforced cell shared exactly
    np.testing.assert_array_equal(
        spectra["values"][:, 0], base.block_spectra["values"][:, 0]
    )
    # base strengths keep their spectra; extra strengths come from the extra cell
    np.testing.assert_array_equal(
        spectra["values"][:, 1:5], base.block_spectra["values"][:, 1:5]
    )
    np.testing.assert_array_equal(
        spectra["values"][:, 5:9], extra.block_spectra["values"][:, 1:5]
    )
