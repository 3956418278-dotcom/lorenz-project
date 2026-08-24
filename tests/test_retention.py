import json

import numpy as np
import pytest

from lorenz.retention import (
    RetentionPolicy,
    chunk_file_name,
    chunk_ranges,
    condition_axis_permutation,
    condition_index,
    condition_labels,
    condition_table,
    load_spectrum_conditions,
    paired_condition_indices,
    paired_condition_vectors,
    parse_retention_policy,
    write_block_level_chunk,
    write_condition_metadata,
)


def test_default_policy_retains_contract_objects_and_not_dense():
    policy = parse_retention_policy({"block_count": 64}, 64)
    assert policy == RetentionPolicy(
        per_cycle_fourier_blocks=64,
        phase_samples_blocks=64,
        dense_trajectory_blocks=0,
        block_spectra=True,
        chunk_blocks=8,
    )


def test_policy_resolves_all_and_explicit_counts():
    config = {
        "retention": {
            "per_cycle_fourier_blocks": "all",
            "phase_samples_blocks": 3,
            "dense_trajectory_blocks": "all",
            "block_spectra": False,
            "chunk_blocks": 5,
        }
    }
    policy = parse_retention_policy(config, 10)
    assert policy.per_cycle_fourier_blocks == 10
    assert policy.phase_samples_blocks == 3
    assert policy.dense_trajectory_blocks == 10
    assert policy.block_spectra is False
    assert policy.chunk_blocks == 5


def test_policy_rejects_invalid_values():
    for value in ("everything", -1, True, 2.5):
        with pytest.raises(ValueError):
            parse_retention_policy(
                {"retention": {"per_cycle_fourier_blocks": value}}, 10
            )
    with pytest.raises(ValueError):
        parse_retention_policy({"retention": {"chunk_blocks": 0}}, 10)


def test_condition_table_pairs_signs_strengths_and_directions():
    forcing_vectors = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0],
         [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    table = condition_table(forcing_vectors)
    np.testing.assert_array_equal(table["sign"], [0, 1, -1, 1, -1, 0])
    np.testing.assert_allclose(
        table["signed_strength"], [0.0, 0.5, -0.5, 2.0, -2.0, 1.0]
    )
    np.testing.assert_allclose(table["direction"][1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(table["direction"][2], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(table["direction"][5], [0.0, 0.0, 1.0])
    assert condition_labels(forcing_vectors) == (
        "unforced", "+0.5", "-0.5", "+2", "-2", "+1",
    )


def test_paired_condition_axis_owns_lookup_and_reordering():
    directions = np.asarray(((1, 0, 0), (0, 1, 0)), dtype=float)
    strengths = np.asarray((0.5, 2.0))
    canonical = paired_condition_vectors(directions, strengths)
    assert canonical.shape == (9, 3)
    assert condition_index(canonical, [0.0, 2.0, 0.0]) == 7
    assert paired_condition_indices(canonical, directions[1], 2.0) == (7, 8)
    source = canonical[[0, 3, 4, 1, 2, 7, 8, 5, 6]]
    permutation = condition_axis_permutation(source, canonical)
    np.testing.assert_array_equal(source[permutation], canonical)


def test_selected_spectrum_conditions_load_across_chunks(tmp_path):
    directory = tmp_path / "omega_2"
    directory.mkdir()
    grid = np.asarray((0.0, 1.0, 2.0))
    for start in (0, 2):
        spectrum = np.arange(2 * 3 * 1 * 3).reshape(2, 3, 1, 3) + 100 * start
        write_block_level_chunk(
            directory / chunk_file_name("omega_2", start, start + 2),
            {
                "block_ids": np.arange(start, start + 2),
                "spectrum": spectrum.astype(complex),
                "spectrum_frequency_grid": grid,
            },
        )
    selected, loaded_grid = load_spectrum_conditions(
        tmp_path, "omega_2", [2, 0]
    )
    assert selected.shape == (4, 2, 1, 3)
    np.testing.assert_array_equal(loaded_grid, grid)


def test_chunk_ranges_cover_the_block_axis():
    assert list(chunk_ranges(10, 3)) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert list(chunk_ranges(4, 8)) == [(0, 4)]
    assert chunk_file_name("omega_8", 0, 8) == "omega_8_blocks_000000_000008.npz"


def test_block_level_chunk_round_trips(tmp_path):
    path = tmp_path / chunk_file_name("omega_2", 0, 2)
    arrays = {
        "block_ids": np.asarray([7, 9], dtype=np.uint32),
        "spectrum": np.arange(2 * 3 * 4).reshape(2, 3, 4) * (1 + 1j),
        "spectrum_frequency_grid": np.arange(4, dtype=float),
    }
    write_block_level_chunk(path, arrays)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        np.testing.assert_array_equal(loaded["block_ids"], arrays["block_ids"])
        np.testing.assert_array_equal(loaded["spectrum"], arrays["spectrum"])


def test_condition_metadata_json_strips_array_fields(tmp_path):
    config = {
        "lorenz": {"sigma": 10.0},
        "solver": {"method": "DOP853"},
        "initial_ensemble": {"spinup_time": 40.0},
        "protocol": {"direction": [1.0, 0.0, 0.0], "phase": 0.0},
    }
    sampling_records = {
        "8.0": {
            "omega": 8.0,
            "n_cycle": 510,
            "n_phase": 32,
            "condition_labels": ["unforced", "+0.5"],
            "_phase_sample_times": np.zeros((2, 4)),
        }
    }
    policy = parse_retention_policy({}, 64)
    path = tmp_path / "condition_metadata.json"
    write_condition_metadata(
        path, config, (0, 1), (8.0,), sampling_records, policy
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document["sampling_records"][0]
    assert record["omega"] == 8.0
    assert record["n_cycle"] == 510
    assert "_phase_sample_times" not in record
    assert document["observable_labels"] == ["x", "y", "z"]
    assert document["retention_policy"]["block_spectra"] is True
