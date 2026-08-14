import json

import numpy as np

from lorenz.spinup_diagnostics import (
    SpinupStudyData,
    analyze_spinup_study,
    energy_distance,
    paired_endpoint_drift,
    persist_spinup_study,
    summarize_states,
)


LORENZ = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}


def _study_data():
    first = np.array(
        [[-2.0, -1.0, 20.0], [-1.0, -2.0, 24.0], [1.0, 2.0, 26.0], [2.0, 1.0, 30.0]]
    )
    endpoints = np.stack(
        (
            np.stack((first + [1.0, 0.0, 0.0], first + [0.5, 0.0, 0.0], first)),
            np.stack((first + [3.0, 1.0, 2.0], first + [1.0, 0.5, 1.0], first + [0.2, 0.1, 0.3])),
        )
    )
    return SpinupStudyData(
        proposal_names=("first", "second"),
        burnin_times=np.array([1.0, 2.0, 4.0]),
        block_ids=np.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.uint32),
        raw_proposals=np.stack((first, first + 1.0)),
        endpoint_states=endpoints,
        child_spawn_keys=np.array([[[0], [1], [2], [3]], [[10], [11], [12], [13]]], dtype=np.uint32),
        generation_metadata=({"proposal_name": "first"}, {"proposal_name": "second"}),
        batch_ids=np.array([[0, 0, 1, 1], [0, 0, 1, 1]]),
    )


def test_energy_distance_is_zero_for_same_sample_and_detects_shift():
    sample = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0], [2.0, 1.0, 3.0]])

    assert energy_distance(sample, sample) == 0.0
    assert energy_distance(sample, sample + [2.0, 0.0, 0.0]) > 0.0
    assert energy_distance(sample, sample + [2.0, 0.0, 0.0], [2.0, 1.0, 1.0]) > 0.0


def test_state_and_paired_summaries_preserve_physical_scales():
    states = _study_data().endpoint_states[0, -1]
    summary = summarize_states(states, LORENZ)
    drift = paired_endpoint_drift(states - [1.0, 2.0, 3.0], states)

    np.testing.assert_allclose(summary["mean"], states.mean(axis=0))
    assert summary["covariance"].shape == (3, 3)
    assert set(summary["quantiles"]) == {"0.05", "0.25", "0.50", "0.75", "0.95"}
    assert summary["lobe_and_symmetry"]["x_positive_fraction"] == 0.5
    assert len(summary["vector_field_balance"]["mean_over_rms"]) == 3
    np.testing.assert_allclose(drift["mean_delta"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(drift["euclidean_distance_rms"], np.sqrt(14.0))


def test_analysis_uses_longest_endpoint_only_as_labeled_reference():
    result = analyze_spinup_study(_study_data(), LORENZ)

    assert "not a known sample from mu0" in result["interpretation"]
    final = result["proposals"]["first"]["comparisons_to_longest_burnin"]["4.0"]
    assert final["energy_distance_physical"] == 0.0
    earlier = result["proposals"]["first"]["comparisons_to_longest_burnin"]["1.0"]
    assert earlier["energy_distance_physical"] > 0.0
    split_reference = result["proposals"]["first"]["endpoint_summaries"]["4.0"]
    assert split_reference["alternating_split_half_energy_reference"][
        "energy_distance_common_standardized"
    ] > 0.0
    assert "first__vs__second" in result["alternate_proposal_sensitivity"]
    replicated = result["replicated_batch_diagnostics"]
    assert replicated["blocks_per_batch"] == 2
    assert replicated["proposals"]["first"]["1.0"][
        "batch_energy_to_longest_burnin_standardized"
    ]["n"] == 2
    assert replicated["alternate_proposal_sensitivity"]["first__vs__second"][
        "1.0"
    ]["same_proposal_between_batch_reference_standardized"]["n"] == 2


def test_persistence_round_trip_includes_raw_arrays_and_hashes(tmp_path):
    data = _study_data()
    derived = analyze_spinup_study(data, LORENZ)
    config = {
        "study_id": "test",
        "lorenz": LORENZ,
        "solver": {"rtol": 1e-8, "atol": 1e-10},
    }

    manifest = persist_spinup_study(
        tmp_path / "run", data, derived, config, {"config_identifier": "test"}
    )

    with np.load(tmp_path / "run/raw_endpoints.npz") as stored:
        np.testing.assert_array_equal(stored["raw_proposals"], data.raw_proposals)
        np.testing.assert_array_equal(stored["endpoint_states"], data.endpoint_states)
        np.testing.assert_array_equal(stored["block_ids"], data.block_ids)
        np.testing.assert_array_equal(stored["batch_ids"], data.batch_ids)
    loaded_manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert loaded_manifest["files"] == manifest["files"]
    assert all(value.startswith("sha256:") for value in manifest["files"].values())
    assert manifest["array_semantics"]["endpoint_states"] == "proposal, burnin_time, block, state"
