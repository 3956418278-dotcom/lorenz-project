import numpy as np

from lorenz import high_order_probe as probe
from lorenz.retention import (
    condition_axis_permutation,
    condition_index,
    paired_condition_vectors,
)


def _design():
    directions = [
        np.asarray(value, dtype=float)
        for value in (
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (2**-0.5, 2**-0.5, 0),
            (2**-0.5, 0, 2**-0.5),
            (0, 2**-0.5, 2**-0.5),
        )
    ]
    strengths = np.asarray((2.0, 4.0, 8.0))
    vectors = [np.zeros(3)]
    vectors.extend(
        sign * strength * direction
        for direction in directions
        for strength in strengths
        for sign in (1.0, -1.0)
    )
    return directions, strengths, np.asarray(vectors)


def test_six_direction_lookup_finds_every_signed_condition():
    directions, strengths, vectors = _design()
    cell = {"condition_vectors": vectors}
    found = {
        condition_index(cell["condition_vectors"], sign * strength * direction)
        for direction in directions
        for strength in strengths
        for sign in (1.0, -1.0)
    }
    assert found == set(range(1, 37))


def test_probe_config_accepts_the_demonstrated_three_axis_design():
    config = {
        "omega": 5.938,
        "strengths": [5.0, 6.0, 7.0],
        "block_count": 256,
        "discard_time": 160.0,
        "n_phase": 32,
        "harmonics": [0, 1, 2, 3, 4, 5],
        "observation_rule": {
            "minimum_cycles": 64,
            "minimum_physical_time": 400.0,
        },
        "protocol": {
            "directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "dense": {
            "dt": 0.05,
            "n_theta_bins": 256,
            "welch_segment": 512,
            "max_psd_bins": 256,
        },
        "retention": {},
    }
    checked = probe.validate_probe_config(config)
    assert len(checked["directions"]) == 3
    assert probe.probe_task_counts(checked)["conditions_per_block"] == 19


def test_even_harmonic_uses_paired_unforced_subtraction():
    directions, strengths, vectors = _design()
    blocks = 6
    means = np.zeros((blocks, len(vectors), 3, 6), dtype=complex)
    baseline = 5.0 + 2.0j
    response = 1.0 - 0.5j
    means[:, 0, 0, 2] = baseline
    plus = condition_index(vectors, 2.0 * directions[1])
    minus = condition_index(vectors, -2.0 * directions[1])
    means[:, plus, 0, 2] = baseline + response
    means[:, minus, 0, 2] = baseline + response
    entries = probe.build_response_map(
        {"means": means, "condition_vectors": vectors},
        {
            "directions": directions,
            "strengths": strengths,
            "harmonics": list(range(6)),
        },
    )
    entry = next(
        item for item in entries
        if item["direction"] == 1 and item["strength"] == 2.0
        and item["output"] == "x" and item["harmonic"] == 2
    )
    assert entry["mean_cos"] == 1.0
    assert entry["mean_sin"] == 0.5


def test_bh_q_values_are_monotone_and_keep_raw_p_values():
    p_values = (0.01, 0.04, 0.03)
    entries = [
        {"harmonic": 1, "hotelling_p": value}
        for value in p_values
    ]
    probe.add_probe_q_values(entries)
    assert [entry["hotelling_p"] for entry in entries] == list(p_values)
    np.testing.assert_allclose(
        [entry["hotelling_q_bh"] for entry in entries],
        [0.03, 0.04, 0.04],
    )
    assert all(entry["detected_q05"] for entry in entries)


def test_extension_counts_only_redundant_unforced_and_new_pairs():
    directions, _, _ = _design()
    checked = {
        "block_count": 256,
        "directions": directions,
        "strengths": np.asarray((2.0, 4.0, 8.0, 16.0)),
        "extension": {"new_strengths": np.asarray((16.0,))},
    }
    counts = probe.probe_task_counts(checked)
    assert counts == {
        "unforced_trajectories": 256,
        "forced_trajectories": 3072,
        "total_condition_integrations": 3328,
        "spinup_integrations": 256,
        "conditions_per_block": 13,
        "combined_conditions_per_block": 49,
    }


def test_extension_condition_axis_is_reordered_to_canonical_four_strength_design():
    directions, base_strengths, _ = _design()
    base_vectors = paired_condition_vectors(
        directions, base_strengths, include_unforced=True
    )
    extension_vectors = paired_condition_vectors(
        directions, np.asarray((16.0,)), include_unforced=True
    )
    appended = np.concatenate((base_vectors, extension_vectors[1:]), axis=0)
    canonical = paired_condition_vectors(
        directions, np.asarray((2.0, 4.0, 8.0, 16.0)), include_unforced=True
    )
    permutation = condition_axis_permutation(appended, canonical)
    np.testing.assert_allclose(appended[permutation], canonical, rtol=0.0, atol=0.0)
    assert len(set(permutation.tolist())) == 49


def test_four_strength_response_map_defines_360_cell_bh_family():
    directions, _, _ = _design()
    strengths = np.asarray((2.0, 4.0, 8.0, 16.0))
    vectors = paired_condition_vectors(directions, strengths, include_unforced=True)
    rng = np.random.default_rng(7)
    means = (
        rng.normal(size=(8, len(vectors), 3, 6))
        + 1j * rng.normal(size=(8, len(vectors), 3, 6))
    )
    entries = probe.build_response_map(
        {"means": means, "condition_vectors": vectors},
        {
            "directions": directions,
            "strengths": strengths,
            "harmonics": list(range(6)),
        },
    )
    probe.add_probe_q_values(entries)
    tested = [entry for entry in entries if entry["harmonic"] != 0]
    assert len(tested) == 360
    assert all("hotelling_q_bh" in entry for entry in tested)


def test_three_strength_chained_extension_counts_and_630_cell_family():
    directions, _, _ = _design()
    strengths = np.asarray((2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0))
    checked = {
        "block_count": 256,
        "directions": directions,
        "strengths": strengths,
        "extension": {"new_strengths": np.asarray((10.0, 12.0, 14.0))},
    }
    counts = probe.probe_task_counts(checked)
    assert counts["conditions_per_block"] == 37
    assert counts["combined_conditions_per_block"] == 85
    assert counts["unforced_trajectories"] == 256
    assert counts["forced_trajectories"] == 9216
    assert counts["total_condition_integrations"] == 9472

    vectors = paired_condition_vectors(directions, strengths, include_unforced=True)
    rng = np.random.default_rng(11)
    means = (
        rng.normal(size=(8, len(vectors), 3, 6))
        + 1j * rng.normal(size=(8, len(vectors), 3, 6))
    )
    entries = probe.build_response_map(
        {"means": means, "condition_vectors": vectors},
        {
            "directions": directions,
            "strengths": strengths,
            "harmonics": list(range(6)),
        },
    )
    probe.add_probe_q_values(entries)
    assert len([entry for entry in entries if entry["harmonic"] != 0]) == 630
