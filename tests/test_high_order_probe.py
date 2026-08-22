import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_high_order_probe", REPO / "experiments/run_high_order_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


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
        probe._condition_for(cell, direction, sign * strength)
        for direction in directions
        for strength in strengths
        for sign in (1.0, -1.0)
    }
    assert found == set(range(1, 37))


def test_even_harmonic_uses_paired_unforced_subtraction():
    directions, strengths, vectors = _design()
    blocks = 6
    means = np.zeros((blocks, len(vectors), 3, 6), dtype=complex)
    baseline = 5.0 + 2.0j
    response = 1.0 - 0.5j
    means[:, 0, 0, 2] = baseline
    plus = probe._condition_for({"condition_vectors": vectors}, directions[1], 2.0)
    minus = probe._condition_for({"condition_vectors": vectors}, directions[1], -2.0)
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
    base_vectors = probe._forcing_vectors(
        directions, base_strengths, include_unforced=True
    )
    extension_vectors = probe._forcing_vectors(
        directions, np.asarray((16.0,)), include_unforced=True
    )
    appended = np.concatenate((base_vectors, extension_vectors[1:]), axis=0)
    canonical = probe._forcing_vectors(
        directions, np.asarray((2.0, 4.0, 8.0, 16.0)), include_unforced=True
    )
    permutation = probe._condition_axis_permutation(appended, canonical)
    np.testing.assert_allclose(appended[permutation], canonical, rtol=0.0, atol=0.0)
    assert len(set(permutation.tolist())) == 49


def test_four_strength_response_map_defines_360_cell_bh_family():
    directions, _, _ = _design()
    strengths = np.asarray((2.0, 4.0, 8.0, 16.0))
    vectors = probe._forcing_vectors(directions, strengths, include_unforced=True)
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

    vectors = probe._forcing_vectors(directions, strengths, include_unforced=True)
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
