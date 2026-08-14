import json

import numpy as np
import pytest

from lorenz.core import simulate_phase_samples
from lorenz.forced_transient import (
    CONDITION_NAMES,
    ForcedTransientStudyData,
    _nested_cycle_fourier,
    analyze_forced_transient_study,
    persist_forced_transient_study,
    generate_forced_transient_study,
)


CFG = {
    "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "solver": {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11},
}


def _synthetic_data():
    rng = np.random.default_rng(7)
    discard_times = np.array([0.0, 5.0, 10.0])
    harmonics = np.arange(5)
    coefficients = rng.normal(size=(3, 4, 3, 3, 2, 5))
    coefficients = coefficients + 1j * rng.normal(size=coefficients.shape)
    coefficients[-1] = coefficients[-2]
    return ForcedTransientStudyData(
        block_ids=(10, 11, 12, 13),
        discard_times=discard_times,
        harmonics=harmonics,
        omega=2.0,
        n_phase=8,
        cycle_fourier=coefficients,
        phase_offsets=np.array([0.2, 1.2, 2.2]),
        raw_proposals=np.zeros((4, 3)),
        initial_states=np.ones((4, 3)),
        child_spawn_keys=((10,), (11,), (12,), (13,)),
        generation_metadata={"root_entropy": 4},
    )


def test_nested_fourier_uses_each_discard_phase_offset():
    discard_times = np.array([0.0, 0.7])
    harmonics = np.array([0, 1, 2])
    omega = 2.3
    phase = 0.4
    actual = _nested_cycle_fourier(
        (
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            discard_times,
            harmonics,
            omega,
            phase,
            2,
            8,
            CFG,
        )
    )

    assert actual.shape == (2, 3, 2, 3)
    for discard_index, discard in enumerate(discard_times):
        samples = simulate_phase_samples(
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            omega,
            phase,
            discard,
            2,
            8,
            CFG,
        )
        from lorenz.response import phase_fourier

        expected = phase_fourier(
            samples.values, harmonics, samples.phase_offset
        )
        np.testing.assert_allclose(actual[discard_index], expected)


def test_analysis_labels_targets_and_uses_block_paired_differences():
    result = analyze_forced_transient_study(_synthetic_data())

    assert result["condition_order"] == CONDITION_NAMES
    assert result["observation_window"]["same_for_every_discard"] is True
    np.testing.assert_allclose(
        result["observation_window"]["represented_duration_physical"],
        2 * np.pi,
    )
    assert result["contrast_diagnostics"]["odd"]["1"]["role"] == "target"
    assert (
        result["contrast_diagnostics"]["odd"]["2"]["role"]
        == "parity_forbidden_diagnostic"
    )
    assert (
        result["contrast_diagnostics"]["odd"]["3"]["role"]
        == "higher_harmonic_diagnostic"
    )
    assert (
        result["contrast_diagnostics"]["even"]["1"]["role"]
        == "parity_forbidden_diagnostic"
    )
    assert result["contrast_diagnostics"]["even"]["0"]["role"] == "target"
    comparison = result["contrast_diagnostics"]["even"]["2"][
        "comparisons_to_longest_discard"
    ]["5.0"]
    assert comparison["n_block"] == 4
    assert comparison["mean_delta_rms"] == 0.0
    assert comparison["block_difference_rms"] == 0.0
    assert "production sampling covariance" in result["interpretation"]


def test_persistence_preserves_cycle_fourier_and_hashes(tmp_path):
    data = _synthetic_data()
    derived = analyze_forced_transient_study(data)
    config = {
        "study_id": "test",
        "protocol": {"omega": 2.0, "strength": 0.5, "direction": [1, 0, 0]},
        "lorenz": CFG["lorenz"],
        "solver": CFG["solver"],
    }

    manifest = persist_forced_transient_study(
        tmp_path / "run", data, derived, config, {"config_identifier": "test"}
    )

    with np.load(tmp_path / "run/raw_fourier_summaries.npz") as stored:
        np.testing.assert_array_equal(stored["cycle_fourier"], data.cycle_fourier)
        np.testing.assert_array_equal(stored["block_ids"], data.block_ids)
        np.testing.assert_array_equal(stored["harmonics"], data.harmonics)
    loaded = json.loads((tmp_path / "run/manifest.json").read_text())
    assert loaded["protocol_status"] == "provisional_method_validation_only"
    assert loaded["files"] == manifest["files"]
    assert all(value.startswith("sha256:") for value in manifest["files"].values())


def test_generation_rejects_harmonics_that_alias_on_phase_grid():
    config = {
        "discard_times": [0.0, 1.0],
        "harmonics": [0, 1, 2],
        "n_cycle": 1,
        "n_phase": 2,
        "protocol": {
            "omega": 2.0,
            "strength": 0.5,
            "direction": [1.0, 0.0, 0.0],
        },
    }

    with pytest.raises(ValueError, match="alias"):
        generate_forced_transient_study(config)
