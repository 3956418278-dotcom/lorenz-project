import json

import numpy as np
import pytest

from lorenz.numerical_convergence import (
    NumericalConvergenceStudyData,
    _checked_config,
    _integrate_phase_variants,
    analyze_numerical_convergence_study,
    persist_numerical_convergence_study,
)


CFG = {
    "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "solver_profiles": {
        "loose": {"method": "DOP853", "rtol": 1e-7, "atol": 1e-9},
        "tight": {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12},
    },
    "reference_solver": "tight",
    "discard_time": 1.0,
    "observation_cycles": [2, 4],
    "phase_resolutions": [8, 16],
    "harmonics": [0, 1, 2, 3],
    "protocol": {
        "omega": 2.0,
        "strength": 0.5,
        "direction": [1.0, 0.0, 0.0],
        "phase": 0.2,
    },
}


def _synthetic_data():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(5, 3, 3, 1, 4))
    base = base + 1j * rng.normal(size=base.shape)
    cycle_fourier = np.broadcast_to(base, (2, 4, 5, 3, 3, 4, 4)).copy()
    return NumericalConvergenceStudyData(
        block_ids=(10, 11, 12, 13, 14),
        solver_names=("loose", "tight"),
        reference_solver="tight",
        phase_variant_names=("n8_base", "n8_shifted", "n16_base", "n16_shifted"),
        phase_resolutions=np.array([8, 8, 16, 16]),
        phase_shifted=np.array([False, True, False, True]),
        observation_cycles=np.array([2, 4]),
        harmonics=np.array([0, 1, 2, 3]),
        omega=2.0,
        discard_time=10.0,
        master_phase_count=32,
        cycle_fourier=cycle_fourier,
        raw_proposals=np.zeros((5, 3)),
        initial_states=np.ones((5, 3)),
        child_spawn_keys=((10,), (11,), (12,), (13,), (14,)),
        generation_metadata={"root_entropy": 1},
    )


def test_config_requires_nested_alias_safe_phase_grids():
    config = dict(CFG)
    config["phase_resolutions"] = [8, 12]
    with pytest.raises(ValueError, match="nested divisors"):
        _checked_config(config)

    config = dict(CFG)
    config["harmonics"] = [0, 1, 2, 4]
    with pytest.raises(ValueError, match="twice the largest harmonic"):
        _checked_config(config)


def test_phase_variants_include_base_and_half_cell_shift():
    actual = _integrate_phase_variants(
        (
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            2.0,
            0.2,
            1.0,
            2,
            32,
            np.array([8, 16]),
            np.array([0, 1, 2, 3]),
            CFG["lorenz"],
            CFG["solver_profiles"]["tight"],
        )
    )

    assert actual.shape == (4, 3, 2, 4)
    assert np.isfinite(actual).all()
    assert not np.array_equal(actual[0], actual[1])


def test_analysis_separates_window_phase_shift_and_solver_checks():
    result = analyze_numerical_convergence_study(_synthetic_data())

    assert result["reference_settings"]["solver"] == "tight"
    assert result["reference_settings"]["phase_points_per_cycle"] == 16
    np.testing.assert_array_equal(
        result["observation_window_semantics"]["cycle_counts"], [2, 4]
    )
    assert set(result["phase_shifted_subgrid_comparisons"]) == {"8", "16"}
    assert set(result["solver_profile_comparisons_to_reference"]) == {"loose"}
    assert (
        result["observation_window_comparisons_to_longest"]["2"]["odd"]["1"][
            "role"
        ]
        == "target"
    )
    checks = (
        result["observation_window_comparisons_to_longest"]["2"],
        result["phase_refinement_comparisons_to_finest"]["8"],
        result["phase_shifted_subgrid_comparisons"]["16"],
        result["solver_profile_comparisons_to_reference"]["loose"],
    )
    for check in checks:
        assert check["even"]["2"]["drift"]["mean_delta_rms"] == 0.0
    assert "formal bounds" in result["interpretation"]


def test_persistence_preserves_axes_and_hashes(tmp_path):
    data = _synthetic_data()
    derived = analyze_numerical_convergence_study(data)
    config = {
        "study_id": "test",
        "protocol": CFG["protocol"],
        "lorenz": CFG["lorenz"],
        "solver_profiles": CFG["solver_profiles"],
    }
    manifest = persist_numerical_convergence_study(
        tmp_path / "run", data, derived, config, {"config_identifier": "test"}
    )

    with np.load(tmp_path / "run/raw_fourier_summaries.npz") as stored:
        np.testing.assert_array_equal(stored["cycle_fourier"], data.cycle_fourier)
        np.testing.assert_array_equal(stored["block_ids"], data.block_ids)
        np.testing.assert_array_equal(stored["phase_shifted"], data.phase_shifted)
    loaded = json.loads((tmp_path / "run/manifest.json").read_text())
    assert loaded["array_semantics"]["cycle_fourier"].startswith("solver")
    assert loaded["files"] == manifest["files"]
    assert all(value.startswith("sha256:") for value in manifest["files"].values())
