import json
from pathlib import Path

import numpy as np
import pytest

import lorenz.direction_reconnaissance as reconnaissance
from lorenz.direction_design import (
    balanced_quadratic_direction_design,
    minimal_quadratic_direction_design,
)
from lorenz.direction_reconnaissance import (
    DirectionFrequencyData,
    RAW_BASELINE_AXES,
    RAW_FORCED_AXES,
    analyze_direction_reconnaissance,
    build_joint_direction_family,
    direction_parity_null_masks,
    direction_target_contrasts,
    persist_direction_reconnaissance,
    reconstruct_strength_tensors,
)


STRENGTHS = np.asarray((0.25, 0.5, 1.0, 2.0, 3.0, 4.0))
HARMONICS = np.arange(6)


def _contract_quadratic(tensor, direction):
    return np.einsum("oij,i,j->o", tensor, direction, direction)


def _synthetic_data(frequencies=(0.5,), n_block=5):
    design = minimal_quadratic_direction_design()
    linear = np.array(
        [
            [1 + 0.2j, 0.4 - 0.1j, 0],
            [-0.3 + 0.5j, 0.8 + 0.4j, 0],
            [0, 0, 1.2 - 0.2j],
        ]
    )
    second = np.zeros((3, 3, 3), dtype=complex)
    second[0, 0, 2] = second[0, 2, 0] = 0.3 + 0.1j
    second[0, 1, 2] = second[0, 2, 1] = -0.2j
    second[1, 0, 2] = second[1, 2, 0] = 0.15 - 0.05j
    second[1, 1, 2] = second[1, 2, 1] = -0.25 + 0.2j
    second[2, 0, 0] = 0.7 + 0.1j
    second[2, 1, 1] = -0.4 + 0.05j
    second[2, 2, 2] = 0.9 - 0.1j
    second[2, 0, 1] = second[2, 1, 0] = 0.12 + 0.03j
    dc = second.real * 0.6

    positive, negative, baseline = {}, {}, {}
    for omega in frequencies:
        cycles = 2
        shape = (n_block, 6, 6, 3, cycles, len(HARMONICS))
        pos = np.zeros(shape, dtype=complex)
        neg = np.zeros(shape, dtype=complex)
        unforced = np.zeros((n_block, 3, cycles, len(HARMONICS)), dtype=complex)
        unforced[..., 0] = np.array([2.0, 3.0, 4.0])[None, :, None]
        unforced[..., 2] = np.array([0.01j, -0.02j, 0.03j])[None, :, None]
        for direction_index, direction in enumerate(design.directions):
            first_value = linear @ direction
            second_value = _contract_quadratic(second, direction)
            dc_value = _contract_quadratic(dc, direction)
            for strength_index, strength in enumerate(STRENGTHS):
                odd = strength * first_value / (2j)
                a2 = -(strength**2) * second_value / 4
                a0 = unforced[:, :, :, 0] + strength**2 * dc_value[None, :, None]
                pos[:, direction_index, strength_index, :, :, 1] = odd[None, :, None]
                neg[:, direction_index, strength_index, :, :, 1] = -odd[None, :, None]
                pos[:, direction_index, strength_index, :, :, 2] = a2[None, :, None]
                neg[:, direction_index, strength_index, :, :, 2] = a2[None, :, None]
                pos[:, direction_index, strength_index, :, :, 0] = a0
                neg[:, direction_index, strength_index, :, :, 0] = a0
        positive[omega] = pos
        negative[omega] = neg
        baseline[omega] = unforced
    data = DirectionFrequencyData(
        block_ids=tuple(range(100, 100 + n_block)),
        strengths=STRENGTHS,
        harmonics=HARMONICS,
        frequencies=tuple(frequencies),
        design=design,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=baseline,
        raw_proposals=np.zeros((n_block, 3)),
        initial_states=np.ones((n_block, 3)),
        child_spawn_keys=tuple((index,) for index in range(n_block)),
        source={omega: {"baseline": "one"} for omega in frequencies},
    )
    return data, linear, second, dc


def _analysis_config(resamples=29):
    return {
        "bootstrap": {
            "confidence": 0.95,
            "resamples": resamples,
            "root_entropy": 123,
            "batch_size": 7,
            "higher_order_fraction_limit": 0.2,
        }
    }


def _sampling_config(design):
    return {
        "frequencies": [0.75, 3.25],
        "strengths": [0.1, 0.3, 0.9],
        "block_count": 7,
        "harmonics": [0, 1, 2],
        "n_phase": 16,
        "observation_rule": {
            "minimum_cycles": 11,
            "minimum_physical_time": 20.0,
        },
        "direction_design": {
            "name": design.name,
            "direction_names": design.direction_names,
            "directions": design.directions,
        },
        "reuse_heldout_direction": {
            "direction_index": 2,
            "direction_name": design.direction_names[2],
        },
    }


def test_config_owns_direction_and_shared_sampling_instances():
    design = balanced_quadratic_direction_design()
    checked = reconnaissance.validate_direction_reconnaissance_config(
        _sampling_config(design)
    )

    assert checked["frequencies"] == (0.75, 3.25)
    np.testing.assert_array_equal(checked["strengths"], [0.1, 0.3, 0.9])
    assert checked["block_count"] == 7
    assert checked["design"].name == "D9"
    assert checked["design"].quadratic_residual_degrees_of_freedom == 3
    assert checked["reused_direction_index"] == 2


def test_config_rejects_rank_deficient_design_and_reuse_name_mismatch():
    design = minimal_quadratic_direction_design()
    config = _sampling_config(design)
    config["direction_design"] = {
        "name": "basis_only",
        "direction_names": ["x", "y", "z"],
        "directions": np.eye(3),
    }
    with pytest.raises(ValueError, match="full linear and quadratic rank"):
        reconnaissance.validate_direction_reconnaissance_config(config)

    config = _sampling_config(design)
    config["reuse_heldout_direction"]["direction_name"] = "wrong"
    with pytest.raises(ValueError, match="name does not match"):
        reconnaissance.validate_direction_reconnaissance_config(config)


def test_d6_strength_fit_and_tensor_reconstruction_are_exact_by_block():
    data, linear, second, dc = _synthetic_data()
    targets, u2 = direction_target_contrasts(data, 0.5)
    fits = reconstruct_strength_tensors(
        data.block_ids, data.design.directions, data.strengths, targets, u2
    )

    np.testing.assert_allclose(
        fits.leading.first_order, np.broadcast_to(linear, fits.leading.first_order.shape), atol=1e-12
    )
    np.testing.assert_allclose(
        fits.leading.second_harmonic,
        np.broadcast_to(second, fits.leading.second_harmonic.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fits.leading.rectification,
        np.broadcast_to(dc, fits.leading.rectification.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(fits.higher_at_max_strength.first_order, 0, atol=1e-12)
    np.testing.assert_allclose(fits.higher_at_max_strength.second_harmonic, 0, atol=1e-12)


def test_raw_axes_single_baseline_and_estimator_semantics_are_explicit():
    data, _, _, _ = _synthetic_data()
    targets, u2 = direction_target_contrasts(data, 0.5)

    assert RAW_FORCED_AXES == (
        "block", "direction", "strength", "state", "cycle", "harmonic"
    )
    assert RAW_BASELINE_AXES == ("block", "state", "cycle", "harmonic")
    assert data.positive_cycle_fourier[0.5].ndim == 6
    assert data.unforced_cycle_fourier[0.5].ndim == 4
    # A2 remains unchanged when U2 is nonzero; only DC subtracts U0.
    expected_a2 = data.positive_cycle_fourier[0.5].mean(axis=-2)[..., 2]
    np.testing.assert_allclose(targets["even_second_harmonic"], expected_a2)
    np.testing.assert_allclose(u2[:, 0], 0.01j)
    assert np.max(np.abs(targets["even_dc"])) > 0


def test_direction_data_rejects_axes_and_block_metadata_mismatch():
    data, _, _, _ = _synthetic_data()
    broken_positive = dict(data.positive_cycle_fourier)
    broken_positive[0.5] = broken_positive[0.5][:, :-1]
    with pytest.raises(ValueError, match="forced summaries"):
        DirectionFrequencyData(**{**data.__dict__, "positive_cycle_fourier": broken_positive})
    with pytest.raises(ValueError, match="spawn keys"):
        DirectionFrequencyData(**{**data.__dict__, "child_spawn_keys": ((0,),)})


def test_joint_family_uses_shared_block_rows_and_reports_null_sectors(monkeypatch):
    data, _, _, _ = _synthetic_data((0.5, 2.0), n_block=6)
    matrix, members, groups, _ = build_joint_direction_family(data)
    assert matrix.shape[0] == len(data.block_ids)
    assert {member["omega"] for member in members} == {0.5, 2.0}
    captured = {}

    def fake_bootstrap(samples, **kwargs):
        captured["samples"] = samples.copy()
        return (
            samples.mean(axis=0),
            samples.std(axis=0, ddof=1) / np.sqrt(len(samples)),
            np.zeros(kwargs["resamples"]),
            {"root_entropy": kwargs["root_entropy"]},
        )

    monkeypatch.setattr(reconnaissance, "bootstrap_max_statistic", fake_bootstrap)
    result = analyze_direction_reconnaissance(data, _analysis_config())

    np.testing.assert_array_equal(captured["samples"], matrix)
    assert result["family"]["row_block_ids"] == data.block_ids
    assert "shared across all directions" in result["coverage"]["resampling_unit"]
    assert result["design"]["quadratic_residual_degrees_of_freedom"] == 0
    assert not result["design"]["quadratic_residual_diagnostic_available"]
    sectors = {entry["sector"] for entry in result["tensor_sector_diagnosis"]}
    assert sectors == {"allowed", "forbidden"}
    assert any(group["structural_null"] for group in groups)


def test_direction_parity_nulls_only_assert_eigenvector_controls():
    masks = direction_parity_null_masks(minimal_quadratic_direction_design().directions)
    np.testing.assert_array_equal(masks[0]["odd_fundamental"], [False, False, True])
    np.testing.assert_array_equal(masks[2]["odd_fundamental"], [True, True, False])
    np.testing.assert_array_equal(masks[0]["even_second_harmonic"], [True, True, False])
    # xz and yz are not parity eigenvectors, so no directional null is asserted.
    assert masks[4]["direction_parity"] is None
    assert not masks[4]["even_dc"].any()


def test_parent_manifest_mismatch_fails_before_artifact_use(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text("{}\n", encoding="utf-8")
    config = {
        "reuse_heldout_direction": {
            "artifact": "parent",
            "manifest_sha256": "0" * 64,
        }
    }
    with pytest.raises(ValueError, match="manifest hash"):
        reconnaissance._validated_parent(tmp_path, config, {})


def test_generation_integrates_only_five_directions_and_never_a_baseline(monkeypatch):
    design = minimal_quadratic_direction_design()
    frequencies = (0.5, 2.0, 8.0)
    n_block, n_cycle = 2, 3
    block_ids = (50, 51)
    raw_values = {
        "block_ids": np.asarray(block_ids),
        "strengths": STRENGTHS,
        "harmonics": HARMONICS,
        "frequencies": np.asarray(frequencies),
    }
    for omega in frequencies:
        prefix = f"omega_{reconnaissance.frequency_key(omega)}"
        raw_values[f"{prefix}_positive_cycle_fourier"] = np.ones(
            (n_block, 6, 3, n_cycle, 6), dtype=complex
        )
        raw_values[f"{prefix}_negative_cycle_fourier"] = -np.ones(
            (n_block, 6, 3, n_cycle, 6), dtype=complex
        )
        raw_values[f"{prefix}_unforced_cycle_fourier"] = np.zeros(
            (n_block, 3, n_cycle, 6), dtype=complex
        )
        raw_values[f"{prefix}_raw_proposals"] = np.zeros((n_block, 3))
        raw_values[f"{prefix}_initial_states"] = np.ones((n_block, 3))
        raw_values[f"{prefix}_child_spawn_keys"] = np.arange(n_block)[:, None]

    class FakeRaw(dict):
        def close(self):
            pass

    checked = {
        "frequencies": frequencies,
        "strengths": STRENGTHS,
        "harmonics": HARMONICS,
        "block_count": n_block,
        "cycles": {omega: n_cycle for omega in frequencies},
        "design": design,
        "reused_direction_index": 2,
    }
    monkeypatch.setattr(
        reconnaissance, "validate_direction_reconnaissance_config", lambda config: checked
    )
    monkeypatch.setattr(
        reconnaissance,
        "_validated_parent",
        lambda repo_root, config, checked: (
            {"files": {"raw_frequency_summaries.npz": "sha256:parent"}},
            FakeRaw(raw_values),
        ),
    )
    calls = []

    def fake_integrate(block_ids_arg, states, forcing_vectors, sampling, numerical, workers):
        calls.append(np.asarray(forcing_vectors))
        return np.zeros((n_block, 60, 3, n_cycle, 6), dtype=complex)

    monkeypatch.setattr(
        reconnaissance, "integrate_cycle_fourier_conditions", fake_integrate
    )
    config = {
        "initial_ensemble": {"block_id_start": 50},
        "protocol": {"phase": 0.0},
        "discard_time": 160.0,
        "n_phase": 32,
        "lorenz": {},
        "solver": {},
        "workers": 1,
        "reuse_heldout_direction": {"artifact": "parent"},
    }
    data = reconnaissance.generate_direction_reconnaissance(config, Path("."))

    assert len(calls) == 3
    assert all(call.shape == (5 * 6 * 2, 3) for call in calls)
    assert all(not np.any(np.all(call == 0, axis=1)) for call in calls)
    assert data.positive_cycle_fourier[0.5].shape[:3] == (2, 6, 6)
    np.testing.assert_array_equal(data.positive_cycle_fourier[0.5][:, 2], 1)
    np.testing.assert_array_equal(data.positive_cycle_fourier[0.5][:, 0], 0)
    assert data.unforced_cycle_fourier[0.5].shape == (2, 3, 3, 6)


def test_persistence_records_axes_and_active_fingerprint(tmp_path):
    data, _, _, _ = _synthetic_data()
    derived = analyze_direction_reconnaissance(data, _analysis_config(resamples=9))
    output = tmp_path / "artifact"
    config = {"study_id": "synthetic_direction"}
    provenance = {
        "code_identifiers": {"src/lorenz/direction_reconnaissance.py": "sha256:active"},
        "environment": {"python": "test"},
    }
    manifest = persist_direction_reconnaissance(
        output, data, derived, config, provenance
    )

    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["provenance"] == provenance
    assert stored["array_semantics"]["forced_axes"] == list(RAW_FORCED_AXES)
    assert stored["array_semantics"]["unforced_axes"] == list(RAW_BASELINE_AXES)
    assert manifest["files"]["raw_direction_summaries.npz"].startswith("sha256:")
