import numpy as np

import lorenz.strength_study as strength_study
from lorenz.strength_study import (
    CycleFourierSampling,
    FrequencyStudyData,
    StrengthStudyData,
    dc_components,
    frequency_key,
    integrate_cycle_fourier_conditions,
    second_harmonic_estimators,
    strength_fourier_components,
)


def _study(block_ids=(4, 9)):
    strengths = np.array([0.5, 1.0])
    harmonics = np.array([0, 1, 2])
    shape = (len(block_ids), len(strengths), 3, 2, len(harmonics))
    positive = np.full(shape, 7 + 0j)
    negative = np.full(shape, 3 + 0j)
    unforced = np.full((len(block_ids), 3, 2, len(harmonics)), 1 + 0j)
    return StrengthStudyData(
        block_ids=block_ids,
        strengths=strengths,
        harmonics=harmonics,
        omega=2.0,
        n_phase=8,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=np.zeros((len(block_ids), 3)),
        initial_states=np.ones((len(block_ids), 3)),
        child_spawn_keys=tuple((value,) for value in block_ids),
        generation_metadata={},
    )


def test_condition_integration_preserves_block_and_condition_axes(monkeypatch):
    sampling = CycleFourierSampling(
        omega=2.0,
        phase=0.0,
        discard_time=4.0,
        n_cycle=2,
        n_phase=8,
        harmonics=np.array([0, 1, 2]),
    )
    block_ids = (11, 27)
    initial_states = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    forcing_vectors = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    def fake_integrate(arguments):
        initial_state, forcing, local_sampling, _ = arguments
        value = initial_state + forcing
        return np.broadcast_to(
            value[:, None, None],
            (3, local_sampling.n_cycle, len(local_sampling.harmonics)),
        )

    monkeypatch.setattr(strength_study, "_integrate_cycle_fourier", fake_integrate)
    result = integrate_cycle_fourier_conditions(
        block_ids,
        initial_states,
        forcing_vectors,
        sampling,
        {"lorenz": {}, "solver": {}},
    )

    assert result.shape == (2, 2, 3, 2, 3)
    np.testing.assert_allclose(result[0, 0, :, 0, 0], initial_states[0])
    np.testing.assert_allclose(
        result[1, 1, :, 0, 0], initial_states[1] + forcing_vectors[1]
    )


def test_explicit_components_keep_a2_u2_and_dc_semantics_separate():
    study = _study()
    components = strength_fourier_components(study)
    np.testing.assert_allclose(components.odd, 2)
    np.testing.assert_allclose(components.forced_even, 5)
    np.testing.assert_allclose(components.unforced, 1)

    second = second_harmonic_estimators(study)
    np.testing.assert_allclose(second["alternative_A2"], 5)
    np.testing.assert_allclose(second["unforced_U2_diagnostic"], 1)
    np.testing.assert_allclose(second["current_E2"], 4)

    dc = dc_components(study)
    np.testing.assert_allclose(dc["forced_even_A0"], 5)
    np.testing.assert_allclose(dc["unforced_U0"], 1)
    np.testing.assert_allclose(dc["current_E0"], 4)


def test_frequency_container_preserves_crossed_blocks_and_stable_keys():
    first = _study()
    second = StrengthStudyData(**{**first.__dict__, "omega": 4.0})
    data = FrequencyStudyData(
        block_ids=first.block_ids,
        strengths=first.strengths,
        harmonics=first.harmonics,
        frequencies=(2.0, 4.0),
        studies={2.0: first, 4.0: second},
        frequency_runtime_seconds={2.0: 1.0, 4.0: 2.0},
        source={2.0: {}, 4.0: {}},
    )

    assert data.studies[4.0].block_ids == data.block_ids
    assert frequency_key(0.5) == "0p5"
    assert frequency_key(2.0) == "2"
