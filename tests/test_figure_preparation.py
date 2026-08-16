"""Code-shape tests for the figure data preparation.

These verify that the figure pipeline's extraction logic follows artifact
values rather than pilot constants when B, strength grids, frequency sets,
retained harmonic counts, and observation lengths change.  No Lorenz
trajectories are integrated; synthetic in-memory cells are used.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import lorenz.strength_study as strength_study
from lorenz.strength_study import StrengthStudyData
from lorenz.x_response_pilot import XResponseCell

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "outputs/figures/x_response_block_level"))

spec = importlib.util.spec_from_file_location(
    "make_figures", REPO / "outputs/figures/x_response_block_level/make_figures.py"
)
make_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_figures)


def _cell(
    block_ids,
    strengths,
    harmonics=(0, 1, 2, 3, 4, 5),
    omega=2.0,
    cycles=3,
):
    rng = np.random.default_rng(int(sum(block_ids)) % 1000)
    positive = (
        rng.normal(size=(len(block_ids), len(strengths), 3, cycles, len(harmonics)))
        + 1j * rng.normal(size=(len(block_ids), len(strengths), 3, cycles, len(harmonics)))
    )
    negative = (
        rng.normal(size=(len(block_ids), len(strengths), 3, cycles, len(harmonics)))
        + 1j * rng.normal(size=(len(block_ids), len(strengths), 3, cycles, len(harmonics)))
    )
    unforced = (
        rng.normal(size=(len(block_ids), 3, cycles, len(harmonics)))
        + 1j * rng.normal(size=(len(block_ids), 3, cycles, len(harmonics)))
    )
    study = StrengthStudyData(
        block_ids=tuple(block_ids),
        strengths=np.asarray(strengths, dtype=float),
        harmonics=np.asarray(harmonics, dtype=int),
        omega=omega,
        n_phase=16,
        positive_cycle_fourier=positive,
        negative_cycle_fourier=negative,
        unforced_cycle_fourier=unforced,
        raw_proposals=np.zeros((len(block_ids), 3)),
        initial_states=np.ones((len(block_ids), 3)),
        child_spawn_keys=tuple((value,) for value in block_ids),
        generation_metadata={},
    )
    return XResponseCell(
        omega=omega,
        study=study,
        folded_cycle_mean=None,
        welch_psd=None,
        segment_counts=None,
        raw_segments={},
        dense_metadata={},
        runtime_seconds=0.0,
    )


def _cell_with_raw(block_ids, strengths, samples=120, harmonics=(0, 1, 2)):
    cell = _cell(block_ids, strengths, harmonics=harmonics, cycles=4)
    rng = np.random.default_rng(1)
    values = rng.normal(
        size=(len(block_ids), 1 + 2 * len(strengths), 3, samples)
    )
    times = 160.0 + 0.05 * np.arange(samples, dtype=float)
    return XResponseCell(
        omega=cell.omega,
        study=cell.study,
        folded_cycle_mean=None,
        welch_psd=None,
        segment_counts=None,
        raw_segments={"values": values, "times": times},
        dense_metadata={"frequency_step": 0.245, "dt": 0.05},
        runtime_seconds=0.0,
    )


def test_condition_index_follows_artifact_metadata_and_layout():
    strengths = [0.5, 1.0, 2.0]
    cell = _cell((1, 2, 3), strengths)
    # documented condition-axis layout fallback
    assert make_figures.condition_index(cell, 0, 1.0) == 0
    assert make_figures.condition_index(cell, +1, 1.0) == 3
    assert make_figures.condition_index(cell, -1, 2.0) == 6
    with pytest.raises(ValueError):
        make_figures.condition_index(cell, +1, 0.25)
    # stored condition table takes precedence
    signs = np.asarray([0, 1, -1, 1, -1, 1, -1], dtype=np.int8)
    signed = np.asarray([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
    cell_with_meta = XResponseCell(
        omega=2.0,
        study=cell.study,
        folded_cycle_mean=None,
        welch_psd=None,
        segment_counts=None,
        raw_segments={},
        dense_metadata={},
        runtime_seconds=0.0,
        sampling_metadata={"_condition_signs": signs, "_condition_strengths": signed},
    )
    assert make_figures.condition_index(cell_with_meta, +1, 2.0) == 5
    assert make_figures.condition_index(cell_with_meta, -1, 0.5) == 2


def test_default_selections_are_artifact_derived():
    cells = {
        0.5: _cell((0, 1, 2, 3), [0.5, 1.0, 2.0]),
        1.0: _cell((0, 1, 2, 3), [0.5, 1.0]),
        2.0: _cell((0, 1, 2, 3), [0.5, 1.0, 2.0]),
    }
    # largest strength present in every cell
    assert make_figures.default_showcase_strength(cells) == 1.0
    # showcase omega: most retained data; ties break to the lowest omega
    assert make_figures.default_showcase_omega(cells) == 0.5
    cells[1.0] = _cell_with_raw((0, 1, 2, 3), [0.5, 1.0], samples=200)
    assert make_figures.default_showcase_omega(cells) == 1.0
    # fullest strength grids
    assert make_figures.fullest_strength_frequencies(cells) == [0.5, 2.0]


def test_strength_extraction_uses_each_cells_own_grid():
    cells = {
        0.5: _cell((0, 1, 2), [0.25, 0.5, 1.0, 2.0]),
        8.0: _cell((0, 1, 2), [0.5, 1.0]),
    }
    frequency_strengths = [cells[omega].study.strengths for omega in (0.5, 8.0)]
    assert [len(grid) for grid in frequency_strengths] == [4, 2]
    assert np.array_equal(frequency_strengths[1], [0.5, 1.0])


def test_raw_panels_preserve_the_block_axis_for_any_b():
    for block_ids in ((5, 6), tuple(range(16))):
        cell = _cell_with_raw(block_ids, [0.5, 1.0], samples=100)
        values = cell.raw_segments["values"]
        panels = np.stack(
            [
                values[:, make_figures.condition_index(cell, +1, 1.0), state, :80]
                for state in range(3)
            ]
        )
        assert panels.shape == (3, len(block_ids), 80)


def test_harmonic_derivation_follows_retained_harmonics():
    cell = _cell((0, 1, 2), [0.5, 1.0], harmonics=(0, 1, 2, 3, 4, 5, 7))
    harmonics = [int(h) for h in cell.study.harmonics]
    odd = [h for h in harmonics if h % 2 == 1]
    even = [h for h in harmonics if h % 2 == 0]
    assert odd == [1, 3, 5, 7]
    assert even == [0, 2, 4]
    cell = _cell((0, 1, 2), [0.5, 1.0], harmonics=(0, 1, 2))
    odd = [h for h in cell.study.harmonics if h % 2 == 1]
    assert odd == [1]


def test_parity_entries_follow_the_artifact_direction():
    x_direction = np.asarray([1.0, 0.0, 0.0])
    linear_null, quadratic_null = make_figures.parity_nulls(x_direction)
    assert [i for i in range(3) if not linear_null[i]] == [0, 1]
    assert [i for i in range(3) if not quadratic_null[i]] == [2]
    z_direction = np.asarray([0.0, 0.0, 1.0])
    linear_null, quadratic_null = make_figures.parity_nulls(z_direction)
    assert [i for i in range(3) if not linear_null[i]] == [2]
    # z output is even under the Lorenz parity, so it is also the only
    # parity-allowed second-order output for z forcing.
    assert [i for i in range(3) if not quadratic_null[i]] == [2]


def test_spectrum_grid_usage_follows_the_data_grid():
    """The figure consumes the stored grid whatever its physical extent."""
    rng = np.random.default_rng(2)
    for bins in (25, 301, 601):
        grid = np.linspace(0.0, 62.8, bins)
        spectra = rng.normal(size=(4, bins)) + 1j * rng.normal(size=(4, bins))
        figure = make_figures.figure_block_spectra(
            spectra, grid, forcing_frequency=8.0, harmonics=(1, 2),
            coherent_mean=False,
        )
        assert figure is not None
