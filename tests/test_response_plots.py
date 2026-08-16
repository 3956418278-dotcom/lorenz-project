import numpy as np
import pytest

from lorenz.response_plots import (
    figure_block_spectra,
    figure_frequency_response,
    figure_harmonic_content,
    figure_raw_trajectories,
    figure_strength_dependence,
    mean_and_standard_error,
    save_figure,
)


def test_mean_and_standard_error_over_blocks():
    values = np.asarray([1.0 + 2j, 3.0 + 4j, 5.0 + 6j, 7.0 + 8j])
    mean, se_real, se_imag = mean_and_standard_error(values)
    assert mean == pytest.approx(4.0 + 5j)
    # SE of [1,3,5,7] with ddof=1 over sqrt(4)
    assert se_real == pytest.approx(np.std([1.0, 3, 5, 7], ddof=1) / 2)
    assert se_imag == pytest.approx(se_real)


def test_block_spectra_requires_complex_for_coherent_mean():
    real = np.random.default_rng(0).random((4, 16))
    grid = np.arange(16, dtype=float)
    with pytest.raises(ValueError):
        figure_block_spectra(real, grid, forcing_frequency=2.0)
    figure = figure_block_spectra(
        real, grid, forcing_frequency=2.0, coherent_mean=False,
    )
    assert figure is not None


def _separate_uncertainty(rng, shape):
    """A (real, imaginary) uncertainty pair with distinct values."""
    return np.abs(rng.normal(size=shape)), np.abs(rng.normal(size=shape))


def test_figure_functions_render_and_save(tmp_path):
    rng = np.random.default_rng(7)
    grid = np.arange(64, dtype=float)
    spectra = rng.normal(size=(10, 64)) + 1j * rng.normal(size=(10, 64))
    figure = figure_block_spectra(
        spectra, grid, forcing_frequency=5.0, harmonics=(1, 2, 3),
        resamples=50,
    )
    save_figure(figure, tmp_path, "test_spectra")

    times = np.arange(200, dtype=float)
    trajectories = rng.normal(size=(3, 40, 200))  # panels, trajectories, time
    figure = figure_raw_trajectories(
        trajectories, times, panel_labels=["x", "y", "z"], max_traces=10,
    )
    save_figure(figure, tmp_path, "test_raw")

    frequencies = np.asarray([0.5, 1.0, 2.0, 4.0])
    values = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    uncertainty = _separate_uncertainty(rng, (4, 2))
    figure = figure_frequency_response(
        frequencies, values, uncertainty, entry_labels=["entry a", "entry b"],
    )
    save_figure(figure, tmp_path, "test_frequency")

    figure = figure_strength_dependence(
        [np.asarray([0.25, 0.5, 1.0, 2.0]), np.asarray([0.5, 1.0, 2.0])],
        [rng.normal(size=(4,)) + 1j * rng.normal(size=(4,)),
         rng.normal(size=(3,)) + 1j * rng.normal(size=(3,))],
        [_separate_uncertainty(rng, (4,)), _separate_uncertainty(rng, (3,))],
        [rng.normal(size=(4,)) + 1j * rng.normal(size=(4,)),
         rng.normal(size=(3,)) + 1j * rng.normal(size=(3,))],
        [_separate_uncertainty(rng, (4,)), _separate_uncertainty(rng, (3,))],
        frequency_labels=["0.5", "8.0"],
    )
    save_figure(figure, tmp_path, "test_strength")

    figure = figure_harmonic_content(
        [
            {
                "harmonics": np.asarray([1.0, 3.0]),
                "point": np.asarray([1 + 2j, 0.1 + 0.2j]),
                "uncertainty": _separate_uncertainty(rng, (2,)),
                "label": "odd",
            },
        ],
        block_values=rng.normal(size=(30, 2)) + 1j * rng.normal(size=(30, 2)),
        block_values_harmonic=1.0,
    )
    save_figure(figure, tmp_path, "test_harmonics")

    assert (tmp_path / "test_spectra.png").is_file()
    assert (tmp_path / "test_raw.png").is_file()
    assert (tmp_path / "test_frequency.png").is_file()
    assert (tmp_path / "test_strength.png").is_file()
    assert (tmp_path / "test_harmonics.png").is_file()


def test_raw_trajectory_plotting_accepts_arbitrary_block_counts(tmp_path):
    """Fig 2 must work unchanged for B=2 and B=1024 populations."""
    rng = np.random.default_rng(11)
    times = np.arange(120, dtype=float)
    for block_count in (1, 2, 64, 1024):
        values = rng.normal(size=(2, block_count, 120))
        figure = figure_raw_trajectories(
            values, times, panel_labels=["a", "b"],
            max_traces=None if block_count <= 64 else 128,
        )
        save_figure(figure, tmp_path, f"raw_b{block_count}")
        assert (tmp_path / f"raw_b{block_count}.png").is_file()


def test_shape_validation():
    with pytest.raises(ValueError):
        figure_frequency_response(
            np.asarray([1.0, 2.0]),
            np.ones((3, 1), dtype=complex),
            (np.ones((3, 1)), np.ones((3, 1))),
        )
    # mismatched uncertainty part shape is rejected
    with pytest.raises(ValueError):
        figure_frequency_response(
            np.asarray([1.0, 2.0]),
            np.ones((2, 1), dtype=complex),
            (np.ones((2, 1)), np.ones((3, 1))),
        )
    with pytest.raises(ValueError):
        figure_strength_dependence(
            [np.asarray([1.0])],
            [np.ones((2,), dtype=complex)],
            [(np.ones((2,)), np.ones((2,)))],
            [np.ones((1,), dtype=complex)],
            [(np.ones((1,)), np.ones((1,)))],
            frequency_labels=["0.5"],
        )
    with pytest.raises(ValueError):
        figure_harmonic_content([])
    with pytest.raises(ValueError):
        mean_and_standard_error(np.ones(1))
