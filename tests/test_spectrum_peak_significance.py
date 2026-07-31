import numpy as np

from lorenz_sine.natural_spectrum import (
    _analyze_saved_spectrum,
    _candidate_peak_specs,
    _detect_peaks_by_coordinate,
    _log_peak_background_ratios,
    _peak_significance_cfg,
)


def test_peak_significance_uses_seed_log_peak_background_ratio():
    freqs = np.arange(16, dtype=float)
    psd_seed = np.ones((6, 3, len(freqs)), dtype=float)
    psd_seed[:, 2, 8] = np.asarray([8.0, 9.0, 10.0, 8.5, 9.5, 10.5])
    peaks = np.asarray([[1, 8, 8.0, 2.0 * np.pi * 8.0, 0.0, 0.0, 1.0, 1.0]])
    cfg = {
        "confidence_level": 0.95,
        "significance_alpha": 0.05,
        "target_omegas": [],
        "peak_significance": {
            "enabled": True,
            "peak_power": "nearest_bin",
            "peak_window_bins": 0,
            "exclude_bins": 1,
            "background_bins": 2,
            "include_detected_peaks": True,
            "include_target_omegas": False,
        },
    }

    candidates = _candidate_peak_specs(freqs, peaks, cfg, cfg["peak_significance"])
    rows, values = _log_peak_background_ratios(
        psd_seed, freqs, candidates, cfg, "test")

    z_row = [row for row in rows if row["coordinate"] == "z"][0]
    assert values.shape == (3, 6)
    assert z_row["n_seed"] == 6
    assert z_row["mean_log_peak_ratio"] > 2.0
    assert z_row["p_value_one_sided"] < 0.001
    assert z_row["p_value_adjusted"] < 0.05
    assert z_row["significant"] is True


def test_coordinate_peak_detection_is_prominence_ranked_and_separated():
    freqs = np.linspace(0.0, 3.0, 601)
    psd_mean = np.ones((3, freqs.size), dtype=float)

    def add_peak(coordinate, center, amplitude, width=0.01):
        psd_mean[coordinate] += amplitude * np.exp(
            -0.5 * ((freqs - center) / width) ** 2
        )

    add_peak(0, 0.50, 8.0)
    add_peak(0, 0.56, 6.0)
    add_peak(1, 0.90, 7.0)
    add_peak(2, 1.32, 10.0)
    add_peak(2, 2.62, 4.0)
    cfg = {
        "f_min": 0.01,
        "prominence": 0.05,
        "peak_min_distance_frequency": 0.10,
        "top_n": 4,
    }

    rows = _detect_peaks_by_coordinate(freqs, psd_mean, cfg)

    x_rows = [row for row in rows if row["coordinate"] == "x"]
    z_rows = [row for row in rows if row["coordinate"] == "z"]
    assert len(x_rows) == 1
    assert [row["rank"] for row in z_rows] == [1, 2]
    assert np.allclose([row["frequency"] for row in z_rows], [1.32, 2.62])
    assert z_rows[0]["prominence"] > z_rows[1]["prominence"]
    assert z_rows[0]["width"] > 0.0
    assert set(z_rows[0]) == {
        "coordinate",
        "rank",
        "frequency",
        "omega",
        "bin",
        "psd_value",
        "prominence",
        "width",
    }


def test_predefined_target_uses_fixed_nearest_bin_not_band_maximum():
    freqs = np.linspace(0.0, 3.0, 301)
    options = _peak_significance_cfg({
        "peak_significance": {
            "predefined_target_frequencies": [2.63],
        },
    })
    misleading_reference = np.zeros_like(freqs)
    misleading_reference[260] = 1e9

    candidates = _candidate_peak_specs(
        freqs,
        [],
        {},
        options,
        reference_psd=misleading_reference,
    )

    assert len(candidates) == 3
    assert {candidate["coordinate"] for candidate in candidates} == {
        "x",
        "y",
        "z",
    }
    assert all(
        candidate["source"] == "predefined_target"
        and candidate["nearest_bin"] == 263
        and candidate["target_frequency"] == 2.63
        for candidate in candidates
    )


def test_saved_spectrum_analysis_uses_disjoint_fixed_seed_split():
    freqs = np.linspace(0.0, 6.3, 64)
    psd_seed = np.ones((6, 3, freqs.size), dtype=float)
    discovery_bins = (10, 20, 30)
    test_only_bins = (12, 22, 32)
    for seed in range(3):
        for coordinate, peak_bin in enumerate(discovery_bins):
            psd_seed[seed, coordinate, peak_bin] = 20.0 + seed
    for seed in range(3, 6):
        for coordinate, peak_bin in enumerate(test_only_bins):
            psd_seed[seed, coordinate, peak_bin] = 30.0 + seed
            psd_seed[seed, coordinate, discovery_bins[coordinate]] = 5.0 + seed
    cfg = {
        "f_min": 0.1,
        "prominence": 0.1,
        "peak_min_distance_frequency": 0.2,
        "top_n": 1,
        "confidence_level": 0.95,
        "significance_alpha": 0.05,
        "peak_significance": {
            "enabled": True,
            "peak_power": "nearest_bin",
            "peak_window_bins": 0,
            "exclude_bins": 1,
            "background_bins": 2,
            "include_detected_peaks": True,
            "predefined_target_frequencies": [2.63],
            "discovery_seed_fraction": 0.5,
            "seed_split_strategy": "contiguous_seed_index",
        },
    }

    result = _analyze_saved_spectrum(
        freqs,
        psd_seed,
        np.mean(psd_seed, axis=0),
        cfg,
    )

    assert np.array_equal(
        result["seed_split"]["discovery_seed_indices"], np.asarray([0, 1, 2])
    )
    assert np.array_equal(
        result["seed_split"]["test_seed_indices"], np.asarray([3, 4, 5])
    )
    assert [
        row["bin"] for row in result["detected_peaks_by_coordinate"]
    ] == list(discovery_bins)
    assert result["peak_significance_values"].shape == (6, 3)
    assert all(
        row["n_seed"] == 3 for row in result["peak_significance_rows"]
    )
