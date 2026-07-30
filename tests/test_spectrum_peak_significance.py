import numpy as np

from lorenz_sine.natural_spectrum import (
    _candidate_peak_specs,
    _log_peak_background_ratios,
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
