import csv

import numpy as np

from lorenz_sine.amplitude_scan import _write_tables
from lorenz_sine.statistics import one_sample_t_test


def test_one_sample_t_test_signed_coefficients():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    result = one_sample_t_test(values, confidence_level=0.95, axis=0)

    assert result["n"] == 4
    assert result["df"] == 3
    assert np.isclose(result["mean"], 2.5)
    assert result["p_value"] < 0.05
    assert result["ci_low"] > 0.0


def test_signed_fourier_table_uses_target_components(tmp_path):
    shape = (1, 1, 3, 3, 6, 2)
    test_shape = (1, 1, 3, 3, 6, 2)
    cfg = {
        "forcing_directions": [0],
        "fourier_kmax": 5,
        "confidence_level": 0.95,
    }
    tests = {
        "n": 4,
        "df": 3,
        "alpha": 0.05,
        "confidence_level": 0.95,
        "mean": np.zeros(test_shape),
        "std": np.ones(test_shape),
        "se": np.ones(test_shape),
        "t_statistic": np.zeros(test_shape),
        "p_value": np.ones(test_shape),
        "ci_low": -np.ones(test_shape),
        "ci_high": np.ones(test_shape),
        "significant": np.zeros(test_shape, dtype=bool),
    }
    result = {
        "frequencies": np.asarray([1.0]),
        "amplitudes": np.asarray([0.1]),
        "odd_norm_mean": np.zeros((1, 1, 3, 3)),
        "even_norm_mean": np.zeros((1, 1, 3, 3)),
        "odd_scaled_norm_mean": np.zeros((1, 1, 3, 3)),
        "even_scaled_norm_mean": np.zeros((1, 1, 3, 3)),
        "odd_leakage_mean": np.zeros((1, 1, 3, 3)),
        "even_leakage_mean": np.zeros((1, 1, 3, 3)),
        "odd_fourier_t_test": tests,
        "even_fourier_t_test": tests,
        "sampling_metadata": [],
    }

    _write_tables(tmp_path, result, cfg)

    fourier_path = tmp_path / "tables" / "signed_fourier_t_tests.csv"
    fft_path = tmp_path / "tables" / "signed_fft_t_tests.csv"
    assert fourier_path.read_text() == fft_path.read_text()

    with fft_path.open() as fp:
        rows = list(csv.DictReader(fp))

    observed = {
        (row["response_group"], int(row["harmonic"]), row["component"])
        for row in rows
    }
    assert observed == {
        ("odd", 1, "cos"),
        ("odd", 1, "sin"),
        ("odd", 3, "cos"),
        ("odd", 3, "sin"),
        ("odd", 5, "cos"),
        ("odd", 5, "sin"),
        ("even", 0, "dc"),
        ("even", 2, "cos"),
        ("even", 2, "sin"),
        ("even", 4, "cos"),
        ("even", 4, "sin"),
    }
