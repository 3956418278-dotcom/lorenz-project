"""Self-contained unit tests for GLS regression and end-to-end probe checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.regression import (
    crossed_block_gls_fit,
    joint_crossed_block_model_comparison,
    power_series_operator,
)
from lorenz import high_order_probe as runner
from lorenz.artifacts import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_power_series_operator_and_gls_recover_exact_polynomial_coefficients():
    rng = np.random.default_rng(12345)
    strengths = np.array([0.5, 1.0, 1.5, 2.0])
    design, coeff_map = power_series_operator(strengths, (2, 4))
    np.testing.assert_allclose(coeff_map @ design, np.eye(2), atol=1e-13)

    true_coeff = np.array([1.25, -0.15])
    noise = rng.normal(0.0, 1e-4, size=(200, len(strengths)))
    noise -= noise.mean(axis=0, keepdims=True)
    block_values = (design @ true_coeff)[None, :] + noise

    fit = crossed_block_gls_fit(block_values, strengths, (2, 4))
    np.testing.assert_allclose(fit["mean_coefficients"], true_coeff, atol=1e-12)
    np.testing.assert_allclose(fit["residuals"], 0.0, atol=1e-12)


def test_joint_crossed_block_model_comparison_detects_missing_higher_order():
    rng = np.random.default_rng(54321)
    strengths = np.array([0.5, 1.0, 1.5, 2.0])
    true_mean = np.column_stack([
        1.0 * strengths**2 + 0.5 * strengths**4,
        -0.5 * strengths**2 + 0.3 * strengths**4,
    ])
    noise = rng.normal(0.0, 0.01, size=(120, len(strengths), 2))
    features = true_mean[None, :, :] + noise

    comparisons = joint_crossed_block_model_comparison(
        features, strengths, ((2,), (2, 4))
    )
    assert len(comparisons) == 2
    assert comparisons[0]["lack_of_fit_p"] < 1e-6
    assert comparisons[1]["nested_improvement_p"] < 1e-6
    assert comparisons[1]["lack_of_fit_p"] > 0.01


@pytest.mark.parametrize(
    "config_name",
    ["example_single_axis.json", "example_mixed_pair.json"],
)
def test_example_configs_dry_run_and_checkpoint_round_trip(config_name, tmp_path):
    config = json.loads((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    summary = runner.dry_run_summary(config)
    assert summary["block_count"] == config["block_count"]
    assert summary["conditions_per_block"] > 1

    config["output_root"] = str(tmp_path / "runs")
    config["observation_rule"]["minimum_cycles"] = 2
    config["observation_rule"]["minimum_physical_time"] = 0.5
    config["discard_time"] = 0.2
    config["initial_ensemble"]["spinup_time"] = 0.2
    config_path = tmp_path / config_name
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifact = runner.run_probe(config_path)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        assert "sha256:" + file_sha256(artifact / relative) == expected

    with pytest.raises(ValueError, match="immutable"):
        runner.run_probe(config_path, resume_dir=artifact)
