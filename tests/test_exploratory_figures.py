import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lorenz.exploratory_figures import (
    _errorbar_interval,
    _heldout_second_data,
    _save_figure,
    complex_block_summary,
    load_figure_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_CONFIG = json.loads(
    (REPO_ROOT / "configs/figures/exploratory_existing.json").read_text(
        encoding="utf-8"
    )
)


def test_pinned_artifact_loader_fails_closed_on_manifest_hash():
    specification = dict(FIGURE_CONFIG["artifacts"]["strength"])
    specification["manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_figure_artifact(REPO_ROOT, "strength", specification)


def test_pinned_artifact_loader_checks_schema_and_files():
    artifact = load_figure_artifact(
        REPO_ROOT, "numerical", FIGURE_CONFIG["artifacts"]["numerical"]
    )

    assert artifact.manifest["schema_version"] == 1
    assert artifact.manifest["study_id"] == (
        "numerical_convergence_provisional_protocol_v1"
    )
    assert artifact.raw_path().name == "raw_fourier_summaries.npz"


def test_complex_block_summary_keeps_real_and_imaginary_sampling_axes():
    values = np.asarray([1 + 1j, 3 + 5j, 5 + 3j])
    summary = complex_block_summary(values)

    assert summary["mean_real"] == pytest.approx(3.0)
    assert summary["mean_imaginary"] == pytest.approx(3.0)
    assert summary["mean_magnitude"] == pytest.approx(np.hypot(3, 3))
    expected = np.hypot(values.real.std(ddof=1), values.imag.std(ddof=1))
    assert summary["radial_standard_error"] == pytest.approx(
        expected / np.sqrt(len(values))
    )


def test_heldout_a2_bounds_are_raw_contrast_not_chi2_scale():
    artifact = load_figure_artifact(
        REPO_ROOT, "heldout", FIGURE_CONFIG["artifacts"]["heldout"]
    )
    data = _heldout_second_data(artifact, 2.0, "z")

    assert np.all(data["mean"] >= data["lower"])
    assert np.all(data["mean"] <= data["upper"])
    normalized_upper = data["upper"] / data["strengths"] ** 2
    assert normalized_upper[0] > 1.0
    assert not data["identified"].any()


def test_interval_plot_rejects_inconsistent_magnitude_bounds():
    figure, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="outside"):
            _errorbar_interval(
                axis,
                [1.0],
                [2.0],
                [0.0],
                [1.0],
                color="black",
                label="bad",
            )
    finally:
        plt.close(figure)


def test_figure_save_writes_png_pdf_and_traceable_sidecar(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    record = _save_figure(
        figure,
        tmp_path,
        "smoke",
        {
            "question": "render smoke",
            "quantity": "unitless",
            "uncertainty": "none",
            "sources": [],
            "plotted_data": {"x": [0, 1], "y": [0, 1]},
        },
    )

    assert set(record) == {"figure_id", "png", "pdf", "source"}
    assert (tmp_path / record["png"]).stat().st_size > 0
    assert (tmp_path / record["pdf"]).stat().st_size > 0
    source = json.loads((tmp_path / record["source"]).read_text(encoding="utf-8"))
    assert set(source["outputs"]) == {"smoke.png", "smoke.pdf"}
