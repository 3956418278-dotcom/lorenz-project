import importlib.util
import json
from pathlib import Path

import numpy as np

from lorenz.retention import paired_condition_vectors


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_high_order_probe_figure_a",
    REPO_ROOT / "experiments/plot_high_order_probe_figure_a.py",
)
figure_a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figure_a)


def test_figure_a_streams_artifact_and_derives_design_axes(tmp_path):
    artifact = tmp_path / "artifact"
    prefix = "omega_1"
    directions = np.asarray(((1, 0, 0), (0, 1, 0)), dtype=float)
    strengths = np.asarray((1.0, 2.0))
    vectors = paired_condition_vectors(directions, strengths)
    harmonics = np.asarray((0, 1, 2))
    block_count = 4
    config = {
        "omega": 1.0,
        "strengths": strengths.tolist(),
        "block_count": block_count,
        "harmonics": harmonics.tolist(),
        "protocol": {"directions": directions.tolist()},
        "retention": {"chunk_blocks": 2},
    }
    artifact.mkdir()
    (artifact / "config_snapshot.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    checkpoint_directory = artifact / "checkpoints"
    checkpoint_directory.mkdir()
    block_ids = np.arange(10, 14, dtype=np.uint32)
    means = np.zeros(
        (block_count, len(vectors), 3, len(harmonics)), dtype=complex
    )
    means[:, 1, :, 1] = 3.0 + 2.0j
    means[:, 2, :, 1] = 1.0 + 0.0j
    np.savez(
        checkpoint_directory / f"{prefix}.npz",
        **{
            f"{prefix}_block_ids": block_ids,
            f"{prefix}_condition_means": means,
            f"{prefix}_condition_vectors": vectors,
            f"{prefix}_harmonics": harmonics,
            f"{prefix}_strengths": strengths,
        },
    )
    spectrum_directory = artifact / "block_level" / prefix
    spectrum_directory.mkdir(parents=True)
    grid = np.asarray((0.0, 1.0, 2.0))
    full_spectra = np.empty((block_count, len(vectors), 3, len(grid)), complex)
    for block in range(block_count):
        full_spectra[block] = block + np.arange(len(vectors))[:, None, None]
    for start in (0, 2):
        np.savez(
            spectrum_directory / f"chunk_{start}.npz",
            block_ids=block_ids[start:start + 2],
            condition_means=means[start:start + 2],
            spectrum=full_spectra[start:start + 2],
            spectrum_frequency_grid=grid,
            spectrum_segment_counts=np.ones((2, len(vectors)), dtype=int),
        )

    data = figure_a.load_artifact(artifact)

    assert data["block_count"] == block_count
    np.testing.assert_array_equal(data["unforced_spectra"], full_spectra[:, 0])
    np.testing.assert_allclose(data["coherent_spectra"], full_spectra.mean(axis=0))
    records = figure_a.response_records(data)
    assert len(records) == len(directions) * len(strengths)
    assert records[0]["harmonics"] == (1, 2)
    np.testing.assert_allclose(records[0]["direct"][:, 0], 1.0 + 1.0j)
