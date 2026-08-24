#!/usr/bin/env python3
"""Rebuild Figure-A-style spectra from a completed high-order probe.

This is an artifact view: it performs no Lorenz integrations and consumes
only the retained per-block complex spectra plus the direct known-frequency
condition means.  One PNG is produced for every direction/strength pair.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-matplotlib")
import matplotlib.pyplot as plt

from lorenz.artifacts import file_sha256, write_json_atomic
from lorenz.direction_design import direction_identity
from lorenz.response import harmonic_order_contrast, paired_order_contrasts
from lorenz.response_plots import apply_style, ensemble_noise_floor
from lorenz.retention import OBSERVABLE_LABELS, paired_condition_indices


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "outputs/production/high_order_probe_omega_6p7_h10_h12_h14_h16_extension_v1"
    / "20260822T061155_75ca70131a4b"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/figures/high_order_probe_figure_a"

INK = "#111111"
MUTED = "#777777"
BACKGROUND_BAND = "#d8d6d1"
NOISE = "#d97941"
NOISE_BAND = "#efb28e"
ODD = "#124f8c"
EVEN = "#16857a"


def load_artifact(artifact: Path) -> dict:
    """Stream a completed artifact into the statistics needed by this view."""
    config = json.loads((artifact / "config_snapshot.json").read_text(encoding="utf-8"))
    omega = float(config["omega"])
    strength_count = len(config["strengths"])
    direction_count = len(config["protocol"]["directions"])
    expected_conditions = 1 + 2 * strength_count * direction_count
    expected_chunks = int(np.ceil(
        int(config["block_count"]) / int(config["retention"]["chunk_blocks"])
    ))
    prefix = f"omega_{format(omega, '.12g').replace('-', 'm').replace('.', 'p')}"
    chunk_paths = sorted((artifact / "block_level" / prefix).glob("*.npz"))
    if len(chunk_paths) != expected_chunks:
        raise ValueError(
            f"expected {expected_chunks} block-level chunks, found {len(chunk_paths)}"
        )

    coherent_spectrum_sum = None
    unforced_spectra = []
    grid = None
    block_ids = []
    offset = 0
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as chunk:
            required = {
                "block_ids", "condition_means", "spectrum",
                "spectrum_frequency_grid", "spectrum_segment_counts",
            }
            missing = required.difference(chunk.files)
            if missing:
                raise ValueError(f"{path.name} lacks arrays {sorted(missing)}")
            chunk_spectra = np.asarray(chunk["spectrum"])
            if (
                chunk_spectra.shape[1:3] != (expected_conditions, 3)
                or not np.iscomplexobj(chunk_spectra)
            ):
                raise ValueError(f"unexpected complex spectrum axes in {path.name}")
            chunk_grid = np.asarray(chunk["spectrum_frequency_grid"], dtype=float)
            if grid is None:
                grid = chunk_grid
                coherent_spectrum_sum = np.zeros(
                    chunk_spectra.shape[1:], dtype=np.complex128
                )
            elif not np.array_equal(grid, chunk_grid):
                raise ValueError("spectrum grids differ across chunks")
            coherent_spectrum_sum += chunk_spectra.sum(axis=0, dtype=np.complex128)
            unforced_spectra.append(np.array(chunk_spectra[:, 0], copy=True))
            ids = np.asarray(chunk["block_ids"], dtype=np.uint32)
            block_ids.extend(ids.tolist())
            offset += len(chunk_spectra)

    if offset != int(config["block_count"]) or len(set(block_ids)) != offset:
        raise ValueError("chunk block coverage is incomplete or duplicated")
    checkpoint_path = artifact / "checkpoints" / f"{prefix}.npz"
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        checkpoint_ids = np.asarray(checkpoint[f"{prefix}_block_ids"], dtype=np.uint32)
        condition_means = np.asarray(checkpoint[f"{prefix}_condition_means"])
        condition_vectors = np.asarray(checkpoint[f"{prefix}_condition_vectors"], dtype=float)
        harmonics = np.asarray(checkpoint[f"{prefix}_harmonics"], dtype=int)
        strengths = np.asarray(checkpoint[f"{prefix}_strengths"], dtype=float)
    if not np.array_equal(checkpoint_ids, np.asarray(block_ids, dtype=np.uint32)):
        raise ValueError("checkpoint and spectrum block identities differ")
    if condition_means.shape != (
        offset, expected_conditions, 3, len(harmonics)
    ):
        raise ValueError("condition mean axes do not match the retained spectra")
    if not np.iscomplexobj(condition_means):
        raise ValueError("direct known-frequency condition means must be complex")
    if 0 not in harmonics or not np.any(harmonics > 0):
        raise ValueError("artifact must retain DC and at least one positive harmonic")
    if not np.allclose(condition_vectors[0], 0.0):
        raise ValueError("condition zero is not the unforced reference")
    max_frequency = int(np.max(harmonics[harmonics > 0])) * omega
    if (
        grid[0] != 0.0
        or grid[-1] > max_frequency
        or max_frequency - grid[-1] > grid[1] - grid[0]
    ):
        raise ValueError("continuous spectrum does not cover the retained harmonics")
    return {
        "config": config,
        "omega": omega,
        "grid": grid,
        "block_count": offset,
        "unforced_spectra": np.concatenate(unforced_spectra, axis=0),
        "coherent_spectra": coherent_spectrum_sum / offset,
        "condition_means": condition_means,
        "condition_vectors": condition_vectors,
        "harmonics": harmonics,
        "strengths": strengths,
        "state_names": OBSERVABLE_LABELS,
    }


def response_records(data: dict) -> list[dict]:
    """Build continuous paired contrasts and exact n-omega responses."""
    spectra = data["coherent_spectra"]
    means = data["condition_means"].mean(axis=0)
    condition_vectors = data["condition_vectors"]
    unforced_spectrum = spectra[0]
    unforced_mean = means[0]
    harmonic_indices = {
        int(harmonic): index for index, harmonic in enumerate(data["harmonics"])
    }
    plot_harmonics = tuple(sorted(h for h in harmonic_indices if h > 0))
    records = []
    for direction_index, configured in enumerate(data["config"]["protocol"]["directions"]):
        direction = np.asarray(configured, dtype=float)
        direction_label, direction_slug = direction_identity(direction)
        for strength in data["strengths"]:
            strength = float(strength)
            plus, minus = paired_condition_indices(
                condition_vectors, direction, strength
            )
            odd_spectrum, even_spectrum = paired_order_contrasts(
                spectra[plus], spectra[minus], unforced_spectrum
            )
            odd_mean, even_mean = paired_order_contrasts(
                means[plus], means[minus], unforced_mean
            )
            direct = np.empty(
                (len(data["state_names"]), len(plot_harmonics)), dtype=complex
            )
            for harmonic_index, harmonic in enumerate(plot_harmonics):
                selected = harmonic_order_contrast(odd_mean, even_mean, harmonic)
                direct[:, harmonic_index] = selected[:, harmonic_indices[harmonic]]
            records.append({
                "direction_index": direction_index,
                "direction_label": direction_label,
                "direction_slug": direction_slug,
                "strength": strength,
                "harmonics": plot_harmonics,
                "coherent_odd": odd_spectrum,
                "coherent_even": even_spectrum,
                "direct": direct,
            })
    return records


def common_y_limits(
    background_q95: np.ndarray,
    noise_low: np.ndarray,
    noise_high: np.ndarray,
    records: list[dict],
) -> tuple[float, float]:
    continuous = [background_q95[:, 1:], noise_low[:, 1:], noise_high[:, 1:]]
    exact = []
    for record in records:
        continuous.extend((
            np.abs(record["coherent_odd"][:, 1:]),
            np.abs(record["coherent_even"][:, 1:]),
        ))
        exact.append(np.abs(record["direct"]).ravel())
    positive = np.concatenate([value.ravel() for value in continuous])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    exact_positive = np.concatenate(exact)
    exact_positive = exact_positive[exact_positive > 0]
    lower_reference = float(np.quantile(positive, 0.001))
    if len(exact_positive):
        lower_reference = min(lower_reference, float(exact_positive.min()) / 2)
    upper_reference = float(max(value.max() for value in continuous))
    lower = 10.0 ** np.floor(np.log10(lower_reference))
    upper = 10.0 ** np.ceil(np.log10(1.2 * upper_reference))
    return lower, upper


def plot_record(
    record: dict,
    data: dict,
    background_q05: np.ndarray,
    background_q95: np.ndarray,
    noise_median: np.ndarray,
    noise_low: np.ndarray,
    noise_high: np.ndarray,
    y_limits: tuple[float, float],
    output_directory: Path,
    resamples: int,
) -> Path:
    omega = data["omega"]
    grid = data["grid"]
    state_names = data["state_names"]
    figure, axes = plt.subplots(
        1, len(state_names), figsize=(5.2 * len(state_names), 5.2),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]
    for output, axis in enumerate(axes):
        axis.fill_between(
            grid, background_q05[output], background_q95[output],
            color=BACKGROUND_BAND, alpha=0.30, linewidth=0, zorder=1,
            label="single-block unforced background (5-95%)",
        )
        axis.fill_between(
            grid, noise_low[output], noise_high[output],
            color=NOISE_BAND, alpha=0.18, linewidth=0, zorder=2,
            label="ensemble noise floor (whole-block bootstrap 95%)",
        )
        axis.plot(
            grid, noise_median[output], color=NOISE, linewidth=0.6, zorder=7,
            label="ensemble noise floor median",
        )
        axis.plot(
            grid, np.abs(record["coherent_odd"][output]),
            color=ODD, linewidth=1.25, zorder=5,
            label="coherent odd response |mean (S+ - S-)/2|",
        )
        axis.plot(
            grid, np.abs(record["coherent_even"][output]),
            color=EVEN, linewidth=1.15, zorder=5,
            label="coherent even response |mean ((S+ + S-)/2 - S0)|",
        )
        for harmonic_index, harmonic in enumerate(record["harmonics"]):
            frequency = harmonic * omega
            color = ODD if harmonic % 2 else EVEN
            axis.axvline(frequency, color=MUTED, linestyle=":", linewidth=0.9)
            axis.scatter(
                frequency, abs(record["direct"][output, harmonic_index]),
                marker="D", s=30, facecolor=color, edgecolor="white",
                linewidth=0.65, zorder=6, clip_on=False,
                label="exact n*omega response from condition_means"
                if harmonic == record["harmonics"][0] else None,
            )
            axis.text(
                frequency, 0.985, f"n={harmonic}",
                transform=axis.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=MUTED,
            )
        axis.set_title(f"{state_names[output]} output", loc="left")
        axis.set_yscale("log")
        axis.set_xlim(0.0, max(record["harmonics"]) * omega)
        axis.set_ylim(*y_limits)
    axes[0].set_ylabel("spectral amplitude (state)")
    axes[len(axes) // 2].set_xlabel("Omega [rad / Lorenz time]")
    figure.suptitle(
        "Figure A-style spectral diagnostic - "
        f"forcing {record['direction_label']}, h={record['strength']:g}, "
        f"omega={omega:g}, B={data['block_count']}",
        x=0.02, ha="left", fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.055), fontsize=7.5,
    )
    figure.text(
        0.5, 0.012,
        f"Gray: 5-95% quantiles of |S_b| across all {data['block_count']} "
        "unforced blocks. "
        f"Orange: median and central 95% band of |mean_b S_b| from {resamples} "
        "whole-block bootstrap resamples. Blue/green: complex paired response "
        "spectra averaged across blocks before magnitude. Diamonds: direct known-frequency "
        "condition_means (odd contrast for odd n, even-minus-unforced for even n), never "
        "nearest-bin values. All continuous curves are mean-removed fluctuation spectra.",
        ha="center", va="bottom", fontsize=7.2, color=MUTED, wrap=True,
    )
    figure.subplots_adjust(bottom=0.24, top=0.88, left=0.07, right=0.985, wspace=0.10)
    filename = (
        f"figure_A_direction_{record['direction_slug']}_h_"
        f"{format(record['strength'], 'g').replace('.', 'p')}.png"
    )
    path = output_directory / filename
    # Use the fixed figure canvas.  A tight bounding box can be distorted by
    # narrow off-harmonic response peaks in individual direction/strength
    # cases, producing inconsistent title clipping across the figure set.
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.resamples < 100:
        raise ValueError("at least 100 whole-block resamples are required")
    artifact = arguments.artifact.resolve()
    output_directory = (
        arguments.output_directory.resolve()
        if arguments.output_directory is not None
        else DEFAULT_OUTPUT_ROOT / f"{artifact.parent.name}_{artifact.name}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()) and not arguments.overwrite:
        raise ValueError(f"output directory is not clean: {output_directory}")
    if arguments.overwrite and any(
        path.is_dir()
        or (path.suffix.lower() != ".png" and path.name != "view_manifest.json")
        for path in output_directory.iterdir()
    ):
        raise ValueError("overwrite is limited to generated PNG and view-manifest files")

    apply_style()
    data = load_artifact(artifact)
    unforced = data["unforced_spectra"]
    amplitude = np.abs(unforced)
    background_q05, background_q95 = np.quantile(amplitude, (0.05, 0.95), axis=0)
    noise = [
        ensemble_noise_floor(
            unforced[:, output], resamples=arguments.resamples,
            seed=arguments.seed,
        )
        for output in range(unforced.shape[1])
    ]
    noise_median = np.stack([item[0] for item in noise])
    noise_low = np.stack([item[1] for item in noise])
    noise_high = np.stack([item[2] for item in noise])
    records = response_records(data)
    expected_records = (
        len(data["config"]["protocol"]["directions"])
        * len(data["strengths"])
    )
    if len(records) != expected_records:
        raise ValueError(
            f"expected {expected_records} direction/strength records, "
            f"found {len(records)}"
        )
    y_limits = common_y_limits(
        background_q95, noise_low, noise_high, records
    )
    paths = [
        plot_record(
            record, data, background_q05, background_q95,
            noise_median, noise_low, noise_high, y_limits,
            output_directory, arguments.resamples,
        )
        for record in records
    ]
    if len(paths) != expected_records or len({path.name for path in paths}) != expected_records:
        raise ValueError("figure output names are incomplete or duplicated")
    summary = {
        "artifact": str(artifact),
        "artifact_manifest_identifier": (
            f"sha256:{file_sha256(artifact / 'manifest.json')}"
        ),
        "output_directory": str(output_directory),
        "figure_count": len(paths),
        "resamples": arguments.resamples,
        "seed": arguments.seed,
        "block_count": data["block_count"],
        "harmonics": [int(value) for value in data["harmonics"] if value > 0],
        "filenames": [path.name for path in paths],
    }
    write_json_atomic(output_directory / "view_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
