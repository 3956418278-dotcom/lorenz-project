#!/usr/bin/env python3
"""Screen forcing frequencies using retained unforced block spectra only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-matplotlib")
import matplotlib.pyplot as plt

from lorenz.response_plots import apply_style


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "outputs/production/high_order_probe_v1/20260820T114248_31cc7991dfff"
)
OUTPUT_SUBDIRECTORY = "frequency_screening"
OUTPUT_NAMES = ("x", "y", "z")


def load_unforced_spectra(artifact: Path) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted((artifact / "block_level" / "omega_6p7").glob("*.npz"))
    if len(paths) != 8:
        raise ValueError(f"expected eight spectrum chunks, found {len(paths)}")
    pieces = []
    grid = None
    block_ids = []
    for path in paths:
        with np.load(path, allow_pickle=False) as chunk:
            spectrum = np.asarray(chunk["spectrum"])
            if spectrum.shape[1:3] != (37, 3) or not np.iscomplexobj(spectrum):
                raise ValueError(f"unexpected spectrum axes in {path.name}")
            pieces.append(spectrum[:, 0])
            chunk_grid = np.asarray(chunk["spectrum_frequency_grid"], dtype=float)
            if grid is None:
                grid = chunk_grid
            elif not np.array_equal(grid, chunk_grid):
                raise ValueError("spectrum grids differ across chunks")
            block_ids.extend(np.asarray(chunk["block_ids"]).tolist())
    spectra = np.concatenate(pieces, axis=0)
    if spectra.shape != (256, 3, len(grid)) or len(set(block_ids)) != 256:
        raise ValueError("unforced block coverage is incomplete")
    return spectra, grid


def smooth_local_baseline(
    median_spectrum: np.ndarray,
    grid: np.ndarray,
    *,
    baseline_window: float,
    baseline_percentile: float,
    baseline_smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smooth lower local envelope in log-spectrum space."""
    frequency_step = float(grid[1] - grid[0])
    window_bins = 2 * int(np.ceil(0.5 * baseline_window / frequency_step)) + 1
    smoothing_bins = float(baseline_smoothing / frequency_step)
    log_spectrum = np.log(np.maximum(median_spectrum, 1e-300))
    log_baseline = np.empty_like(log_spectrum)
    for output in range(3):
        local_lower_envelope = percentile_filter(
            log_spectrum[output], percentile=baseline_percentile,
            size=window_bins, mode="nearest",
        )
        log_baseline[output] = gaussian_filter1d(
            local_lower_envelope, sigma=smoothing_bins, mode="nearest"
        )
    baseline = np.exp(log_baseline)
    return baseline, np.exp(log_spectrum - log_baseline)


def contamination_scan(
    elevation_ratio: np.ndarray,
    grid: np.ndarray,
    *,
    neighborhood_half_width: float,
    omega_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense omega grid, five elevation scores, and responsible output.

    In each output and harmonic neighborhood, the narrow-peak term is the
    95th percentile of spectrum/baseline.  The broad-elevation term is the
    geometric mean of positive log elevation.  Their product penalizes a
    narrow peak, while broad structures receive an additional sustained-
    elevation penalty.  The harmonic score is the maximum over x/y/z.
    """
    omega_min = float(neighborhood_half_width)
    omega_max = float((grid[-1] - neighborhood_half_width) / 5)
    omegas = np.arange(omega_min, omega_max + 0.5 * omega_step, omega_step)
    scores = np.empty((len(omegas), 5), dtype=float)
    responsible_output = np.empty((len(omegas), 5), dtype=np.int8)
    for harmonic in range(1, 6):
        for omega_index, omega in enumerate(omegas):
            center = harmonic * omega
            start = int(np.searchsorted(grid, center - neighborhood_half_width))
            end = int(np.searchsorted(
                grid, center + neighborhood_half_width, side="right"
            ))
            by_output = []
            for output in range(3):
                local = elevation_ratio[output, start:end]
                narrow_peak = float(np.quantile(local, 0.95))
                broad_elevation = float(np.exp(
                    np.mean(np.maximum(np.log(local), 0.0))
                ))
                by_output.append(narrow_peak * broad_elevation)
            scores[omega_index, harmonic - 1] = max(by_output)
            responsible_output[omega_index, harmonic - 1] = int(
                np.argmax(by_output)
            )
    return omegas, scores, responsible_output


def select_candidates(
    omegas: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    minimum_separation: float,
) -> np.ndarray:
    # Strict minimax ranking.  Remaining harmonics are lexicographic
    # tie-breakers only when candidates share the same worst score.
    descending = -np.sort(-scores, axis=1)
    order = sorted(range(len(omegas)), key=lambda index: tuple(descending[index]))
    selected = []
    for index in order:
        if all(
            abs(float(omegas[index] - omegas[previous])) >= minimum_separation
            for previous in selected
        ):
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("unable to select enough separated candidates")
    return np.asarray(selected, dtype=int)


def write_table(
    path: Path,
    omegas: np.ndarray,
    scores: np.ndarray,
    responsible: np.ndarray,
    selected: np.ndarray,
) -> None:
    fields = ["rank", "omega", "worst_peak_contamination"]
    fields.extend(
        f"peak_contamination_{harmonic}omega" for harmonic in range(1, 6)
    )
    fields.extend(f"worst_output_{harmonic}omega" for harmonic in range(1, 6))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(selected, 1):
            row = {
                "rank": rank,
                "omega": float(omegas[index]),
                "worst_peak_contamination": float(scores[index].max()),
            }
            row.update({
                f"peak_contamination_{harmonic}omega": float(
                    scores[index, harmonic - 1]
                )
                for harmonic in range(1, 6)
            })
            row.update({
                f"worst_output_{harmonic}omega": OUTPUT_NAMES[
                    int(responsible[index, harmonic - 1])
                ]
                for harmonic in range(1, 6)
            })
            writer.writerow(row)


def make_plot(
    path: Path,
    grid: np.ndarray,
    quantiles: np.ndarray,
    baseline: np.ndarray,
    elevation_ratio: np.ndarray,
    omegas: np.ndarray,
    selected: np.ndarray,
    neighborhood_half_width: float,
) -> None:
    q05, q50, q95 = quantiles
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(selected)))
    figure, axis = plt.subplots(1, 1, figsize=(15.5, 5.8))
    z = 2
    axis.fill_between(
        grid, q05[z], q95[z], color="#d5d3ce", alpha=0.35,
        linewidth=0, label="z unforced single-block 5-95% band",
    )
    elevated = elevation_ratio[z] >= 1.25
    axis.fill_between(
        grid, baseline[z], q50[z], where=elevated,
        color="#e07a3f", alpha=0.30, linewidth=0,
        label="locally elevated z spectrum (median/baseline >= 1.25)",
    )
    axis.plot(
        grid, q50[z], color="#242424", linewidth=1.15,
        label="z unforced single-block median |S_b|",
    )
    axis.plot(
        grid, baseline[z], color="#c55a27", linewidth=1.0, linestyle="--",
        label="smooth local baseline",
    )
    for candidate_rank, (index, color) in enumerate(zip(selected, colors), 1):
        omega = float(omegas[index])
        for harmonic in range(1, 6):
            frequency = harmonic * omega
            axis.axvline(
                frequency, color=color, linestyle=":", linewidth=0.9,
                alpha=0.85,
                label=f"rank {candidate_rank}: omega={omega:.3f}"
                if harmonic == 1 else None,
            )
            axis.scatter(
                frequency, np.interp(frequency, grid, q50[z]),
                color=color, s=20, edgecolor="white", linewidth=0.5,
                zorder=4,
            )
    axis.set_title("z unforced spectrum: intrinsic peaks and candidate harmonics", loc="left")
    axis.set_yscale("log")
    axis.set_xlim(0.0, grid[-1])
    axis.set_ylabel("single-block fluctuation-spectrum amplitude")
    axis.set_xlabel("Omega [rad / Lorenz time]")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.055), fontsize=7.4,
    )
    figure.suptitle(
        "Local spectral-elevation minimax screening: candidate harmonic sets",
        x=0.02, ha="left", fontsize=11,
    )
    figure.text(
        0.5, 0.012,
        "Dimensionless harmonic contamination is evaluated over +/-"
        f"{neighborhood_half_width:g} rad/time in x/y/z: local 95th-percentile "
        "spectrum/baseline ratio multiplied by sustained positive geometric-mean "
        "elevation. Ranking uses the worst n=1..5 and worst output. Orange fill marks "
        "z regions at least 25% above its smooth local baseline.",
        ha="center", va="bottom", fontsize=7.2, color="#777777", wrap=True,
    )
    figure.subplots_adjust(bottom=0.24, top=0.88, left=0.07, right=0.985)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--neighborhood-half-width", type=float, default=0.5)
    parser.add_argument("--baseline-window", type=float, default=4.0)
    parser.add_argument("--baseline-percentile", type=float, default=25.0)
    parser.add_argument("--baseline-smoothing", type=float, default=0.35)
    parser.add_argument("--omega-step", type=float, default=0.002)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--minimum-separation", type=float, default=0.2)
    arguments = parser.parse_args()
    artifact = arguments.artifact.resolve()
    output_directory = artifact / OUTPUT_SUBDIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    expected = {"unforced_frequency_screening.png", "best_frequency_candidates.csv"}
    unexpected = {path.name for path in output_directory.iterdir()}.difference(expected)
    if unexpected:
        raise ValueError(f"screening output directory is not clean: {sorted(unexpected)}")

    apply_style()
    unforced, grid = load_unforced_spectra(artifact)
    quantiles = np.quantile(np.abs(unforced), (0.05, 0.5, 0.95), axis=0)
    baseline, elevation_ratio = smooth_local_baseline(
        quantiles[1], grid,
        baseline_window=arguments.baseline_window,
        baseline_percentile=arguments.baseline_percentile,
        baseline_smoothing=arguments.baseline_smoothing,
    )
    omegas, scores, responsible = contamination_scan(
        elevation_ratio, grid,
        neighborhood_half_width=arguments.neighborhood_half_width,
        omega_step=arguments.omega_step,
    )
    selected = select_candidates(
        omegas, scores, count=arguments.candidate_count,
        minimum_separation=arguments.minimum_separation,
    )
    table_path = output_directory / "best_frequency_candidates.csv"
    plot_path = output_directory / "unforced_frequency_screening.png"
    write_table(table_path, omegas, scores, responsible, selected)
    make_plot(
        plot_path, grid, quantiles, baseline, elevation_ratio, omegas, selected,
        arguments.neighborhood_half_width,
    )
    print(json.dumps({
        "plot": str(plot_path),
        "table": str(table_path),
        "recommendations": [float(omegas[index]) for index in selected],
        "worst_scores": [float(scores[index].max()) for index in selected],
    }, indent=2))


if __name__ == "__main__":
    main()
