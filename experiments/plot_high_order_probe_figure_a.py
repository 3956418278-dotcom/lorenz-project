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

from lorenz.response_plots import apply_style


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "outputs/production/high_order_probe_v1/20260820T114248_31cc7991dfff"
)
DEFAULT_SUBDIRECTORY = "figure_A_by_direction_strength"
STATE_NAMES = ("x", "y", "z")

INK = "#111111"
MUTED = "#777777"
BACKGROUND_BAND = "#d8d6d1"
NOISE = "#d97941"
NOISE_BAND = "#efb28e"
ODD = "#124f8c"
EVEN = "#16857a"


def _direction_identity(direction: np.ndarray) -> tuple[str, str]:
    direction = np.asarray(direction, dtype=float)
    nonzero = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(nonzero) == 1 and np.isclose(direction[nonzero[0]], 1.0):
        name = STATE_NAMES[nonzero[0]]
        return name, name
    if (
        len(nonzero) == 2
        and np.allclose(direction[nonzero], 2 ** -0.5, atol=1e-12)
    ):
        left, right = (STATE_NAMES[index] for index in nonzero)
        return f"({left}+{right})/sqrt(2)", f"{left}_plus_{right}"
    label = "".join(
        f"{direction[index]:+g}{STATE_NAMES[index]}" for index in nonzero
    )
    slug = label.replace("+", "plus_").replace("-", "minus_").replace(".", "p")
    return label, slug


def _condition_index(condition_vectors: np.ndarray, target: np.ndarray) -> int:
    matches = np.flatnonzero(
        np.all(np.isclose(condition_vectors, target, rtol=0.0, atol=1e-9), axis=1)
    )
    if len(matches) != 1:
        raise ValueError(f"expected one condition matching {target}, found {len(matches)}")
    return int(matches[0])


def load_artifact(artifact: Path) -> dict:
    """Load and validate the saved block spectra and condition means."""
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

    spectra = None
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
                spectra = np.empty(
                    (int(config["block_count"]),) + chunk_spectra.shape[1:],
                    dtype=chunk_spectra.dtype,
                )
            elif not np.array_equal(grid, chunk_grid):
                raise ValueError("spectrum grids differ across chunks")
            end = offset + len(chunk_spectra)
            spectra[offset:end] = chunk_spectra
            ids = np.asarray(chunk["block_ids"], dtype=np.uint32)
            block_ids.extend(ids.tolist())
            offset = end

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
    if not np.array_equal(harmonics, np.arange(6)):
        raise ValueError("expected direct known-frequency harmonics n=0..5")
    if not np.allclose(condition_vectors[0], 0.0):
        raise ValueError("condition zero is not the unforced reference")
    if grid[0] != 0.0 or grid[-1] > 5 * omega or 5 * omega - grid[-1] > grid[1] - grid[0]:
        raise ValueError("continuous spectrum does not cover 0..5 omega as configured")
    return {
        "config": config,
        "omega": omega,
        "grid": grid,
        "spectra": spectra,
        "condition_means": condition_means,
        "condition_vectors": condition_vectors,
        "harmonics": harmonics,
        "strengths": strengths,
    }


def whole_block_noise_floor(
    unforced_spectra: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap |mean S_b| by resampling whole block vectors."""
    blocks, outputs, bins = unforced_spectra.shape
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, blocks, size=(resamples, blocks))
    weights = np.zeros((resamples, blocks), dtype=float)
    rows = np.repeat(np.arange(resamples), blocks)
    np.add.at(weights, (rows, draws.ravel()), 1.0 / blocks)
    result = []
    for output in range(outputs):
        bootstrap_means = weights @ unforced_spectra[:, output]
        amplitude = np.abs(bootstrap_means)
        result.append(np.quantile(amplitude, (0.025, 0.5, 0.975), axis=0))
    quantiles = np.stack(result, axis=1)  # quantile, output, frequency
    return quantiles[1], quantiles[0], quantiles[2]


def response_records(data: dict) -> list[dict]:
    """Build continuous paired contrasts and exact n-omega responses."""
    spectra = data["spectra"]
    means = data["condition_means"]
    condition_vectors = data["condition_vectors"]
    unforced_spectrum = spectra[:, 0]
    unforced_mean = means[:, 0]
    records = []
    for direction_index, configured in enumerate(data["config"]["protocol"]["directions"]):
        direction = np.asarray(configured, dtype=float)
        direction_label, direction_slug = _direction_identity(direction)
        for strength in data["strengths"]:
            strength = float(strength)
            plus = _condition_index(condition_vectors, strength * direction)
            minus = _condition_index(condition_vectors, -strength * direction)
            odd_spectrum = (spectra[:, plus] - spectra[:, minus]) / 2
            even_spectrum = (spectra[:, plus] + spectra[:, minus]) / 2 - unforced_spectrum
            odd_mean = (means[:, plus] - means[:, minus]) / 2
            even_mean = (means[:, plus] + means[:, minus]) / 2 - unforced_mean
            direct = np.empty((3, 5), dtype=complex)
            for harmonic in range(1, 6):
                selected = odd_mean if harmonic % 2 else even_mean
                direct[:, harmonic - 1] = selected[:, :, harmonic].mean(axis=0)
            records.append({
                "direction_index": direction_index,
                "direction_label": direction_label,
                "direction_slug": direction_slug,
                "strength": strength,
                "coherent_odd": odd_spectrum.mean(axis=0),
                "coherent_even": even_spectrum.mean(axis=0),
                "direct": direct,
            })
    return records


def common_y_limits(
    grid: np.ndarray,
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
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)
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
        for harmonic in range(1, 6):
            frequency = harmonic * omega
            color = ODD if harmonic % 2 else EVEN
            axis.axvline(frequency, color=MUTED, linestyle=":", linewidth=0.9)
            axis.scatter(
                frequency, abs(record["direct"][output, harmonic - 1]),
                marker="D", s=30, facecolor=color, edgecolor="white",
                linewidth=0.65, zorder=6, clip_on=False,
                label="exact n*omega response from condition_means"
                if harmonic == 1 else None,
            )
            axis.text(
                frequency, 0.985, f"n={harmonic}",
                transform=axis.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=MUTED,
            )
        axis.set_title(f"{STATE_NAMES[output]} output", loc="left")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 5.0 * omega)
        axis.set_ylim(*y_limits)
    axes[0].set_ylabel("spectral amplitude (state)")
    axes[1].set_xlabel("Omega [rad / Lorenz time]")
    figure.suptitle(
        "Figure A-style spectral diagnostic - "
        f"forcing {record['direction_label']}, h={record['strength']:g}, "
        f"omega={omega:g}, B={len(data['spectra'])}",
        x=0.02, ha="left", fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.055), fontsize=7.5,
    )
    figure.text(
        0.5, 0.012,
        f"Gray: 5-95% quantiles of |S_b| across all 256 unforced blocks. "
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
    parser.add_argument("--output-subdirectory", default=DEFAULT_SUBDIRECTORY)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.resamples < 100:
        raise ValueError("at least 100 whole-block resamples are required")
    artifact = arguments.artifact.resolve()
    output_directory = artifact / arguments.output_subdirectory
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()) and not arguments.overwrite:
        raise ValueError(f"output directory is not clean: {output_directory}")
    if arguments.overwrite and any(
        path.is_dir() or path.suffix.lower() != ".png"
        for path in output_directory.iterdir()
    ):
        raise ValueError("overwrite is limited to the generated PNG-only directory")

    apply_style()
    data = load_artifact(artifact)
    unforced = data["spectra"][:, 0]
    amplitude = np.abs(unforced)
    background_q05, background_q95 = np.quantile(amplitude, (0.05, 0.95), axis=0)
    noise_median, noise_low, noise_high = whole_block_noise_floor(
        unforced, resamples=arguments.resamples, seed=arguments.seed
    )
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
        data["grid"], background_q95, noise_low, noise_high, records
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
    print(json.dumps({
        "artifact": str(artifact),
        "output_directory": str(output_directory),
        "figure_count": len(paths),
        "resamples": arguments.resamples,
        "filenames": [path.name for path in paths],
    }, indent=2))


if __name__ == "__main__":
    main()
