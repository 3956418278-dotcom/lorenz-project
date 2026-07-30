"""Post-process an unforced spectrum run with harmonic-band peak tests."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from lorenz_sine import storage  # noqa: E402
from lorenz_sine.natural_spectrum import (  # noqa: E402
    _candidate_peak_specs,
    _log_peak_background_ratios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test peak/background significance in frequency bands around "
            "integer multiples of a natural-spectrum fundamental."
        )
    )
    parser.add_argument("spectrum_run_id")
    parser.add_argument(
        "--fundamental-frequency",
        type=float,
        help="Base frequency in Hz. Defaults to rank-1 detected spectrum peak.",
    )
    parser.add_argument("--max-harmonic", type=int, default=8)
    parser.add_argument("--band-half-width", type=float, default=0.04)
    parser.add_argument("--edge-guard-bins", type=int, default=3)
    parser.add_argument("--exclude-bins", type=int, default=3)
    parser.add_argument("--background-bins", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args()


def _fundamental_frequency(result: dict, override: float | None) -> float:
    if override is not None:
        if override <= 0.0:
            raise ValueError("--fundamental-frequency must be positive")
        return float(override)
    peaks = np.asarray(result["peaks"], dtype=float)
    rows = peaks[peaks[:, 0].astype(int) == 1]
    if len(rows) != 1:
        raise ValueError("spectrum run has no unique rank-1 peak")
    return float(rows[0, 2])


def _make_cfg(base_cfg: dict, args: argparse.Namespace, f0: float) -> dict:
    cfg = dict(base_cfg)
    if args.alpha is not None:
        cfg["significance_alpha"] = float(args.alpha)
    bands = []
    for harmonic in range(1, int(args.max_harmonic) + 1):
        center = harmonic * f0
        bands.append({
            "label": f"k{harmonic}",
            "min": center - float(args.band_half_width),
            "max": center + float(args.band_half_width),
        })
    cfg["peak_significance"] = {
        "enabled": True,
        "peak_power": "nearest_bin",
        "peak_window_bins": 0,
        "exclude_bins": int(args.exclude_bins),
        "background_bins": int(args.background_bins),
        "include_detected_peaks": False,
        "include_target_omegas": False,
        "include_target_frequencies": True,
        "target_frequencies": [],
        "target_frequency_bands": bands,
        "fundamental_omegas": [],
        "fundamental_frequencies": [],
        "harmonic_count": 0,
        "strict_split": False,
        "discovery_seed_fraction": 0.5,
    }
    return cfg


def _rows_with_harmonics(
    rows: list[dict],
    f0: float,
    frequency_resolution: float,
    edge_guard_bins: int,
) -> list[dict]:
    enriched = []
    for row in rows:
        copied = dict(row)
        label = str(copied["peak_rank"])
        if label.startswith("k"):
            harmonic = int(label[1:])
        else:
            harmonic = int(round(float(copied["target_frequency"]) / f0))
        copied["harmonic_index"] = harmonic
        copied["harmonic_center_frequency"] = harmonic * f0
        band_min = float(copied["target_band_min"])
        band_max = float(copied["target_band_max"])
        tested = float(copied["tested_frequency"])
        copied["peak_band_position"] = (tested - band_min) / (band_max - band_min)
        edge_distance = min(tested - band_min, band_max - tested)
        copied["peak_at_band_edge"] = bool(
            edge_distance <= int(edge_guard_bins) * float(frequency_resolution)
        )
        enriched.append(copied)
    return enriched


def _write_table(run_dir: Path, rows: list[dict]) -> None:
    header = [
        "coordinate", "harmonic_index", "harmonic_center_frequency",
        "selection_mode", "peak_source", "peak_rank", "target_frequency",
        "target_band_min", "target_band_max", "tested_frequency",
        "peak_band_position", "peak_at_band_edge", "nearest_bin", "tested_bin",
        "peak_power_method", "exclude_bins", "background_bins_per_side",
        "background_left_min",
        "background_left_max", "background_right_min",
        "background_right_max", "n_seed", "mean_log_peak_ratio",
        "std_log_peak_ratio", "standard_error", "t_statistic",
        "degrees_of_freedom", "p_value_one_sided", "p_value_adjusted",
        "confidence_interval_low", "confidence_interval_high", "significant",
        "significance_alpha", "confidence_level",
    ]
    storage.write_csv(
        run_dir / "tables" / "spectrum_harmonic_peak_significance.csv",
        header,
        [[row[key] for key in header] for row in rows],
    )


def _write_figure(run_dir: Path, result: dict, rows: list[dict], f0: float) -> None:
    import matplotlib.pyplot as plt

    labels = ("x", "y", "z")
    freqs = result["freqs"]
    psd_mean = result["psd_mean"]
    max_frequency = max(row["target_band_max"] for row in rows) + 0.25
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for coordinate_index, coordinate in enumerate(labels):
        coord_rows = sorted(
            [row for row in rows if row["coordinate"] == coordinate],
            key=lambda row: row["harmonic_index"],
        )
        psd_axis = axes[coordinate_index, 0]
        ratio_axis = axes[coordinate_index, 1]
        use = (freqs > 0.0) & (freqs <= max_frequency)
        psd_axis.loglog(freqs[use], psd_mean[coordinate_index, use],
                        color="0.15", linewidth=0.9)
        for row in coord_rows:
            psd_axis.axvspan(row["target_band_min"], row["target_band_max"],
                             color="tab:blue", alpha=0.08)
            color = "tab:green" if row["significant"] else "tab:red"
            psd_axis.axvline(row["tested_frequency"], color=color,
                             linewidth=0.9, alpha=0.8)
        psd_axis.set_ylabel(f"{coordinate} PSD")
        psd_axis.grid(True, which="both", alpha=0.2)

        x = np.asarray([row["harmonic_index"] for row in coord_rows], dtype=float)
        means = np.asarray([row["mean_log_peak_ratio"] for row in coord_rows])
        low = np.asarray([row["confidence_interval_low"] for row in coord_rows])
        high = np.asarray([row["confidence_interval_high"] for row in coord_rows])
        ratio_axis.errorbar(
            x,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="none",
            ecolor="0.35",
            capsize=3,
            linewidth=0.9,
        )
        for harmonic, mean, row in zip(x, means, coord_rows):
            color = "tab:green" if row["significant"] else "tab:red"
            marker = "^" if row["peak_at_band_edge"] else "o"
            ratio_axis.scatter([harmonic], [mean], c=color, marker=marker,
                               s=34, zorder=3)
        ratio_axis.axhline(0.0, color="black", linewidth=0.8)
        ratio_axis.set_xticks(x, [f"k={int(value)}" for value in x])
        ratio_axis.set_ylabel("mean log peak/background")
        ratio_axis.grid(True, axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("frequency")
    axes[-1, 1].set_xlabel("harmonic")
    fig.suptitle(
        "Natural-spectrum harmonic-band significance\n"
        f"f0={f0:.6g}; green: Holm-significant; triangle: band-edge maximum"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "spectrum_harmonic_peak_significance.pdf")
    fig.savefig(run_dir / "figures" / "spectrum_harmonic_peak_significance.png",
                dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir, _, result = storage.require_run(args.spectrum_run_id, "spectrum")
    for key in ("freqs", "psd_seed", "psd_mean", "combined_psd", "peaks"):
        if key not in result:
            raise ValueError(f"spectrum run lacks {key!r}")
    if int(args.max_harmonic) < 1:
        raise ValueError("--max-harmonic must be at least one")
    f0 = _fundamental_frequency(result, args.fundamental_frequency)
    cfg = _make_cfg(storage.read_json(run_dir / "config.json"), args, f0)
    candidates = _candidate_peak_specs(
        result["freqs"],
        np.empty((0, 8), dtype=float),
        cfg,
        cfg["peak_significance"],
        reference_psd=result["combined_psd"],
    )
    rows, values = _log_peak_background_ratios(
        result["psd_seed"],
        result["freqs"],
        candidates,
        cfg,
        selection_mode="harmonic_band_postprocess",
    )
    frequency_resolution = float(result["freqs"][1] - result["freqs"][0])
    rows = _rows_with_harmonics(
        rows,
        f0,
        frequency_resolution,
        int(args.edge_guard_bins),
    )
    result["harmonic_peak_significance_rows"] = rows
    result["harmonic_peak_significance_values"] = values
    storage.save_result(run_dir, result)
    _write_table(run_dir, rows)
    _write_figure(run_dir, result, rows, f0)
    print(run_dir / "tables" / "spectrum_harmonic_peak_significance.csv")
    print(run_dir / "figures" / "spectrum_harmonic_peak_significance.pdf")
    print(run_dir / "figures" / "spectrum_harmonic_peak_significance.png")


if __name__ == "__main__":
    main()
