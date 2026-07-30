#!/usr/bin/env python3
"""Compare multiple unforced Lorenz-63 x/y/z spectrum runs.

Place this file at the repository root.

Default input:
    outputs/unforced_xyz_spectrum/

Default output:
    outputs/unforced_xyz_spectrum/comparisons/compare_<timestamp>/

The script recursively discovers both:
    spectrum_data.npz
    spectrum_data_<run_tag>.npz

It pairs each data file with:
    run_info.json
    run_info_<run_tag>.json

No smoothing and no automatic peak detection are performed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "lorenz_compare_unforced_xyz_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = Path("outputs/unforced_xyz_spectrum")
DEFAULT_ZOOM = (0.5, 4.0)

COORDINATES = ("x", "y", "z")
STD_KEYS = ("std_x", "std_y", "std_z")
MEAN_KEYS = ("mean_fft_x", "mean_fft_y", "mean_fft_z")


@dataclass(frozen=True)
class SpectrumRun:
    data_path: Path
    info_path: Path | None
    freq: np.ndarray
    std_q: np.ndarray
    mean_q: np.ndarray
    run_info: dict[str, float | int | str]
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay multiple unforced Lorenz-63 x/y/z spectrum runs without "
            "reintegrating trajectories."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory searched recursively for spectrum_data*.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact output directory. By default a new timestamped directory "
            "is created under <input-dir>/comparisons/."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_ZOOM,
        help="Zoom frequency interval in Hz. Default: 0.5 4.0.",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help=(
            "Optional substring filter applied to the spectrum data path. "
            "Example: --contains seed_0"
        ),
    )
    parser.add_argument(
        "--peak-window",
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Read the exact maximum within a manually chosen frequency window. "
            "Repeat this option for multiple visible peaks, for example: "
            "--peak-window 1.25 1.40 --peak-window 1.50 1.70."
        ),
    )
    return parser.parse_args()


def run_tag_from_data_path(data_path: Path) -> str | None:
    stem = data_path.stem
    prefix = "spectrum_data_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return None


def find_run_info_path(data_path: Path) -> Path | None:
    run_tag = run_tag_from_data_path(data_path)

    candidates: list[Path] = []
    if run_tag:
        candidates.append(data_path.parent / f"run_info_{run_tag}.json")
    candidates.append(data_path.parent / "run_info.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_n_traj_from_name(data_path: Path) -> int | None:
    match = re.search(r"(?:^|_)ntraj(\d+)(?:_|$)", data_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def load_run_info(info_path: Path | None) -> dict[str, float | int | str]:
    if info_path is None:
        return {}

    with info_path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Run-info file must contain a JSON object: {info_path}")
    return content


def validate_frequency(freq: np.ndarray, data_path: Path) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)

    if freq.ndim != 1:
        raise ValueError(f"'freq' must be one-dimensional in {data_path}")
    if len(freq) < 2:
        raise ValueError(f"'freq' must contain at least two values in {data_path}")
    if not np.all(np.isfinite(freq)):
        raise ValueError(f"'freq' contains non-finite values in {data_path}")
    if np.any(np.diff(freq) <= 0.0):
        raise ValueError(f"'freq' must be strictly increasing in {data_path}")

    return freq


def load_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected_length: int,
    data_path: Path,
    *,
    complex_allowed: bool,
) -> np.ndarray:
    if key not in archive.files:
        raise KeyError(f"Missing key {key!r} in {data_path}")

    vector = np.asarray(archive[key])
    if vector.ndim != 1 or len(vector) != expected_length:
        raise ValueError(
            f"{key!r} in {data_path} must have shape ({expected_length},), "
            f"got {vector.shape}"
        )

    if not complex_allowed:
        vector = np.asarray(vector, dtype=np.float64)

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{key!r} contains non-finite values in {data_path}")

    return vector


def compact_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def build_label(
    run_info: dict[str, float | int | str],
    data_path: Path,
) -> str:
    fields = (
        ("seed", "SEED"),
        ("N", "N_TRAJ"),
        ("time", "EFFECTIVE_TIME"),
        ("spin", "SPINUP_TIME"),
        ("fs", "FS"),
        ("sub", "RK4_SUBSTEPS"),
    )

    pieces = [
        f"{display_name}={compact_number(run_info[key])}"
        for display_name, key in fields
        if key in run_info
    ]

    if pieces:
        return ", ".join(pieces)

    run_tag = run_tag_from_data_path(data_path)
    return run_tag if run_tag else str(data_path.parent)


def load_spectrum_run(data_path: Path) -> SpectrumRun:
    info_path = find_run_info_path(data_path)
    run_info = load_run_info(info_path)

    if "N_TRAJ" not in run_info:
        parsed_n_traj = parse_n_traj_from_name(data_path)
        if parsed_n_traj is not None:
            run_info["N_TRAJ"] = parsed_n_traj

    if "N_TRAJ" not in run_info:
        raise ValueError(
            f"Cannot determine N_TRAJ for {data_path}. "
            "Keep its matching run_info JSON or include '_ntraj<number>_' in the filename."
        )

    n_traj = int(run_info["N_TRAJ"])
    if n_traj < 2:
        raise ValueError(f"N_TRAJ must be at least 2 for {data_path}")

    with np.load(data_path, allow_pickle=False) as archive:
        if "freq" not in archive.files:
            raise KeyError(f"Missing key 'freq' in {data_path}")

        freq = validate_frequency(archive["freq"], data_path)

        std_q = np.stack(
            [
                load_vector(
                    archive,
                    key,
                    len(freq),
                    data_path,
                    complex_allowed=False,
                )
                for key in STD_KEYS
            ],
            axis=0,
        )

        mean_q = np.stack(
            [
                load_vector(
                    archive,
                    key,
                    len(freq),
                    data_path,
                    complex_allowed=True,
                )
                for key in MEAN_KEYS
            ],
            axis=0,
        )

    return SpectrumRun(
        data_path=data_path,
        info_path=info_path,
        freq=freq,
        std_q=std_q,
        mean_q=mean_q,
        run_info=run_info,
        label=build_label(run_info, data_path),
    )


def discover_data_files(input_dir: Path, contains: str | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = []
    for path in input_dir.rglob("spectrum_data*.npz"):
        if "comparisons" in path.parts:
            continue
        if contains is not None and contains not in str(path):
            continue
        files.append(path)

    return sorted(files)


def run_sort_key(run: SpectrumRun) -> tuple[object, ...]:
    info = run.run_info
    return (
        info.get("EFFECTIVE_TIME", float("inf")),
        info.get("N_TRAJ", float("inf")),
        info.get("SEED", float("inf")),
        str(run.data_path),
    )


def make_output_dir(input_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        if requested.exists() and any(requested.iterdir()):
            raise FileExistsError(
                f"Refusing to write into a non-empty output directory: {requested}"
            )
        requested.mkdir(parents=True, exist_ok=True)
        return requested

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = input_dir / "comparisons" / f"compare_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def plot_overlay(
    runs: list[SpectrumRun],
    *,
    value_getter: Callable[[SpectrumRun], np.ndarray],
    ylabel: str,
    title: str,
    output_path: Path,
    zoom: tuple[float, float] | None,
) -> None:
    fig_height = 8.0 + max(0.0, 0.18 * (len(runs) - 4))
    fig, axes = plt.subplots(3, 1, figsize=(11, fig_height), sharex=True)

    for run in runs:
        values = value_getter(run)

        for coordinate_index, axis in enumerate(axes):
            mask = run.freq > 0.0
            if zoom is not None:
                low, high = zoom
                mask &= (run.freq >= low) & (run.freq <= high)

            axis.plot(
                run.freq[mask],
                values[coordinate_index, mask],
                linewidth=0.9,
                alpha=0.85,
                label=run.label,
            )

    for coordinate, axis in zip(COORDINATES, axes):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{coordinate} direction")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)

    if zoom is not None:
        axes[-1].set_xlim(*zoom)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_columns = min(3, max(1, len(handles)))
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_columns,
            fontsize=8,
            frameon=True,
        )

    bottom_margin = min(0.34, 0.09 + 0.025 * ((len(runs) - 1) // 3))
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def extract_window_peaks(
    runs: list[SpectrumRun],
    windows: list[tuple[float, float]],
) -> list[dict[str, object]]:
    """Return the exact FFT-bin maximum of std_Q inside each manual window."""
    rows: list[dict[str, object]] = []

    for run in runs:
        n_traj = int(run.run_info["N_TRAJ"])
        frequency_resolution = float(np.median(np.diff(run.freq)))

        for window_index, (low, high) in enumerate(windows, start=1):
            mask = (run.freq >= low) & (run.freq <= high)
            if not np.any(mask):
                raise ValueError(
                    f"No FFT bins from {run.data_path} fall inside "
                    f"peak window [{low}, {high}] Hz"
                )

            candidate_indices = np.flatnonzero(mask)

            for coordinate_index, coordinate in enumerate(COORDINATES):
                local_std = run.std_q[coordinate_index, candidate_indices]
                local_offset = int(np.argmax(local_std))
                peak_index = int(candidate_indices[local_offset])

                peak_frequency = float(run.freq[peak_index])
                peak_std = float(run.std_q[coordinate_index, peak_index])
                peak_std_error = peak_std / np.sqrt(n_traj)
                peak_abs_mean = float(
                    np.abs(run.mean_q[coordinate_index, peak_index])
                )

                rows.append(
                    {
                        "data_file": str(run.data_path),
                        "run_info_file": (
                            str(run.info_path) if run.info_path is not None else ""
                        ),
                        "label": run.label,
                        "seed": run.run_info.get("SEED", ""),
                        "n_traj": n_traj,
                        "fs": run.run_info.get("FS", ""),
                        "spinup_time": run.run_info.get("SPINUP_TIME", ""),
                        "effective_time": run.run_info.get("EFFECTIVE_TIME", ""),
                        "rk4_substeps": run.run_info.get("RK4_SUBSTEPS", ""),
                        "coordinate": coordinate,
                        "window_index": window_index,
                        "window_low_hz": low,
                        "window_high_hz": high,
                        "fft_bin_index": peak_index,
                        "frequency_resolution_hz": frequency_resolution,
                        "peak_frequency_hz": peak_frequency,
                        "peak_angular_frequency": 2.0 * np.pi * peak_frequency,
                        "peak_period": (
                            1.0 / peak_frequency if peak_frequency > 0.0 else np.inf
                        ),
                        "peak_std_q": peak_std,
                        "peak_std_error": peak_std_error,
                        "peak_abs_mean_q": peak_abs_mean,
                    }
                )

    return rows


def write_peak_tables(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    import csv

    if not rows:
        return

    csv_path = output_dir / "peak_values.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["window_index"]), str(row["coordinate"]))
        grouped.setdefault(key, []).append(row)

    summary_path = output_dir / "peak_values_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        for (window_index, coordinate), group in sorted(grouped.items()):
            frequencies = np.array(
                [float(row["peak_frequency_hz"]) for row in group],
                dtype=np.float64,
            )
            low = float(group[0]["window_low_hz"])
            high = float(group[0]["window_high_hz"])

            file.write(
                f"window {window_index}: [{low:.9g}, {high:.9g}] Hz, "
                f"coordinate={coordinate}\\n"
            )
            file.write(
                f"  median peak frequency = {np.median(frequencies):.12g} Hz\\n"
            )
            file.write(
                f"  mean peak frequency   = {np.mean(frequencies):.12g} Hz\\n"
            )
            file.write(
                f"  min/max frequency     = "
                f"{np.min(frequencies):.12g} / {np.max(frequencies):.12g} Hz\\n"
            )
            file.write(f"  run count             = {len(group)}\\n")

            for row in group:
                file.write(
                    f"    {row['label']}: "
                    f"{float(row['peak_frequency_hz']):.12g} Hz "
                    f"(bin {row['fft_bin_index']}, "
                    f"std_Q={float(row['peak_std_q']):.12g})\\n"
                )
            file.write("\\n")


def write_manifest(output_dir: Path, runs: list[SpectrumRun], args: argparse.Namespace) -> None:
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "zoom_hz": [float(args.zoom[0]), float(args.zoom[1])],
        "contains_filter": args.contains,
        "run_count": len(runs),
        "runs": [
            {
                "data_path": str(run.data_path.resolve()),
                "run_info_path": (
                    str(run.info_path.resolve()) if run.info_path is not None else None
                ),
                "label": run.label,
                "run_info": run.run_info,
            }
            for run in runs
        ],
        "generated_files": [
            "std_error_full.png",
            "std_error_zoom.png",
            "abs_mean_full.png",
            "abs_mean_zoom.png",
            "peak_values.csv (when --peak-window is supplied)",
            "peak_values_summary.txt (when --peak-window is supplied)",
        ],
        "notes": [
            "std_error plots use std_Q / sqrt(N_TRAJ).",
            "abs_mean plots use abs(mean_Q).",
            "No smoothing or automatic peak detection was applied.",
        ],
    }

    with (output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def main() -> None:
    args = parse_args()

    low, high = map(float, args.zoom)
    if not (0.0 < low < high):
        raise ValueError("--zoom requires 0 < LOW < HIGH")
    zoom = (low, high)

    peak_windows: list[tuple[float, float]] = []
    if args.peak_window is not None:
        for raw_low, raw_high in args.peak_window:
            window_low = float(raw_low)
            window_high = float(raw_high)
            if not (0.0 < window_low < window_high):
                raise ValueError("--peak-window requires 0 < LOW < HIGH")
            peak_windows.append((window_low, window_high))

    data_files = discover_data_files(args.input_dir, args.contains)
    if not data_files:
        raise FileNotFoundError(
            f"No spectrum_data*.npz files found under {args.input_dir}"
        )

    runs: list[SpectrumRun] = []
    failures: list[str] = []

    for data_path in data_files:
        try:
            runs.append(load_spectrum_run(data_path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failures.append(f"{data_path}: {error}")

    if failures:
        details = "\n".join(f"  - {message}" for message in failures)
        raise RuntimeError(
            "One or more spectrum files could not be loaded:\n" + details
        )

    runs.sort(key=run_sort_key)
    output_dir = make_output_dir(args.input_dir, args.output_dir)

    plot_overlay(
        runs,
        value_getter=lambda run: run.std_q
        / np.sqrt(float(run.run_info["N_TRAJ"])),
        ylabel="std_Q / sqrt(N_TRAJ)",
        title="Unforced Lorenz-63 spectrum comparison: standard error",
        output_path=output_dir / "std_error_full.png",
        zoom=None,
    )
    plot_overlay(
        runs,
        value_getter=lambda run: run.std_q
        / np.sqrt(float(run.run_info["N_TRAJ"])),
        ylabel="std_Q / sqrt(N_TRAJ)",
        title=f"Unforced Lorenz-63 spectrum comparison: {low:g}-{high:g} Hz",
        output_path=output_dir / "std_error_zoom.png",
        zoom=zoom,
    )
    plot_overlay(
        runs,
        value_getter=lambda run: np.abs(run.mean_q),
        ylabel="abs(mean_Q)",
        title="Unforced Lorenz-63 spectrum comparison: complex mean residual",
        output_path=output_dir / "abs_mean_full.png",
        zoom=None,
    )
    plot_overlay(
        runs,
        value_getter=lambda run: np.abs(run.mean_q),
        ylabel="abs(mean_Q)",
        title=(
            "Unforced Lorenz-63 complex mean residual comparison: "
            f"{low:g}-{high:g} Hz"
        ),
        output_path=output_dir / "abs_mean_zoom.png",
        zoom=zoom,
    )

    if peak_windows:
        peak_rows = extract_window_peaks(runs, peak_windows)
        write_peak_tables(output_dir, peak_rows)

    write_manifest(output_dir, runs, args)

    print(f"Loaded {len(runs)} spectrum runs")
    print(f"Wrote comparison outputs to {output_dir}")
    for run in runs:
        print(f"  - {run.label}: {run.data_path}")


if __name__ == "__main__":
    main()
