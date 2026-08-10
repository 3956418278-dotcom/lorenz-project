"""Core implementation for the amplitude-response-regime experiment.

This experiment is intentionally independent from ``experiments/harmonic_response``:
it imports shared simulation/statistics helpers but writes only under
``outputs/amplitude_response_regime/<run_name>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.harmonic_response.common import (
    COORDINATES,
    ExperimentConfig,
    fit_seed_coefficients,
    hotelling_p_value,
    load_or_compute_response,
    write_csv,
    write_json,
)

from .config import DEFAULT_A_LIST, DEFAULT_FREQUENCIES, RegimeConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "amplitude_response_regime"
TERMS = ("c1_f", "c2_2f", "c3_f", "c3_3f", "c4_2f", "c4_4f")
RATIO_THRESHOLDS = (0.20, 0.10, 0.05)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / "matplotlib"))


def parse_args(default_run_name: str, default_n_traj: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--n-traj", type=int, default=default_n_traj)
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--frequencies", type=float, nargs="*", default=list(DEFAULT_FREQUENCIES))
    parser.add_argument("--a-list", type=float, nargs="*", default=list(DEFAULT_A_LIST))
    parser.add_argument("--n-record-samples", type=int, default=16384)
    parser.add_argument("--sample-rate", type=float, default=32.0)
    parser.add_argument("--seed-offset", type=int, default=2_000_000)
    return parser.parse_args()


def make_run_dir(run_name: str) -> Path:
    clean = run_name.strip().replace("/", "_")
    if not clean:
        raise ValueError("--run-name cannot be empty")
    run_dir = OUTPUT_ROOT / clean
    for subdir in ("cache", "config", "data", "tables", "figures", "status"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def make_configs(args: argparse.Namespace) -> tuple[RegimeConfig, ExperimentConfig]:
    frequencies = tuple(float(value) for value in args.frequencies)
    if 2.265625 not in frequencies:
        raise ValueError("frequency list must include 2.265625")
    a_list = tuple(float(value) for value in args.a_list)
    if not a_list or any(value <= 0.0 for value in a_list):
        raise ValueError("--a-list must contain positive amplitudes")
    if list(a_list) != sorted(a_list):
        raise ValueError("--a-list must be sorted ascending")
    n_record = int(args.n_record_samples)
    if n_record <= 0 or n_record & (n_record - 1):
        raise ValueError("--n-record-samples must be a positive power of two")
    df = float(args.sample_rate) / n_record
    for frequency in frequencies:
        aligned = round(frequency / df) * df
        if not math.isclose(float(frequency), aligned, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"frequency {frequency} is not FFT-bin aligned for df={df}; "
                f"nearest bin is {aligned}"
            )
        if 4.0 * float(frequency) >= 0.98 * float(args.sample_rate) / 2.0:
            raise ValueError(f"4f is too close to Nyquist for frequency {frequency}")
    regime_cfg = RegimeConfig(
        frequencies=frequencies,
        a_list=a_list,
        sample_rate=float(args.sample_rate),
        n_record_samples=n_record,
        seed_offset=int(args.seed_offset),
    )
    lorenz_cfg = ExperimentConfig(
        sample_rate=regime_cfg.sample_rate,
        n_record_samples=regime_cfg.n_record_samples,
        fft_length=regime_cfg.n_record_samples,
        welch_segment_length=max(256, regime_cfg.n_record_samples // 2),
        welch_overlap_samples=max(0, regime_cfg.n_record_samples // 4),
        alpha=regime_cfg.alpha,
        a_list=regime_cfg.a_list,
    )
    return regime_cfg, lorenz_cfg


def coefficient_rows(
    frequency: float,
    coeffs: dict[str, np.ndarray],
    n_seed: int,
) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = []
    stats_by_coord: dict[tuple[str, str], dict] = {}
    for coord_index, coordinate in enumerate(COORDINATES):
        row: dict[str, object] = {
            "frequency": float(frequency),
            "coordinate": coordinate,
            "n_seed": int(n_seed),
        }
        for term in TERMS:
            samples = coeffs[term][:, coord_index]
            _, p_value, _ = hotelling_p_value(samples)
            mean = complex(np.mean(samples))
            term_stats = {
                "mean": mean,
                "abs": float(abs(mean)),
                "p": float(p_value),
            }
            stats_by_coord[(coordinate, term)] = term_stats
            row[f"{term}_mean_real"] = float(mean.real)
            row[f"{term}_mean_imag"] = float(mean.imag)
            row[f"{term}_abs"] = float(abs(mean))
            row[f"{term}_p"] = float(p_value)
        rows.append(row)
    return rows, stats_by_coord


def amplitude_rows(
    frequency: float,
    a_list: tuple[float, ...],
    coeff_stats: dict[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for coordinate in COORDINATES:
        c1 = coeff_stats[(coordinate, "c1_f")]
        c2 = coeff_stats[(coordinate, "c2_2f")]
        c3_f = coeff_stats[(coordinate, "c3_f")]
        c3_3f = coeff_stats[(coordinate, "c3_3f")]
        c4_2f = coeff_stats[(coordinate, "c4_2f")]
        c4_4f = coeff_stats[(coordinate, "c4_4f")]
        for amplitude in a_list:
            a = float(amplitude)
            r1 = c1["abs"] * a
            r2 = c2["abs"] * a ** 2
            r3 = (c3_f["abs"] + c3_3f["abs"]) * a ** 3
            r4 = (c4_2f["abs"] + c4_4f["abs"]) * a ** 4
            r_low = r1 + r2
            r_high = r3 + r4
            rows.append({
                "frequency": float(frequency),
                "coordinate": coordinate,
                "A": a,
                "R1": float(r1),
                "R2": float(r2),
                "R3": float(r3),
                "R4": float(r4),
                "R_low": float(r_low),
                "R_high": float(r_high),
                "higher_ratio": float(r_high / max(r_low, 1e-300)),
                "second_to_first": float(r2 / max(r1, 1e-300)),
                "third_to_second": float(r3 / max(r2, 1e-300)),
                "fourth_to_second": float(r4 / max(r2, 1e-300)),
                "c1_f_p": c1["p"],
                "c2_2f_p": c2["p"],
                "c3_f_p": c3_f["p"],
                "c3_3f_p": c3_3f["p"],
                "c4_2f_p": c4_2f["p"],
                "c4_4f_p": c4_4f["p"],
            })
    return rows


def regime_summary_rows(
    coefficient_table: list[dict],
    amplitude_table: list[dict],
    alpha: float,
) -> list[dict]:
    by_key: dict[tuple[float, str], list[dict]] = {}
    for row in amplitude_table:
        by_key.setdefault((float(row["frequency"]), str(row["coordinate"])), []).append(row)
    summary = []
    for coeff in coefficient_table:
        key = (float(coeff["frequency"]), str(coeff["coordinate"]))
        amp_rows = by_key[key]
        row = {
            "frequency": key[0],
            "coordinate": key[1],
            "c1_significant": bool(float(coeff["c1_f_p"]) < alpha),
            "c2_significant": bool(float(coeff["c2_2f_p"]) < alpha),
            "c3_any_significant": bool(
                float(coeff["c3_f_p"]) < alpha or float(coeff["c3_3f_p"]) < alpha
            ),
            "c4_any_significant": bool(
                float(coeff["c4_2f_p"]) < alpha or float(coeff["c4_4f_p"]) < alpha
            ),
        }
        for threshold in RATIO_THRESHOLDS:
            valid = [
                float(item["A"])
                for item in amp_rows
                if float(item["higher_ratio"]) < threshold
            ]
            label = f"A_where_higher_ratio_below_{threshold:.2f}"
            row[label] = max(valid) if valid else ""
        summary.append(row)
    return summary


def write_tables(run_dir: Path, coefficient_table: list[dict],
                 amplitude_table: list[dict], summary_table: list[dict]) -> None:
    coefficient_header = ["frequency", "coordinate", "n_seed"]
    for term in TERMS:
        coefficient_header.extend([
            f"{term}_mean_real",
            f"{term}_mean_imag",
            f"{term}_abs",
            f"{term}_p",
        ])
    amplitude_header = [
        "frequency", "coordinate", "A", "R1", "R2", "R3", "R4", "R_low", "R_high",
        "higher_ratio", "second_to_first", "third_to_second", "fourth_to_second",
        "c1_f_p", "c2_2f_p", "c3_f_p", "c3_3f_p", "c4_2f_p", "c4_4f_p",
    ]
    summary_header = [
        "frequency", "coordinate", "c1_significant", "c2_significant",
        "c3_any_significant", "c4_any_significant",
        "A_where_higher_ratio_below_0.20",
        "A_where_higher_ratio_below_0.10",
        "A_where_higher_ratio_below_0.05",
    ]
    write_csv(
        run_dir / "tables" / "coefficient_statistics.csv",
        coefficient_header,
        [[row.get(col, "") for col in coefficient_header] for row in coefficient_table],
    )
    write_csv(
        run_dir / "tables" / "amplitude_response.csv",
        amplitude_header,
        [[row.get(col, "") for col in amplitude_header] for row in amplitude_table],
    )
    write_csv(
        run_dir / "tables" / "regime_summary.csv",
        summary_header,
        [[row.get(col, "") for col in summary_header] for row in summary_table],
    )


def _rows_for_frequency(rows: list[dict], frequency: float) -> list[dict]:
    return [row for row in rows if math.isclose(float(row["frequency"]), float(frequency))]


def make_figures(run_dir: Path, cfg: RegimeConfig, responses: list[dict],
                 amplitude_table: list[dict]) -> list[str]:
    import matplotlib.pyplot as plt

    fig_paths = []
    for response in responses:
        frequency = float(response["frequency"])
        freq_rows = _rows_for_frequency(amplitude_table, frequency)
        fig_paths.append(_plot_contributions(run_dir, frequency, freq_rows))
        fig_paths.append(_plot_higher_ratio(run_dir, frequency, freq_rows))
        fig_paths.append(_plot_relative_orders(run_dir, frequency, freq_rows))
        fig_paths.append(_plot_raw_scaling(run_dir, cfg, response))
    return fig_paths


def _plot_contributions(run_dir: Path, frequency: float, rows: list[dict]) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for coord_index, coordinate in enumerate(COORDINATES):
        coord_rows = [row for row in rows if row["coordinate"] == coordinate]
        a = [row["A"] for row in coord_rows]
        axis = axes[coord_index]
        for key in ("R1", "R2", "R3", "R4"):
            axis.plot(a, [row[key] for row in coord_rows], marker="o", label=key)
        axis.set_yscale("log")
        axis.set_ylabel(coordinate)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("A")
    fig.suptitle(f"Actual order contributions vs A, f={frequency:.6f}")
    fig.tight_layout()
    path = run_dir / "figures" / f"contributions_f_{frequency:.6f}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_higher_ratio(run_dir: Path, frequency: float, rows: list[dict]) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for coord_index, coordinate in enumerate(COORDINATES):
        coord_rows = [row for row in rows if row["coordinate"] == coordinate]
        a = [row["A"] for row in coord_rows]
        axis = axes[coord_index]
        axis.plot(a, [row["higher_ratio"] for row in coord_rows], marker="o")
        for threshold in (0.05, 0.10, 0.20):
            axis.axhline(threshold, linestyle="--", linewidth=0.9, label=f"{threshold:.2f}")
        axis.set_yscale("log")
        axis.set_ylabel(coordinate)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("A")
    fig.suptitle(f"Higher-order relative contribution, f={frequency:.6f}")
    fig.tight_layout()
    path = run_dir / "figures" / f"higher_ratio_f_{frequency:.6f}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_relative_orders(run_dir: Path, frequency: float, rows: list[dict]) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for coord_index, coordinate in enumerate(COORDINATES):
        coord_rows = [row for row in rows if row["coordinate"] == coordinate]
        a = [row["A"] for row in coord_rows]
        axis = axes[coord_index]
        for key in ("second_to_first", "third_to_second", "fourth_to_second"):
            axis.plot(a, [row[key] for row in coord_rows], marker="o", label=key)
        axis.set_yscale("log")
        axis.set_ylabel(coordinate)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("A")
    fig.suptitle(f"Second visibility and higher-order leakage, f={frequency:.6f}")
    fig.tight_layout()
    path = run_dir / "figures" / f"relative_orders_f_{frequency:.6f}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_raw_scaling(run_dir: Path, cfg: RegimeConfig, response: dict) -> str:
    import matplotlib.pyplot as plt

    frequency = float(response["frequency"])
    amplitudes = np.asarray(cfg.a_list, dtype=float)
    scaled = [
        ("O_f(A) / A", response["odd"][:, :, :, 0] / amplitudes[None, :, None]),
        ("E_2f(A) / A^2", response["even"][:, :, :, 1] / amplitudes[None, :, None] ** 2),
        ("O_3f(A) / A^3", response["odd"][:, :, :, 2] / amplitudes[None, :, None] ** 3),
        ("E_4f(A) / A^4", response["even"][:, :, :, 3] / amplitudes[None, :, None] ** 4),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    for coord_index, coordinate in enumerate(COORDINATES):
        for quantity_index, (title, values) in enumerate(scaled):
            samples = values[:, :, coord_index]
            mean = np.mean(samples, axis=0)
            axis = axes[coord_index, quantity_index]
            axis.errorbar(
                amplitudes,
                mean.real,
                yerr=stats.sem(samples.real, axis=0),
                marker="o",
                capsize=3,
                label="Re",
            )
            axis.errorbar(
                amplitudes,
                mean.imag,
                yerr=stats.sem(samples.imag, axis=0),
                marker="s",
                capsize=3,
                label="Im",
            )
            if coord_index == 0:
                axis.set_title(title)
            if quantity_index == 0:
                axis.set_ylabel(coordinate)
            axis.grid(True, alpha=0.25)
            if coord_index == 0 and quantity_index == 0:
                axis.legend(fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("A")
    fig.suptitle(f"Raw coherent complex scaling checks, f={frequency:.6f}")
    fig.tight_layout()
    path = run_dir / "figures" / f"raw_scaling_f_{frequency:.6f}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def save_coefficients_npz(run_dir: Path, frequency: float, coeffs: dict[str, np.ndarray]) -> None:
    path = run_dir / "data" / f"coefficients_f_{frequency:.6f}.npz"
    np.savez_compressed(path, frequency=float(frequency), **coeffs)


def execute(args: argparse.Namespace) -> dict:
    started = time.time()
    regime_cfg, lorenz_cfg = make_configs(args)
    run_dir = make_run_dir(args.run_name)
    config_payload = {
        "mode": getattr(args, "entrypoint_mode", "custom"),
        "n_traj": int(args.n_traj),
        "workers": int(args.workers),
        "regime": asdict(regime_cfg),
        "lorenz": asdict(lorenz_cfg),
    }
    write_json(run_dir / "config" / "config.json", config_payload)
    write_json(run_dir / "config.json", config_payload)

    responses = []
    coefficient_table = []
    amplitude_table = []
    for frequency in regime_cfg.frequencies:
        response = load_or_compute_response(
            float(frequency),
            lorenz_cfg,
            int(args.n_traj),
            run_dir,
            int(args.workers),
            int(regime_cfg.seed_offset),
        )
        responses.append(response)
        coeffs = fit_seed_coefficients(response, lorenz_cfg)
        save_coefficients_npz(run_dir, float(frequency), coeffs)
        coeff_rows, coeff_stats = coefficient_rows(
            float(frequency),
            coeffs,
            int(response["seed"].size),
        )
        coefficient_table.extend(coeff_rows)
        amplitude_table.extend(amplitude_rows(float(frequency), regime_cfg.a_list, coeff_stats))

    summary_table = regime_summary_rows(coefficient_table, amplitude_table, regime_cfg.alpha)
    write_tables(run_dir, coefficient_table, amplitude_table, summary_table)
    figures = make_figures(run_dir, regime_cfg, responses, amplitude_table)
    status = {
        "run_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
        "frequencies": [float(value) for value in regime_cfg.frequencies],
        "a_list": [float(value) for value in regime_cfg.a_list],
        "n_traj": int(args.n_traj),
        "seed_offset": int(regime_cfg.seed_offset),
        "tables": [
            str(run_dir / "tables" / "coefficient_statistics.csv"),
            str(run_dir / "tables" / "amplitude_response.csv"),
            str(run_dir / "tables" / "regime_summary.csv"),
        ],
        "figures": figures,
    }
    write_json(run_dir / "status" / "status.json", status)
    write_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def main(default_run_name: str, default_n_traj: int) -> None:
    args = parse_args(default_run_name, default_n_traj)
    args.entrypoint_mode = default_run_name
    execute(args)
