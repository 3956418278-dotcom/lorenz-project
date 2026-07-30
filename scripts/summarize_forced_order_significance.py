"""Summarize signed Fourier t tests from an amplitude-scan run."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from lorenz_sine import storage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact second/higher-order significance outputs.")
    parser.add_argument("amplitude_run_id")
    parser.add_argument("--display-output", type=int, default=2)
    parser.add_argument("--display-direction", type=int, default=2)
    return parser.parse_args()


def _order_class(row: dict) -> str | None:
    group = row["response_group"]
    harmonic = int(row["harmonic"])
    if group == "even" and harmonic in (0, 2):
        return "second_order"
    if group == "odd" and harmonic >= 3:
        return "higher_odd"
    if group == "even" and harmonic >= 4:
        return "higher_even"
    return None


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order, start=1):
        value = min(1.0, (m - rank + 1) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _summarize(rows: list[dict]) -> list[dict]:
    family_members = defaultdict(list)
    for row in rows:
        order = _order_class(row)
        if order is None:
            continue
        family_key = (
            int(row["output"]),
            int(row["forcing_direction"]),
            order,
        )
        family_members[family_key].append(row)
    family_adjusted = {}
    for family_key, values in family_members.items():
        p_values = [float(row["two_sided_p_value"]) for row in values]
        adjusted = _holm_adjust(p_values)
        for row, p_value in zip(values, adjusted):
            component_key = (
                float(row["omega"]),
                float(row["amplitude"]),
                int(row["output"]),
                int(row["forcing_direction"]),
                row["response_group"],
                int(row["harmonic"]),
                row["component"],
            )
            family_adjusted[component_key] = p_value

    grouped = defaultdict(list)
    for row in rows:
        order = _order_class(row)
        if order is None:
            continue
        key = (
            float(row["omega"]),
            float(row["amplitude"]),
            int(row["output"]),
            int(row["forcing_direction"]),
            order,
        )
        grouped[key].append(row)

    summary = []
    for key, values in grouped.items():
        omega, amplitude, output, direction, order = key
        p_values = [float(row["two_sided_p_value"]) for row in values]
        adjusted = _holm_adjust(p_values)
        significant = [
            row for row in values
            if family_adjusted[(
                float(row["omega"]),
                float(row["amplitude"]),
                int(row["output"]),
                int(row["forcing_direction"]),
                row["response_group"],
                int(row["harmonic"]),
                row["component"],
            )] < float(row["significance_alpha"])
        ]
        family_p_values = [
            family_adjusted[(
                float(row["omega"]),
                float(row["amplitude"]),
                int(row["output"]),
                int(row["forcing_direction"]),
                row["response_group"],
                int(row["harmonic"]),
                row["component"],
            )]
            for row in values
        ]
        best_index = min(range(len(values)), key=lambda index: family_p_values[index])
        best = values[best_index]
        summary.append({
            "omega": omega,
            "forcing_frequency": omega / (2.0 * math.pi),
            "amplitude": amplitude,
            "output": output,
            "forcing_direction": direction,
            "order_class": order,
            "n_components_tested": len(values),
            "n_significant_components": len(significant),
            "min_p_value": p_values[best_index],
            "min_p_value_adjusted": adjusted[best_index],
            "min_p_value_family_adjusted": family_p_values[best_index],
            "best_response_group": best["response_group"],
            "best_harmonic": int(best["harmonic"]),
            "best_component": best["component"],
            "best_sample_mean": float(best["sample_mean"]),
            "best_ci_low": float(best["ci_low"]),
            "best_ci_high": float(best["ci_high"]),
            "any_significant": bool(significant),
            "significance_alpha": float(best["significance_alpha"]),
        })
    return sorted(
        summary,
        key=lambda row: (
            row["output"], row["forcing_direction"], row["order_class"],
            row["omega"], row["amplitude"],
        ),
    )


def _write_summary(run_dir: Path, rows: list[dict]) -> None:
    header = [
        "omega", "forcing_frequency", "amplitude", "output",
        "forcing_direction", "order_class", "n_components_tested",
        "n_significant_components", "min_p_value", "min_p_value_adjusted",
        "min_p_value_family_adjusted", "best_response_group", "best_harmonic",
        "best_component", "best_sample_mean", "best_ci_low", "best_ci_high",
        "any_significant", "significance_alpha",
    ]
    storage.write_csv(
        run_dir / "tables" / "forced_order_significance_summary.csv",
        header,
        [[row[key] for key in header] for row in rows],
    )


def _write_display_figure(
    run_dir: Path,
    rows: list[dict],
    display_output: int,
    display_direction: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    classes = ["second_order", "higher_odd", "higher_even"]
    subset = [
        row for row in rows
        if row["output"] == display_output
        and row["forcing_direction"] == display_direction
    ]
    frequencies = sorted({row["forcing_frequency"] for row in subset})
    amplitudes = sorted({row["amplitude"] for row in subset}, reverse=True)
    fig, axes = plt.subplots(1, len(classes), figsize=(13, 4), sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for axis, order in zip(axes, classes):
        values = np.full((len(amplitudes), len(frequencies)), np.nan)
        labels = [["" for _ in frequencies] for _ in amplitudes]
        for row in subset:
            if row["order_class"] != order:
                continue
            i = amplitudes.index(row["amplitude"])
            j = frequencies.index(row["forcing_frequency"])
            p_value = max(float(row["min_p_value_family_adjusted"]), 1e-300)
            values[i, j] = min(-math.log10(p_value), 12.0)
            if row["any_significant"]:
                labels[i][j] = "*"
        image = axis.imshow(values, aspect="auto", origin="upper",
                            vmin=0.0, vmax=6.0, cmap="viridis")
        axis.set_title(order.replace("_", " "))
        axis.set_xticks(range(len(frequencies)),
                        [f"{value:.3g}" for value in frequencies],
                        rotation=45, ha="right")
        axis.set_yticks(range(len(amplitudes)),
                        [f"{value:.3g}" for value in amplitudes])
        axis.set_xlabel("forcing frequency")
        for i in range(len(amplitudes)):
            for j in range(len(frequencies)):
                if labels[i][j]:
                    axis.text(j, i, labels[i][j], color="white",
                              ha="center", va="center", fontsize=12,
                              fontweight="bold")
    axes[0].set_ylabel("forcing amplitude")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("-log10(min p), capped at 12")
    fig.suptitle(
        f"Forced response order significance, output={display_output}, "
        f"forcing direction={display_direction}\n"
        "asterisk: Holm-significant within output/direction/order family"
    )
    fig.savefig(run_dir / "figures" / "forced_order_significance_summary.png",
                dpi=180, bbox_inches="tight")
    fig.savefig(run_dir / "figures" / "forced_order_significance_summary.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir, _, _ = storage.require_run(args.amplitude_run_id, "amplitude-scan")
    source = run_dir / "tables" / "signed_fft_t_tests.csv"
    rows = _read_rows(source)
    summary = _summarize(rows)
    _write_summary(run_dir, summary)
    _write_display_figure(
        run_dir, summary, int(args.display_output), int(args.display_direction))
    print(run_dir / "tables" / "forced_order_significance_summary.csv")
    print(run_dir / "figures" / "forced_order_significance_summary.png")
    print(run_dir / "figures" / "forced_order_significance_summary.pdf")


if __name__ == "__main__":
    main()
