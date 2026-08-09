"""Offline reanalysis for an existing harmonic-response run.

This entry point intentionally does not run Lorenz simulations.  It reloads the
existing response caches, recomputes coherent complex statistics, and refreshes
tables/figures for the discovery run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common import (
    ExperimentConfig,
    analyze_frequency,
    load_formal_natural_peaks,
    make_figures,
    select_validation_frequencies,
    selection_with_explicit_frequencies,
    write_analysis_tables,
    write_csv,
    write_json,
)


def _load_config(path: Path) -> tuple[ExperimentConfig, int, list[float]]:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    cfg_fields = {name for name in ExperimentConfig.__dataclass_fields__}
    cfg = ExperimentConfig(**{key: payload[key] for key in cfg_fields if key in payload})
    n_traj = int(payload["n_traj"])
    return cfg, n_traj, [float(value) for value in payload.get("a_list", cfg.a_list)]


def _load_natural(run_dir: Path) -> dict:
    path = run_dir / "cache" / "natural_spectrum.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing natural spectrum cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "freqs": data["freqs"],
            "psd_seed": data["psd_seed"],
            "psd_mean": data["psd_mean"],
        }


def _load_scanned_frequencies(run_dir: Path) -> list[float]:
    path = run_dir / "tables" / "scanned_frequencies.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing scanned frequency table: {path}")
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.shape == ():
        return [float(rows["frequency"])]
    return [float(value) for value in rows["frequency"]]


def _load_cached_responses(run_dir: Path) -> list[dict]:
    responses = []
    for path in sorted((run_dir / "cache").glob("response_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            responses.append({
                "frequency": float(data["frequency"]),
                "seed": data["seed"],
                "actual_seed": data["actual_seed"] if "actual_seed" in data else data["seed"],
                "bins": data["bins"],
                "zero_coeff": data["zero_coeff"],
                "odd": data["odd"],
                "even": data["even"],
                "cache_file": str(path),
            })
    responses.sort(key=lambda row: float(row["frequency"]))
    if not responses:
        raise FileNotFoundError(f"no response caches found under {run_dir / 'cache'}")
    return responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="formal_20260808_n128_full")
    parser.add_argument("--spectrum-run", default="20260731-151340-434542998_spectrum_a93aa73d")
    args = parser.parse_args()

    started = time.time()
    run_dir = Path("outputs") / "harmonic_response" / args.run_name
    (run_dir / "status").mkdir(parents=True, exist_ok=True)
    cfg, n_traj, _ = _load_config(run_dir / "config.json")
    natural = _load_natural(run_dir)
    scanned = _load_scanned_frequencies(run_dir)
    selection = selection_with_explicit_frequencies(natural, cfg, scanned)
    responses = _load_cached_responses(run_dir)
    expected = {float(f) for f in scanned}
    got = {float(response["frequency"]) for response in responses}
    missing = sorted(expected - got)
    if missing:
        raise RuntimeError(f"missing response caches for frequencies: {missing}")

    formal_peaks = load_formal_natural_peaks(args.spectrum_run, cfg.exclusion_half_width)
    rows = []
    for response in responses:
        rows.extend(analyze_frequency(response, cfg, formal_peaks))
    write_analysis_tables(run_dir, cfg, natural, selection, rows)
    if formal_peaks:
        write_csv(
            run_dir / "tables" / "formal_natural_peaks_used.csv",
            [
                "coordinate", "frequency", "omega", "peak_rank", "n_seed",
                "p_value_adjusted", "exclusion_half_width",
            ],
            [
                [
                    row["coordinate"], row["frequency"], row["omega"], row["peak_rank"],
                    row["n_seed"], row["p_value_adjusted"], row["exclusion_half_width"],
                ]
                for row in formal_peaks
            ],
        )
    figures = make_figures(run_dir, cfg, natural, selection, responses, rows)
    status = {
        "run_dir": str(run_dir),
        "mode": "offline_reanalysis",
        "elapsed_seconds": time.time() - started,
        "n_traj": n_traj,
        "scanned_frequencies": scanned,
        "standalone_natural_spectrum_seed_note": (
            f"Standalone harmonic natural spectrum has n_spectrum_seed={cfg.n_spectrum_seed}; "
            "formal natural-peak exclusion uses the repository spectrum run when available."
        ),
        "formal_natural_spectrum_run": args.spectrum_run if formal_peaks else None,
        "formal_natural_peak_count": len(formal_peaks),
        "recommended_validation_frequencies": select_validation_frequencies(rows, max_count=4),
        "figures": figures,
        "config": asdict(cfg),
    }
    write_json(run_dir / "status_reanalysis.json", status)
    write_json(run_dir / "status" / "status_reanalysis.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
