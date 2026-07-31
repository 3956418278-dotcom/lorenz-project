"""Create coordinate peak and significance outputs from an existing spectrum run.

This postprocessor reads the saved frequency grid, per-seed PSD, and mean PSD
from a completed run.  It performs the fixed discovery/test split and writes a
new derived spectrum run by default.  It never runs the Lorenz integrator.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from lorenz_sine import storage  # noqa: E402
from lorenz_sine.config import config_hash, load_config  # noqa: E402
from lorenz_sine.natural_spectrum import (  # noqa: E402
    _analyze_saved_spectrum,
    _frequency_resolution,
    _legacy_z_peak_array,
    _write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process an existing spectrum run with per-seed "
            "log peak/background t tests."
        )
    )
    parser.add_argument("spectrum_run_id")
    parser.add_argument(
        "--config",
        help=(
            "Optional spectrum config used only for coordinate peak detection "
            "and significance settings. Simulation settings still come from "
            "the saved run config."
        ),
    )
    parser.add_argument(
        "--output-run-id",
        help=(
            "Optional explicit ID for the new derived run. By default a fresh "
            "timestamped spectrum run ID is generated."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Overwrite the input run instead of creating a new derived run. "
            "Use only when intentionally updating old artifacts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_place and args.output_run_id:
        raise ValueError("--output-run-id cannot be used with --in-place")

    source_run_dir, _, source_result = storage.require_run(
        args.spectrum_run_id, "spectrum")
    cfg = storage.read_json(source_run_dir / "config.json")
    if args.config:
        override = load_config(args.config, "spectrum")
        for key in (
            "f_min",
            "prominence",
            "top_n",
            "peak_min_distance_frequency",
            "peak_significance",
            "confidence_level",
            "significance_alpha",
        ):
            if key in override:
                cfg[key] = override[key]
        cfg["config_sources"] = override.get("config_sources", cfg.get(
            "config_sources", []))
        cfg["config_path"] = override.get("config_path", cfg.get("config_path"))
    cfg["config_hash"] = config_hash(cfg)

    for key in ("freqs", "psd_seed", "psd_mean"):
        if key not in source_result:
            raise ValueError(
                f"spectrum run {args.spectrum_run_id} lacks {key!r}; "
                "cannot post-process without the saved Welch PSD arrays"
            )

    analysis = _analyze_saved_spectrum(
        source_result["freqs"],
        source_result["psd_seed"],
        source_result["psd_mean"],
        cfg,
    )
    if not analysis["peak_significance_options"]["enabled"]:
        raise ValueError("peak_significance.enabled is false")
    result = dict(source_result)
    result.update(analysis)
    result["peaks"] = _legacy_z_peak_array(
        analysis["detected_peaks_by_coordinate"],
        _frequency_resolution(source_result["freqs"]),
    )
    result["legacy_peak_coordinate"] = "z"

    started = time.time()
    if args.in_place:
        output_run_id = args.spectrum_run_id
        output_run_dir = source_run_dir
        runtime = None
    else:
        output_run_id, output_run_dir, runtime, _ = storage.begin_run(
            "spectrum",
            cfg,
            sys.argv,
            parent_run_ids={"reanalyzed_spectrum": args.spectrum_run_id},
            input_paths={
                "source_result": str(source_run_dir / "data" / "result.npz"),
                "source_config": str(source_run_dir / "config.json"),
            },
            backend="postprocess",
            worker_count=1,
            resume=args.output_run_id,
        )
        storage.append_log(
            output_run_dir,
            f"postprocess_source_run={args.spectrum_run_id}",
        )

    _write_outputs(output_run_dir, result, cfg)
    storage.save_result(output_run_dir, result)
    if runtime is not None:
        storage.complete_run(output_run_dir, runtime, started)

    print("output run id:", output_run_id)
    for relative in (
        "tables/detected_peaks_by_coordinate.csv",
        "tables/predefined_target_tests.csv",
        "figures/coordinate_psd_peaks.png",
        "figures/peak_significance.png",
    ):
        print(output_run_dir / relative)
    z_frequencies = [
        row["frequency"]
        for row in analysis["detected_peaks_by_coordinate"]
        if row["coordinate"] == "z"
    ]
    print("z automatic peak frequencies:", ", ".join(
        f"{frequency:.9g}" for frequency in z_frequencies
    ))


if __name__ == "__main__":
    main()
