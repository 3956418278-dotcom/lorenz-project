"""Add peak significance outputs to an existing unforced spectrum run.

This is a post-processing helper: it reads the saved per-seed PSD from a
completed spectrum run and writes the same significance table and figure as a
fresh spectrum run.  It never runs the Lorenz integrator.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from lorenz_sine import storage  # noqa: E402
from lorenz_sine.config import load_config  # noqa: E402
from lorenz_sine.natural_spectrum import (  # noqa: E402
    _candidate_peak_specs,
    _log_peak_background_ratios,
    _peak_significance_cfg,
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
            "Optional spectrum config used only for peak_significance settings. "
            "Simulation settings still come from the saved run config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, _, result = storage.require_run(args.spectrum_run_id, "spectrum")
    cfg = storage.read_json(run_dir / "config.json")
    if args.config:
        override = load_config(args.config, "spectrum")
        for key in ("peak_significance", "confidence_level",
                    "significance_alpha", "target_omegas"):
            if key in override:
                cfg[key] = override[key]

    for key in ("freqs", "psd_seed", "peaks"):
        if key not in result:
            raise ValueError(
                f"spectrum run {args.spectrum_run_id} lacks {key!r}; "
                "cannot test peak significance without per-seed PSD data"
            )

    options = _peak_significance_cfg(cfg)
    if not options["enabled"]:
        raise ValueError("peak_significance.enabled is false")
    candidates = _candidate_peak_specs(
        result["freqs"],
        result["peaks"],
        cfg,
        options,
        reference_psd=result.get("combined_psd"),
    )
    rows, values = _log_peak_background_ratios(
        result["psd_seed"],
        result["freqs"],
        candidates,
        cfg,
        selection_mode="postprocess_existing_spectrum_run",
    )
    result["peak_significance_rows"] = rows
    result["peak_significance_values"] = values
    result["peak_significance_options"] = options
    _write_outputs(run_dir, result, cfg)
    storage.save_result(run_dir, result)
    print(run_dir / "tables" / "spectrum_peak_significance.csv")
    print(run_dir / "figures" / "spectrum_peak_significance.pdf")
    print(run_dir / "figures" / "spectrum_peak_significance.png")


if __name__ == "__main__":
    main()
