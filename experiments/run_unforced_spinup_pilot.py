#!/usr/bin/env python3
"""Run the configured exploratory unforced-spinup convergence study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.spinup_diagnostics import run_spinup_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/exploratory/unforced_spinup_convergence.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest, runtime = run_spinup_study(arguments.config)
    print(
        json.dumps(
            {
                "classification": manifest["classification"],
                "output_dir": str(output_dir),
                "runtime_seconds": runtime,
                "config_identifier": manifest["provenance"]["config_identifier"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
