#!/usr/bin/env python3
"""Run a configured exploratory forced-transient convergence study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.forced_transient import run_forced_transient_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/exploratory/forced_transient_late_windows.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest, runtime = run_forced_transient_study(arguments.config)
    print(
        json.dumps(
            {
                "classification": manifest["classification"],
                "protocol_status": manifest["protocol_status"],
                "output_dir": str(output_dir),
                "runtime_seconds": runtime,
                "config_identifier": manifest["provenance"]["config_identifier"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
