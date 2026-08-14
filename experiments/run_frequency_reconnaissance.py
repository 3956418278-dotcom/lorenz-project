#!/usr/bin/env python3
"""Run the minimal fixed-grid frequency reconnaissance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.frequency_reconnaissance import run_frequency_reconnaissance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/exploratory/frequency_reconnaissance.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest, runtime = run_frequency_reconnaissance(arguments.config)
    print(
        json.dumps(
            {
                "classification": manifest["classification"],
                "output_dir": str(output_dir),
                "runtime_seconds": runtime,
                "config_identifier": manifest["provenance"]["config_identifier"],
                "bootstrap_critical_value": manifest["joint_bootstrap"]["coverage"][
                    "critical_value"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
