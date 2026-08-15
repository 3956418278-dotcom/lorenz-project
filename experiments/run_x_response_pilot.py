#!/usr/bin/env python3
"""Run the x-direction dense frequency response pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.x_response_pilot import run_x_response_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/exploratory/x_response_dense_v1.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest, runtime = run_x_response_pilot(arguments.config)
    print(
        json.dumps(
            {
                "classification": manifest["classification"],
                "output_dir": str(output_dir),
                "runtime_seconds": runtime,
                "config_identifier": manifest["provenance"]["config_identifier"],
                "family_coordinates": manifest["coverage"]["critical_value"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
