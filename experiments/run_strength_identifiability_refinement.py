#!/usr/bin/env python3
"""Run the confirmed whole-block bootstrap strength refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.strength_identifiability import run_strength_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/exploratory/strength_identifiability_refinement.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest, runtime = run_strength_study(arguments.config)
    print(
        json.dumps(
            {
                "classification": manifest["classification"],
                "protocol_status": manifest["protocol_status"],
                "output_dir": str(output_dir),
                "runtime_seconds": runtime,
                "config_identifier": manifest["provenance"]["config_identifier"],
                "bootstrap_critical_value": manifest[
                    "confirmed_block_bootstrap"
                ]["coverage"]["critical_value"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
