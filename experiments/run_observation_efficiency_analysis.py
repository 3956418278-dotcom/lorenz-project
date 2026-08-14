#!/usr/bin/env python3
"""Reanalyze retained B=256 cycle summaries without new trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.observation_efficiency import run_observation_efficiency_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/exploratory/within_block_observation_efficiency.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest, runtime = run_observation_efficiency_study(
        arguments.config
    )
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
