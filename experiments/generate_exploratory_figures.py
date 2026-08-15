#!/usr/bin/env python3
"""Generate scientific figures from existing exploratory artifacts only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.exploratory_figures import generate_exploratory_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/figures/exploratory_existing.json",
    )
    arguments = parser.parse_args()
    output_dir, manifest = generate_exploratory_figures(arguments.config)
    print(
        json.dumps(
            {
                "figure_set_id": manifest["figure_set_id"],
                "output_dir": str(output_dir),
                "figures": [item["figure_id"] for item in manifest["figures"]],
                "unsupported": manifest["unsupported"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
