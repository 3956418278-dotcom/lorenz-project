#!/usr/bin/env python3
"""Run or dry-run the configured high-order response probe."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.high_order_probe import main


if __name__ == "__main__":
    main()
