#!/usr/bin/env python3
"""Merge the dense and c2-extras x-response artifacts into one analysis.

The dense run covers 12 frequencies at strengths {0.25, 0.5, 1, 2}; the extras
run covers {0.5, 1, 4, 8} at {3, 4}.  Both use the same 64 blocks, so the merged
cells carry the full six-strength family at the four anchor frequencies and the
four-strength family elsewhere.  The merged whole-block bootstrap, detection,
and adequacy decisions are persisted as a separate derived artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.artifacts import (
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    verify_file_identifiers,
    write_json_atomic,
)
from lorenz.x_response_pilot import (
    analyze_x_response,
    load_x_response_cells,
    merge_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-artifact", type=Path, required=True)
    parser.add_argument("--extras-artifact", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/exploratory/x_response_merged_v1",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    for artifact in (arguments.dense_artifact, arguments.extras_artifact):
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_file_identifiers(artifact, manifest["files"])
    dense = load_x_response_cells(arguments.dense_artifact / "raw_x_response_summaries.npz")
    extras = load_x_response_cells(arguments.extras_artifact / "raw_x_response_summaries.npz")
    for omega in sorted(extras):
        if omega not in dense:
            raise ValueError(f"extras frequency {omega} is absent from the dense artifact")
        dense[omega] = merge_cell(dense[omega], extras[omega])
    config = json.loads(
        (arguments.dense_artifact / "config_snapshot.json").read_text(encoding="utf-8")
    )
    derived = analyze_x_response(dense, config)
    derived["merge"] = {
        "dense_artifact": str(arguments.dense_artifact),
        "extras_artifact": str(arguments.extras_artifact),
        "merged_strengths_by_frequency": {
            str(omega): [float(value) for value in dense[omega].study.strengths]
            for omega in sorted(dense)
        },
        "note": (
            "cells with six strengths carry the full validity strength family; "
            "the remaining cells carry {0.25, 0.5, 1, 2}"
        ),
    }
    output_dir = arguments.output_root / (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_merged"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "code_identifiers": active_source_identifiers(
            REPO_ROOT, "experiments/run_x_response_merge.py"
        ),
        "git": git_provenance(REPO_ROOT),
        "environment": environment_provenance(),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output_dir / "derived_diagnostics.json", derived)
    merged_spec = {
        "dense_artifact": str(arguments.dense_artifact),
        "dense_manifest_sha256": file_sha256(arguments.dense_artifact / "manifest.json"),
        "extras_artifact": str(arguments.extras_artifact),
        "extras_manifest_sha256": file_sha256(arguments.extras_artifact / "manifest.json"),
        "bootstrap": config["bootstrap"],
    }
    write_json_atomic(output_dir / "merge_spec.json", merged_spec)
    manifest = {
        "schema_version": 1,
        "classification": "x_direction_response_merged_analysis",
        "study_id": "x_direction_response_merged_v1",
        "files": {
            "derived_diagnostics.json": f"sha256:{file_sha256(output_dir / 'derived_diagnostics.json')}",
            "merge_spec.json": f"sha256:{file_sha256(output_dir / 'merge_spec.json')}",
        },
        "merge": merged_spec,
        "coverage": derived["coverage"],
        "interpretation": derived["interpretation"],
        "provenance": provenance,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "frequencies": len(dense),
                "family_coordinates": derived["family"]["scalar_coordinate_count"],
                "critical_value": derived["coverage"]["critical_value"],
                "runtime_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
