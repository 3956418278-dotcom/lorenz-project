#!/usr/bin/env python3
"""Production driver for the approved multi-direction response experiment.

Runs the approved production configuration direction by direction, each as
an independent single-direction artifact (the existing runner, estimator,
and persistence paths are reused unchanged; the unforced condition is
recomputed deterministically per direction).  Between directions only
TECHNICAL integrity checks run (manifest hashes, cell loads, shapes,
finiteness, block identity); no scientific parameter is ever retuned from
partial results.

Resume: a direction whose artifact carries a manifest.json is complete and
skipped; a direction with a partially written artifact directory resumes at
the last completed frequency via the runner's chunk/checkpoint logic.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lorenz.artifacts import (  # noqa: E402
    file_sha256,
    verify_file_identifiers,
    write_json_atomic,
)
from lorenz.x_response_pilot import (  # noqa: E402
    load_x_response_cells,
    run_x_response_pilot,
)

COMPLETION_FILE = "driver_completed.json"


def generate_direction_config(config: dict, direction_index: int, output_root: Path) -> Path:
    direction_config = copy.deepcopy(config)
    directions = direction_config["protocol"].pop("directions")
    direction = directions[direction_index]
    direction_config["protocol"]["direction"] = direction
    direction_config["protocol"]["status"] = (
        f"{config['protocol']['status']} direction {direction_index:02d}"
    )
    direction_config["output_root"] = str(
        Path(config["output_root"]) / f"direction_{direction_index:02d}"
    )
    direction_config["study_id"] = (
        f"{config['study_id']}_direction_{direction_index:02d}"
    )
    generated = output_root / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"direction_{direction_index:02d}.json"
    write_json_atomic(path, direction_config)
    return path


def latest_partial_dir(direction_root: Path):
    """Latest artifact dir of one direction without a manifest.json."""
    if not direction_root.is_dir():
        return None
    candidates = sorted(
        path for path in direction_root.iterdir()
        if path.is_dir() and (path / "config_snapshot.json").is_file()
    )
    for path in reversed(candidates):
        if not (path / "manifest.json").is_file():
            return path
    return None


def verify_direction_artifact(artifact_dir: Path, config: dict) -> None:
    """Technical integrity only: hashes, cell loads, shapes, finiteness."""
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    verify_file_identifiers(artifact_dir, manifest["files"])
    cells = load_x_response_cells(
        artifact_dir / "raw_x_response_summaries.npz", load_block_level=False
    )
    expected_frequencies = [float(value) for value in config["frequencies"]]
    if sorted(cells) != sorted(expected_frequencies):
        raise ValueError(
            f"direction artifact frequencies {sorted(cells)} differ from "
            f"the configuration {expected_frequencies}"
        )
    for omega, cell in cells.items():
        if len(cell.study.block_ids) != int(config["block_count"]):
            raise ValueError(
                f"omega={omega}: block count {len(cell.study.block_ids)} "
                f"differs from {config['block_count']}"
            )
        if not np.array_equal(
            cell.study.strengths,
            np.asarray(config["strengths"], dtype=float),
        ):
            raise ValueError(f"omega={omega}: strength grid mismatch")
        if not np.array_equal(
            cell.study.harmonics,
            np.asarray(config["harmonics"], dtype=int),
        ):
            raise ValueError(f"omega={omega}: harmonic set mismatch")
    print(
        f"[{time.strftime('%H:%M:%S')}] direction artifact integrity "
        f"verified: {len(cells)} cells, B={int(config['block_count'])}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/production/production_response_v1.json",
    )
    parser.add_argument("--start-direction", type=int, default=0)
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    directions = config["protocol"]["directions"]
    if len(directions) != 6:
        raise ValueError("the production configuration must carry 6 directions")
    if not 0 <= arguments.start_direction < len(directions):
        raise ValueError("start-direction out of range")
    output_root = REPO_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "production_config": str(arguments.config.resolve()),
        "directions": [direction for direction in directions],
        "artifacts": {},
    }
    summary_path = output_root / COMPLETION_FILE
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for direction_index in range(arguments.start_direction, len(directions)):
        key = f"direction_{direction_index:02d}"
        if key in summary["artifacts"]:
            print(
                f"[{time.strftime('%H:%M:%S')}] {key} already complete; "
                "skipping",
                flush=True,
            )
            continue
        direction_root = output_root / key
        partial = latest_partial_dir(direction_root)
        config_path = generate_direction_config(
            config, direction_index, output_root
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] starting {key}: "
            f"direction={directions[direction_index]}"
            + (f" (resuming {partial})" if partial else ""),
            flush=True,
        )
        artifact_dir, _manifest, runtime = run_x_response_pilot(
            config_path, resume_dir=partial
        )
        verify_direction_artifact(artifact_dir, config)
        summary["artifacts"][key] = {
            "artifact_dir": str(artifact_dir),
            "manifest_sha256": f"sha256:{file_sha256(artifact_dir / 'manifest.json')}",
            "runtime_seconds": runtime,
            "direction": directions[direction_index],
        }
        write_json_atomic(summary_path, summary)
        print(
            f"[{time.strftime('%H:%M:%S')}] {key} complete in "
            f"{runtime / 3600:.1f}h; artifact {artifact_dir}",
            flush=True,
        )
    print(
        f"[{time.strftime('%H:%M:%S')}] production run finished: "
        f"{len(summary['artifacts'])}/{len(directions)} directions complete",
        flush=True,
    )


if __name__ == "__main__":
    main()
