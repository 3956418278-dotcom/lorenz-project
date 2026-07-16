"""Repository-local cache, run directories, status files, and packages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_ROOT = PROJECT_ROOT / "cache"
SPINUP_CACHE_ROOT = CACHE_ROOT / "spinup"
PHASE_CACHE_ROOT = CACHE_ROOT / "phase_samples"

RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
PACKAGES_ROOT = RESULTS_ROOT / "packages"


def ensure_roots() -> None:
    for path in [SPINUP_CACHE_ROOT, PHASE_CACHE_ROOT, RUNS_ROOT, PACKAGES_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fp:
        json.dump(data, fp, indent=2, sort_keys=True, default=json_default)
    tmp.replace(path)


def read_json(path):
    with Path(path).open("r") as fp:
        return json.load(fp)


def cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    return hashlib.sha256(blob).hexdigest()


def save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fp:
        np.savez_compressed(fp, **arrays)
    tmp.replace(path)


def load_npz(path):
    return np.load(path, allow_pickle=True)


def create_runtime() -> dict:
    return {"stage_times": {}, "cache": {"hits": 0, "misses": 0}, "frequency_details": {}}


def create_run(mode, cfg, resume=None):
    ensure_roots()
    if resume:
        run_id = resume
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{mode}_{cfg['config_hash']}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def init_run_files(run_dir, run_id, mode, cfg, argv):
    start = datetime.now().isoformat()
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "status.json", {
        "status": "running",
        "run_id": run_id,
        "start_time": start,
        "mode": mode,
        "completed_stages": [],
        "completed_frequencies": [],
    })
    (run_dir / "run.log").touch()
    runtime = create_runtime()
    runtime["command"] = " ".join(argv)
    runtime["start_time"] = start
    return runtime


def update_status(run_dir, **updates):
    status = read_json(run_dir / "status.json")
    status.update(updates)
    write_json(run_dir / "status.json", status)


def mark_stage(run_dir, runtime, stage, elapsed):
    status = read_json(run_dir / "status.json")
    if stage not in status["completed_stages"]:
        status["completed_stages"].append(stage)
    write_json(run_dir / "status.json", status)
    runtime["stage_times"][stage] = elapsed


@contextmanager
def stage_timer(run_dir, runtime, stage):
    t0 = time.time()
    try:
        yield
    finally:
        mark_stage(run_dir, runtime, stage, time.time() - t0)


def fail_run(run_dir, exc, stage=None):
    log = run_dir / "run.log"
    with log.open("a") as fp:
        traceback.print_exc(file=fp)
    update_status(run_dir, status="failed", failed_stage=stage,
                  error_type=type(exc).__name__, error_message=str(exc),
                  traceback_file="run.log", end_time=datetime.now().isoformat())


def complete_run(run_dir, start_time):
    elapsed = time.time() - start_time
    status = read_json(run_dir / "status.json")
    status.update(status="completed", end_time=datetime.now().isoformat(),
                  elapsed_seconds=elapsed)
    write_json(run_dir / "status.json", status)


def package_run(run_id):
    ensure_roots()
    run_dir = RUNS_ROOT / run_id
    pkg = PACKAGES_ROOT / f"{run_id}.tar.gz"
    with tarfile.open(pkg, "w:gz") as tar:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(Path(run_id) / path.relative_to(run_dir)))
    print("RESULT PACKAGE:")
    print(pkg.relative_to(PROJECT_ROOT))
    return pkg


def spinup_cache_path(key):
    ensure_roots()
    return SPINUP_CACHE_ROOT / f"{key}.npz"


def phase_cache_path(key):
    ensure_roots()
    return PHASE_CACHE_ROOT / f"{key}.npz"


def write_result(run_dir, spectrum, steady, responses, boundary, runtime):
    save_npz_atomic(
        run_dir / "result.npz",
        spectrum=np.array(spectrum, dtype=object),
        steady=np.array(steady, dtype=object),
        responses=np.array(responses, dtype=object),
        boundary=np.array(boundary, dtype=object),
        runtime=np.array(runtime, dtype=object),
    )


def write_summary(run_dir, spectrum=None, steady=None, responses=None, boundary=None):
    with (run_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["section", "omega", "name", "value_1", "value_2", "value_3", "value_4"])

        if spectrum is not None:
            for row in spectrum.get("peaks", []):
                writer.writerow(["natural_peak", row[2], "rank_f_scan_min_scan_max",
                                 row[0], row[1], row[3], row[4]])

        if steady is not None:
            for omega, detail in steady.items():
                writer.writerow(["steady", omega, "converged_recommended_n_skip",
                                 detail["converged"], detail["recommended_n_skip"], "", ""])
                for n_skip, ok, max_z in detail["rows"]:
                    writer.writerow(["steady_n_skip", omega, "n_skip_ok_max_z",
                                     n_skip, ok, max_z, ""])

        for r in responses or []:
            omega = float(r["omega"])
            writer.writerow(["frequency", omega, "n_skip_n_cycle_average_time",
                             int(r["n_skip"]), int(r["n_cycle"]),
                             float(r["n_cycle"]) * 2.0 * np.pi / omega, ""])
            writer.writerow(["response_norm", omega, "L_k1_H_k0_H_k2",
                             np.linalg.norm(r["L_amp"][..., 1]),
                             np.linalg.norm(r["H_amp"][..., 0]),
                             np.linalg.norm(r["H_amp"][..., 2]), ""])

        for row in boundary or []:
            writer.writerow(["boundary", row[0], "dx_dy_dz_threshold_Astar_ratio",
                             row[1], row[2], row[3], row[5]])
