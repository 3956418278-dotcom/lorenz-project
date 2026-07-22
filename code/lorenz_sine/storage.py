"""Repository-local caches, durable task runs, checkpoints, and packages."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tarfile
import tempfile
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
DASK_CACHE_ROOT = CACHE_ROOT / "dask-worker-space"
MATPLOTLIB_CACHE_ROOT = CACHE_ROOT / "matplotlib"
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
PACKAGES_ROOT = RESULTS_ROOT / "packages"
RUN_SUBDIRS = ("checkpoints", "data", "tables", "figures")


def ensure_roots() -> None:
    for path in [SPINUP_CACHE_ROOT, PHASE_CACHE_ROOT, DASK_CACHE_ROOT,
                 MATPLOTLIB_CACHE_ROOT, RUNS_ROOT, PACKAGES_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_SUBDIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _unique_tmp(path: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                dir=path.parent)
    os.close(fd)
    return Path(name)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(path)
    try:
        with tmp.open("w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True, default=json_default)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    with Path(path).open("r") as fp:
        return json.load(fp)


def cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=json_default).encode()
    return hashlib.sha256(blob).hexdigest()


def save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(path)
    try:
        with tmp.open("wb") as fp:
            np.savez_compressed(fp, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_npz(path):
    return np.load(path, allow_pickle=True)


def valid_npz(path, required=()) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            return all(key in data.files for key in required)
    except Exception:
        return False


def software_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "matplotlib", "dask", "distributed",
                    "dask-mpi", "mpi4py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def create_runtime(backend="serial", worker_count=1) -> dict:
    return {
        "stage_times": {},
        "cache": {"hits": 0, "misses": 0},
        "execution_backend": backend,
        "worker_count": int(worker_count),
    }


def merge_cache_stats(runtime: dict, cache_stats: dict) -> None:
    runtime["cache"]["hits"] += int(cache_stats.get("hits", 0))
    runtime["cache"]["misses"] += int(cache_stats.get("misses", 0))


def _new_run_id(task: str, config_hash: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{stamp}_{task}_{config_hash}"


def _validate_run_id(run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError(f"invalid run id {run_id!r}")
    return run_id


def run_dir_for(run_id: str) -> Path:
    return RUNS_ROOT / _validate_run_id(run_id)


def begin_run(task: str, cfg: dict, argv, parent_run_ids: dict,
              input_paths: dict, backend: str, worker_count: int,
              resume: str | None = None):
    ensure_roots()
    run_id = _validate_run_id(resume) if resume else _new_run_id(
        task, cfg["config_hash"])
    run_dir = run_dir_for(run_id)
    existing = (run_dir / "manifest.json").exists()
    if resume and existing:
        manifest = read_json(run_dir / "manifest.json")
        if manifest.get("task") != task:
            raise ValueError(
                f"resume run {run_id} is task {manifest.get('task')!r}, expected {task!r}"
            )
        if manifest.get("config_hash") != cfg["config_hash"]:
            raise ValueError(
                f"resume config hash mismatch for {run_id}: "
                f"{manifest.get('config_hash')} != {cfg['config_hash']}"
            )
        if manifest.get("parent_run_ids", {}) != parent_run_ids:
            raise ValueError(f"resume parent runs do not match manifest for {run_id}")
        ensure_run_dirs(run_dir)
        runtime_path = run_dir / "runtime.json"
        runtime = read_json(runtime_path) if runtime_path.exists() else create_runtime(
            backend, worker_count)
        runtime["execution_backend"] = backend
        runtime["worker_count"] = int(worker_count)
        previous_status = read_json(run_dir / "status.json")
        if previous_status.get("status") != "completed":
            update_status(run_dir, status="running", resumed_at=datetime.now().isoformat())
        manifest["execution_backend"] = backend
        manifest["worker_count"] = int(worker_count)
        write_json(run_dir / "manifest.json", manifest)
        write_json(runtime_path, runtime)
        return run_id, run_dir, runtime, True
    if resume and run_dir.exists():
        unexpected = [
            path for path in run_dir.iterdir()
            if path.name != "run.log"
            and not (path.name.startswith("slurm-") and path.name.endswith(".out"))
        ]
        if unexpected:
            raise ValueError(f"resume directory exists without a valid manifest: {run_dir}")
    if not resume and run_dir.exists():
        raise FileExistsError(run_dir)
    ensure_run_dirs(run_dir)
    created_at = datetime.now().isoformat()
    write_json(run_dir / "config.json", cfg)
    status = {
        "status": "running",
        "run_id": run_id,
        "task": task,
        "created_at": created_at,
        "completed_parameter_points": [],
        "failed_parameter_points": [],
    }
    write_json(run_dir / "status.json", status)
    manifest = {
        "run_id": run_id,
        "task": task,
        "config_hash": cfg["config_hash"],
        "config_sources": cfg.get("config_sources", []),
        "parent_run_ids": parent_run_ids,
        "input_paths": input_paths,
        "created_at": created_at,
        "software_versions": software_versions(),
        "execution_backend": backend,
        "worker_count": int(worker_count),
        "completed_parameter_points": [],
    }
    write_json(run_dir / "manifest.json", manifest)
    runtime = create_runtime(backend, worker_count)
    runtime.update({"command": " ".join(argv), "created_at": created_at})
    write_json(run_dir / "runtime.json", runtime)
    (run_dir / "run.log").touch()
    append_log(run_dir, f"execution_backend={backend} workers={worker_count}")
    return run_id, run_dir, runtime, False


def append_log(run_dir, message: str) -> None:
    with (Path(run_dir) / "run.log").open("a") as fp:
        fp.write(message.rstrip() + "\n")


def update_status(run_dir, **updates):
    path = Path(run_dir) / "status.json"
    status = read_json(path)
    status.update(updates)
    write_json(path, status)


def mark_parameter(run_dir, key: str, runtime=None) -> None:
    run_dir = Path(run_dir)
    status = read_json(run_dir / "status.json")
    completed = status.setdefault("completed_parameter_points", [])
    if key not in completed:
        completed.append(key)
    write_json(run_dir / "status.json", status)
    manifest = read_json(run_dir / "manifest.json")
    manifest["completed_parameter_points"] = list(completed)
    write_json(run_dir / "manifest.json", manifest)
    if runtime is not None:
        write_json(run_dir / "runtime.json", runtime)


def mark_parameter_failed(run_dir, key: str, exc: Exception) -> None:
    status = read_json(Path(run_dir) / "status.json")
    failed = status.setdefault("failed_parameter_points", [])
    failed.append({"key": key, "error_type": type(exc).__name__, "message": str(exc)})
    write_json(Path(run_dir) / "status.json", status)


def completed_parameters(run_dir) -> set[str]:
    return set(read_json(Path(run_dir) / "status.json").get(
        "completed_parameter_points", []))


def checkpoint_path(run_dir, key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
    return Path(run_dir) / "checkpoints" / f"{safe}.npz"


@contextmanager
def stage_timer(run_dir, runtime, stage):
    started = time.time()
    try:
        yield
    finally:
        runtime["stage_times"][stage] = time.time() - started
        write_json(Path(run_dir) / "runtime.json", runtime)


def save_result(run_dir, result: dict) -> None:
    payload = np.array(result, dtype=object)
    save_npz_atomic(Path(run_dir) / "data" / "result.npz", result=payload)
    save_npz_atomic(Path(run_dir) / "result.npz", result=payload)


def load_result_path(path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        if "result" not in data.files:
            raise ValueError(f"result archive lacks 'result': {path}")
        return data["result"].item()


def load_result(run_dir) -> dict:
    path = Path(run_dir) / "data" / "result.npz"
    if not path.exists():
        path = Path(run_dir) / "result.npz"
    return load_result_path(path)


def require_run(run_id: str | None, expected_task: str):
    if not run_id:
        raise ValueError(f"missing required --{expected_task}-run input")
    run_dir = run_dir_for(run_id)
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    if not manifest_path.exists() or not status_path.exists():
        raise FileNotFoundError(f"input run does not exist or is incomplete: {run_id}")
    manifest = read_json(manifest_path)
    if manifest.get("task") != expected_task:
        raise ValueError(
            f"input run {run_id} has task {manifest.get('task')!r}; "
            f"expected {expected_task!r}"
        )
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise ValueError(
            f"input run {run_id} is not completed (status={status.get('status')!r})"
        )
    return run_dir, manifest, load_result(run_dir)


def find_ancestor_run(run_id: str, task: str):
    """Follow recorded parent IDs only; never scan or guess runs."""
    visited = set()
    queue = [run_id]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        run_dir = run_dir_for(current)
        manifest = read_json(run_dir / "manifest.json")
        if manifest.get("task") == task:
            return current, run_dir, manifest, load_result(run_dir)
        queue.extend(manifest.get("parent_run_ids", {}).values())
    raise ValueError(f"run {run_id} has no recorded {task!r} ancestor")


def complete_run(run_dir, runtime, started_at: float) -> None:
    runtime["elapsed_seconds"] = time.time() - started_at
    runtime["completed_at"] = datetime.now().isoformat()
    write_json(Path(run_dir) / "runtime.json", runtime)
    update_status(run_dir, status="completed", completed_at=runtime["completed_at"],
                  elapsed_seconds=runtime["elapsed_seconds"])
    append_log(run_dir, "status=completed")


def fail_run(run_dir, exc, stage=None):
    run_dir = Path(run_dir)
    with (run_dir / "run.log").open("a") as fp:
        traceback.print_exc(file=fp)
    update_status(run_dir, status="failed", failed_stage=stage,
                  error_type=type(exc).__name__, error_message=str(exc),
                  traceback_file="run.log", failed_at=datetime.now().isoformat())


def package_run(run_id):
    ensure_roots()
    run_dir = run_dir_for(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")
    package = PACKAGES_ROOT / f"{run_id}.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(Path(run_id) / path.relative_to(run_dir)))
    print("RESULT PACKAGE:")
    print(package.relative_to(PROJECT_ROOT))
    return package


def spinup_cache_path(key):
    ensure_roots()
    return SPINUP_CACHE_ROOT / f"{key}.npz"


def phase_cache_path(key):
    ensure_roots()
    return PHASE_CACHE_ROOT / f"{key}.npz"


def write_csv(path, header, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)
