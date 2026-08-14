"""Shared mechanical helpers for experiment artifacts and provenance.

Scientific array meanings, manifest schemas, and experiment interpretation stay
with the experiment modules that own them.  This module only owns serialization,
atomic replacement, hashing, and reproducibility metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import scipy


def json_ready(value: Any) -> Any:
    """Convert NumPy and path values into the project's JSON representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    """Write canonical project JSON and atomically replace ``path``."""
    path = Path(path)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(
    path: str | os.PathLike[str], arrays: Mapping[str, Any]
) -> None:
    """Write a compressed NumPy archive and atomically replace ``path``."""
    path = Path(path)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a file."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_identifiers(
    root: str | os.PathLike[str], identifiers: Mapping[str, str]
) -> None:
    """Verify a manifest-style mapping of relative paths to SHA-256 IDs."""
    root = Path(root)
    for relative, expected in identifiers.items():
        actual = f"sha256:{file_sha256(root / relative)}"
        if actual != expected:
            raise ValueError(f"artifact hash mismatch for {relative}")


def _git_output(repo_root: Path, *arguments: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False, ""
    return result.returncode == 0, result.stdout.strip()


def git_provenance(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return Git commit and dirty-state metadata, or ``None`` on failure."""
    repo_root = Path(repo_root)
    head_ok, head = _git_output(repo_root, "rev-parse", "HEAD")
    status_ok, status = _git_output(repo_root, "status", "--short")
    return {
        "head": head if head_ok else None,
        "worktree_dirty": bool(status) if status_ok else None,
    }


def environment_provenance() -> dict[str, str]:
    """Return the stable runtime fields recorded by current experiments."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _repo_relative(path: Path, repo_root: Path) -> tuple[str, Path]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"source path is outside repository: {path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative.as_posix(), resolved


def active_source_identifiers(
    repo_root: str | os.PathLike[str],
    runner_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Fingerprint project Python sources and an optional current runner.

    The deliberately conservative scan covers every ``*.py`` file below
    ``src`` and ``experiments``.  It may invalidate an artifact for an unrelated
    source edit, but it cannot omit an indirect in-tree Python dependency merely
    because an experiment forgot to maintain a dependency list.
    """
    repo_root = Path(repo_root).resolve()
    sources: dict[str, Path] = {}
    for directory in (repo_root / "src", repo_root / "experiments"):
        if not directory.exists():
            continue
        for source in directory.rglob("*.py"):
            relative, resolved = _repo_relative(source, repo_root)
            sources[relative] = resolved

    if runner_path is not None:
        runner = Path(runner_path)
        if not runner.is_absolute():
            runner = repo_root / runner
        relative, resolved = _repo_relative(runner, repo_root)
        sources[relative] = resolved

    return {
        relative: f"sha256:{file_sha256(sources[relative])}"
        for relative in sorted(sources)
    }
