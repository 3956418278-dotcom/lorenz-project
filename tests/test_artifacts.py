import hashlib
import json
from pathlib import Path

import numpy as np

from lorenz.artifacts import (
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    json_ready,
    verify_file_identifiers,
    write_json_atomic,
    write_npz_atomic,
)


def test_json_ready_and_atomic_write_preserve_project_json_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    value = {
        "z": np.asarray([1, 2], dtype=np.int64),
        "a": (np.float64(1.5), Path("relative/path")),
    }

    assert json_ready(value) == {
        "z": [1, 2],
        "a": [1.5, "relative/path"],
    }
    write_json_atomic(path, value)

    expected = json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_npz_write_round_trips_arrays(tmp_path):
    path = tmp_path / "raw.npz"
    arrays = {
        "real": np.arange(6).reshape(2, 3),
        "complex": np.asarray([1 + 2j, 3 - 4j]),
    }

    write_npz_atomic(path, arrays)

    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)
    assert not list(tmp_path.glob(".*.tmp"))


def test_file_sha256_returns_bare_hex_digest(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"lorenz\x00response")

    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_file_identifiers_rejects_changed_artifact(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"first")
    identifiers = {"raw.bin": f"sha256:{file_sha256(path)}"}

    verify_file_identifiers(tmp_path, identifiers)
    path.write_bytes(b"changed")
    try:
        verify_file_identifiers(tmp_path, identifiers)
    except ValueError as error:
        assert "raw.bin" in str(error)
    else:
        raise AssertionError("changed artifact was not rejected")


def test_git_provenance_reports_none_when_git_commands_fail(tmp_path):
    assert git_provenance(tmp_path) == {
        "head": None,
        "worktree_dirty": None,
    }


def test_environment_provenance_has_current_experiment_fields():
    assert set(environment_provenance()) == {
        "python",
        "platform",
        "numpy",
        "scipy",
    }


def test_active_source_identifiers_scans_source_roots_and_runner(tmp_path):
    (tmp_path / "src/lorenz").mkdir(parents=True)
    (tmp_path / "src/other").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "tests").mkdir()
    files = {
        "src/lorenz/model.py": b"MODEL = 1\n",
        "src/other/adapter.py": b"ADAPTER = 2\n",
        "experiments/run.py": b"RUN = 3\n",
        "tests/test_model.py": b"ignored = True\n",
    }
    for relative, contents in files.items():
        (tmp_path / relative).write_bytes(contents)

    identifiers = active_source_identifiers(tmp_path, "experiments/run.py")

    assert list(identifiers) == [
        "experiments/run.py",
        "src/lorenz/model.py",
        "src/other/adapter.py",
    ]
    for relative in identifiers:
        assert identifiers[relative] == f"sha256:{file_sha256(tmp_path / relative)}"


def test_active_source_identifiers_includes_runner_outside_standard_roots(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    runner = tmp_path / "scripts/custom_runner.py"
    runner.write_text("RUN = True\n", encoding="utf-8")

    assert active_source_identifiers(tmp_path, runner) == {
        "scripts/custom_runner.py": f"sha256:{file_sha256(runner)}"
    }


def test_repository_fingerprint_covers_indirect_response_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    identifiers = active_source_identifiers(
        repo_root, "experiments/run_direction_reconnaissance.py"
    )

    for relative in (
        "src/lorenz/response.py",
        "src/lorenz/strength_series.py",
        "experiments/run_direction_reconnaissance.py",
    ):
        assert identifiers[relative] == f"sha256:{file_sha256(repo_root / relative)}"
