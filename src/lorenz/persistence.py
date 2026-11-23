"""Production snapshots, restart chunks, consolidated checkpoints and manifests."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .artifacts import environment_provenance, file_sha256, git_provenance, write_json_atomic, write_npz_atomic
from .sampling import frequency_key

# Source files recorded in run manifests.
PRODUCTION_SOURCES = (
    "_dop853.c", "_fast_dop853.py", "artifacts.py", "conditions.py", "core.py", "ensemble.py",
    "high_order_probe.py", "parallel.py", "persistence.py", "probe_config.py",
    "response.py", "retention.py", "sampling.py",
)


def prepare_artifact(config, config_path, repo_root, resume_dir=None):
    if resume_dir is not None:
        output = Path(resume_dir).resolve()
        if (output / "manifest.json").exists():
            raise ValueError("completed artifacts are immutable; resume an unfinished run")
        snapshot = output / "config_snapshot.json"
        if not snapshot.is_file():
            raise ValueError("resume directory has no config_snapshot.json")
        if json.loads(snapshot.read_text(encoding="utf-8")) != config:
            raise ValueError("resume config does not match the artifact snapshot")
        return output
    identifier = file_sha256(config_path)[:12]
    output = Path(repo_root) / config["output_root"] / (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{identifier}"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "config_snapshot.json", config)
    return output


def read_chunk(path, block_ids, condition_count, harmonic_count):
    """Reuse only complete chunks matching the snapshot's block and array axes."""
    expected = (len(block_ids), condition_count, 3, harmonic_count)
    with np.load(path, allow_pickle=False) as chunk:
        if (
            "condition_means" not in chunk
            or "block_ids" not in chunk
            or not np.array_equal(chunk["block_ids"], np.asarray(block_ids, dtype=np.uint32))
            or chunk["condition_means"].shape != expected
            or not np.iscomplexobj(chunk["condition_means"])
            or not np.isfinite(chunk["condition_means"]).all()
        ):
            raise ValueError(f"incomplete or incompatible probe chunk: {path}")
        return np.asarray(chunk["condition_means"])


def write_checkpoint(output, checked, blocks, means):
    """Persist block-level cycle-averaged Fourier coefficients ``Z_n``.

    ``means`` has axes ``block, condition, state, harmonic``.  Its complex
    entries are Fourier coefficients, not time-domain cos/sin amplitudes and
    not yet susceptibilities.  The scientific conversion is recorded in
    ``condition_metadata.json`` by :func:`finish_artifact`.
    """
    prefix = f"omega_{frequency_key(checked['omega'])}"
    plan = checked["condition_plan"]
    arrays = {
        f"{prefix}_block_ids": np.asarray(blocks.block_ids, dtype=np.uint32),
        f"{prefix}_condition_means": means,
        f"{prefix}_condition_vectors": plan["forcing_vectors"],
        f"{prefix}_strengths": checked["strengths"],
        f"{prefix}_harmonics": np.asarray(checked["harmonics"], dtype=int),
    }
    if plan["forcing_phases"] is not None:
        arrays[f"{prefix}_condition_phases"] = plan["forcing_phases"]
        arrays[f"{prefix}_mixed_scales"] = plan["scales"]
    if plan["condition_labels"] is not None:
        arrays[f"{prefix}_condition_labels"] = np.asarray(plan["condition_labels"])
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    write_npz_atomic(directory / f"{prefix}.npz", arrays)


def finish_artifact(output, config, config_path, checked, blocks, repo_root):
    """Persist design/provenance metadata; no fitting, inference or plotting."""
    plan = checked["condition_plan"]
    write_json_atomic(output / "condition_metadata.json", {
        "experiment_type": checked["experiment_type"],
        "fourier_convention": "Z_n=mean(x(theta)*exp(-i*n*theta))",
        "signed_component_language": (
            "c_n=Re(Z_n), s_n=-Im(Z_n), so Z_n=c_n-i*s_n; analysis labels "
            "these coordinates cos/sin"
        ),
        "time_domain_translation": (
            "for n>0: x_n(theta)=2*c_n*cos(n*theta)+2*s_n*sin(n*theta); "
            "for n=0: Z_0 is the DC value"
        ),
        "paired_contrast": "odd=(plus-minus)/2; even=(plus+minus)/2-unforced",
        "harmonic_parity": (
            "odd Fourier harmonics use the odd-in-strength contrast and even "
            "harmonics use the even-in-strength contrast; harmonic number is "
            "not perturbative response order"
        ),
        "normalization": "chi1=2i*odd_hat[1]/h; chi2=-4*even_hat[2]/h^2; Qdc=even_hat[0]/h^2",
        "susceptibility_component_translation": {
            "first_order": (
                "if odd_hat[1]/h=c1-i*s1, then Re(chi1)=2*s1 and "
                "Im(chi1)=2*c1 for the directional contraction"
            ),
            "second_harmonic": (
                "if even_hat[2]/h^2=c2-i*s2, then Re(chi2)=-4*c2 and "
                "Im(chi2)=4*s2 for the directional contraction"
            ),
        },
        "mixed_contrast": (
            "Z_cross=(Z_plus+Z_minus)/2-Z0; chi2_jk=-2*b2/(base_a_j*base_a_k); "
            "for fitted chi2 rows component=cos means Re(chi2), component=sin means -Im(chi2)"
        ),
        "observable_labels": ["x", "y", "z"],
        "condition_amplitudes": plan["forcing_vectors"],
        "condition_phases": plan["forcing_phases"],
        "condition_labels": plan["condition_labels"],
        "mixed_pairs": plan["pairs"],
        "axis_conditions": plan.get("axis_records", []),
        "omega": checked["omega"],
        "harmonics": checked["harmonics"],
        "n_cycle": checked["cycles"],
        "n_phase": config["n_phase"],
        "discard_time": config["discard_time"],
        "forcing_phase": config["protocol"].get("phase", 0.0),
        "block_count": len(blocks.block_ids),
        "condition_means_axes": ["block", "condition", "state", "harmonic"],
        "retention": config["retention"],
    })
    root = Path(repo_root)
    identifiers = {
        f"src/lorenz/{name}": f"sha256:{file_sha256(root / 'src/lorenz' / name)}"
        for name in PRODUCTION_SOURCES
    }
    files = {
        path.relative_to(output).as_posix(): f"sha256:{file_sha256(path)}"
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    write_json_atomic(output / "manifest.json", {
        "schema_version": 1,
        "classification": config["classification"],
        "study_id": config["study_id"],
        "files": files,
        "provenance": {
            "config_identifier": f"sha256:{file_sha256(config_path)}",
            "code_identifiers": identifiers,
            "git": git_provenance(root),
            "environment": environment_provenance(),
        },
    })
