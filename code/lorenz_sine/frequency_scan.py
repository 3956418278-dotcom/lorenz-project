"""Frequency scan orchestration and checkpoint handling."""

from __future__ import annotations

import numpy as np

from . import response, storage


def checkpoint_path(run_dir, omega):
    return run_dir / "checkpoints" / f"omega_{omega:.8g}.npz"


def completed_from_status(run_dir):
    status = storage.read_json(run_dir / "status.json")
    return set(float(x) for x in status.get("completed_frequencies", []))


def mark_frequency(run_dir, omega, n_skip, n_cycle, runtime):
    status = storage.read_json(run_dir / "status.json")
    vals = [float(x) for x in status.get("completed_frequencies", [])]
    if float(omega) not in vals:
        vals.append(float(omega))
    status["completed_frequencies"] = vals
    storage.write_json(run_dir / "status.json", status)
    runtime["frequency_details"][str(omega)] = {
        "n_skip": n_skip, "n_cycle": n_cycle,
        "average_time": n_cycle * 2 * np.pi / omega,
    }


def run(omegas, steady, cfg, run_dir, runtime, resume=False):
    results = []
    done = completed_from_status(run_dir) if resume else set()
    for omega in omegas:
        ckpt = checkpoint_path(run_dir, omega)
        if omega in done and ckpt.exists():
            data = dict(np.load(ckpt, allow_pickle=True))
            results.append(data)
            continue
        s = steady.get(str(omega), {})
        n_skip = s.get("recommended_n_skip")
        if n_skip is None:
            n_skip = int(cfg["steady"]["n_skips"][-1])
        res = response.compute_response_for_frequency(omega, int(n_skip), cfg,
                                                      run_dir, runtime)
        results.append(res)
        mark_frequency(run_dir, omega, int(n_skip), int(res["n_cycle"]), runtime)
    return results
