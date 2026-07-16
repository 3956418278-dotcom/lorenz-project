"""Configuration loading, validation, and derived parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "solver": {"method": "DOP853", "rtol": 1e-7, "atol": 1e-10},
    "T_spinup": 100.0,
    "n_phase": 64,
    "n_seed": None,
    "n_cycle_min": 100,
    "target_average_time": 1000.0,
    "amplitudes": [0.1, 0.07, 0.05, 0.035],
    "frequencies": [],
    "phase": 0.0,
    "fourier_kmax": 6,
    "bootstrap_samples": 500,
    "random_seed": 12345,
    "steady": {
        "n_skips": [2, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80],
        "check_amplitude": 0.05,
        "block_cycles": 8,
        "n_blocks": 4,
        "z_threshold": 2.0,
        "rtol": 0.05,
        "atol": 0.0,
        "min_consecutive": 3
    },
    "spectrum": {
        "T": 1000.0,
        "density": 20,
        "segment_time": 200.0,
        "f_min": 0.01,
        "prominence": 0.02,
        "top_n": 8
    },
    "boundary": {
        "ratio_thresholds": [0.05, 0.1, 0.2],
        "directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "test_amplitudes": [0.12]
    }
}


def deep_update(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, smoke_test: bool = False) -> dict:
    path = Path(path)
    with path.open("r") as fp:
        user = json.load(fp)
    cfg = deep_update(DEFAULT_CONFIG, user)
    cfg["config_path"] = str(path)
    validate_config(cfg)
    if smoke_test:
        cfg["smoke_test"] = True
    return derive_config(cfg)


def validate_config(cfg: dict) -> None:
    required = ["T_spinup", "n_phase", "n_cycle_min", "target_average_time", "amplitudes"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"missing config key {key}")
    if not cfg["amplitudes"]:
        raise ValueError("amplitudes must be non-empty")
    if cfg["n_phase"] < 4:
        raise ValueError("n_phase must be at least 4")
    if cfg["n_seed"] is not None and cfg["n_seed"] < 1:
        raise ValueError("n_seed must be null or positive")


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if k not in ("config_hash",)}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def derive_config(cfg: dict) -> dict:
    if cfg.get("n_seed") is None:
        cfg["n_seed"] = 2 if cfg.get("smoke_test") else 46
    cfg["amplitudes"] = [float(x) for x in cfg["amplitudes"]]
    cfg["frequencies"] = [float(x) for x in cfg.get("frequencies", [])]
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def n_cycle_for_frequency(omega: float, cfg: dict) -> int:
    period = 2.0 * 3.141592653589793 / omega
    return max(int(cfg["n_cycle_min"]), int(__import__("math").ceil(cfg["target_average_time"] / period)))


def solver_kwargs(cfg: dict) -> dict:
    return dict(cfg["solver"])
