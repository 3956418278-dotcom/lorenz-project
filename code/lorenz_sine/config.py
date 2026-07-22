"""Task-specific configuration loading and numerical design validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_update(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def require_power_of_two(cfg: dict, key: str) -> None:
    if key not in cfg:
        raise ValueError(f"missing required sampling parameter {key}")
    if not is_power_of_two(cfg[key]):
        raise ValueError(f"{key} must be a positive power of two, got {cfg[key]}")


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError as exc:
        raise ValueError(f"configuration must be inside repository: {resolved}") from exc


def _load_with_bases(path: Path, stack=None):
    stack = [] if stack is None else list(stack)
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in stack + [path])
        raise ValueError(f"cyclic base_config chain: {chain}")
    with path.open("r") as fp:
        current = json.load(fp)
    base_ref = current.pop("base_config", None)
    sources = []
    merged = {}
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        merged, sources = _load_with_bases(base_path, stack + [path])
    merged = deep_update(merged, current)
    sources.append(_relative_to_root(path))
    return merged, sources


def config_hash(cfg: dict) -> str:
    excluded = {"config_hash", "config_path", "config_sources"}
    clean = {key: value for key, value in cfg.items() if key not in excluded}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _require(cfg: dict, keys) -> None:
    for key in keys:
        if key not in cfg:
            raise ValueError(f"missing required config key {key}")


def _validate_common(cfg: dict, require_dynamics=True) -> None:
    if require_dynamics:
        _require(cfg, ["lorenz", "solver", "T_spinup", "n_seed"])
        _require(cfg["lorenz"], ["sigma", "rho", "beta"])
        _require(cfg["solver"], ["method", "rtol", "atol"])
        if int(cfg["n_seed"]) < 1:
            raise ValueError("n_seed must be positive")
        if float(cfg["T_spinup"]) <= 0:
            raise ValueError("T_spinup must be positive")
    confidence = float(cfg.get("confidence_level", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")


def _validate_forced_sampling(cfg: dict) -> None:
    require_power_of_two(cfg, "n_phase")
    require_power_of_two(cfg, "n_record_cycles")
    require_power_of_two(cfg, "n_record_samples")
    expected = int(cfg["n_phase"]) * int(cfg["n_record_cycles"])
    if int(cfg["n_record_samples"]) != expected:
        raise ValueError(
            "n_record_samples must equal n_phase * n_record_cycles "
            f"({expected}), got {cfg['n_record_samples']}"
        )
    if int(cfg.get("fourier_kmax", 0)) >= int(cfg["n_phase"]) // 2:
        raise ValueError("fourier_kmax must be below the phase-grid Nyquist harmonic")


def _validate_frequencies(values, allow_empty=False) -> None:
    if not values and not allow_empty:
        raise ValueError(
            "frequencies is empty; provide explicit frequencies or an explicit spectrum run"
        )
    if any(float(value) <= 0 for value in values):
        raise ValueError("all angular frequencies must be positive")


def _validate_spectrum(cfg: dict) -> None:
    _require(cfg, ["n_record_samples", "sample_rate", "fft_length",
                   "welch_segment_length", "welch_overlap_samples",
                   "f_min", "prominence", "top_n"])
    for key in ("n_record_samples", "fft_length", "welch_segment_length"):
        require_power_of_two(cfg, key)
    if int(cfg["fft_length"]) < int(cfg["welch_segment_length"]):
        raise ValueError("fft_length must be at least welch_segment_length")
    if int(cfg["welch_segment_length"]) > int(cfg["n_record_samples"]):
        raise ValueError("welch_segment_length cannot exceed n_record_samples")
    overlap = int(cfg["welch_overlap_samples"])
    if not 0 <= overlap < int(cfg["welch_segment_length"]):
        raise ValueError("welch_overlap_samples must be in [0, welch_segment_length)")
    if float(cfg["sample_rate"]) <= 0:
        raise ValueError("sample_rate must be positive")


def _validate_sampling_check(cfg: dict) -> None:
    _require(cfg, ["frequencies", "amplitudes", "n_skip", "fourier_kmax",
                   "scan", "reference", "relative_tolerance", "check_forcing",
                   "spectral_evaluation_frequencies"])
    _validate_frequencies(cfg["frequencies"])
    if not cfg["amplitudes"] or any(float(a) <= 0 for a in cfg["amplitudes"]):
        raise ValueError("sampling-check amplitudes must be positive")
    scan = cfg["scan"]
    for key in ("rtol", "atol", "T_spinup", "n_phase", "n_cycle",
                "fft_length", "welch_segment_length"):
        if key not in scan or not scan[key]:
            raise ValueError(f"sampling-check scan.{key} must be non-empty")
    for key in ("n_phase", "n_cycle", "fft_length", "welch_segment_length"):
        bad = [value for value in scan[key] if not is_power_of_two(value)]
        if bad:
            raise ValueError(f"all scan.{key} values must be powers of two: {bad}")
    reference = cfg["reference"]
    _require(reference, ["rtol", "atol", "T_spinup", "n_phase", "n_cycle",
                         "n_record_samples", "sample_rate", "fft_length",
                         "welch_segment_length", "welch_overlap_samples"])
    for key in ("n_phase", "n_cycle", "n_record_samples", "fft_length",
                "welch_segment_length"):
        if not is_power_of_two(reference[key]):
            raise ValueError(f"sampling-check reference.{key} must be a power of two")
    if any(float(value) < 0 for value in cfg["spectral_evaluation_frequencies"]):
        raise ValueError("spectral_evaluation_frequencies must be non-negative")


def _validate_steady(cfg: dict) -> None:
    _require(cfg, ["frequencies", "candidate_n_skip", "block_cycles",
                   "n_blocks", "n_phase", "n_record_cycles",
                   "n_record_samples", "fourier_kmax", "target_harmonics",
                   "check_forcing", "convergence"])
    _validate_frequencies(cfg["frequencies"], allow_empty=True)
    _validate_forced_sampling(cfg)
    require_power_of_two(cfg, "block_cycles")
    if int(cfg["n_record_cycles"]) != int(cfg["block_cycles"]):
        raise ValueError("steady n_record_cycles must equal block_cycles")
    if int(cfg["n_blocks"]) < 2:
        raise ValueError("steady n_blocks must be at least two")
    if not cfg["candidate_n_skip"]:
        raise ValueError("candidate_n_skip must be non-empty")
    if len(cfg["check_forcing"]) != 3:
        raise ValueError("check_forcing must have three components")


def _validate_amplitude_scan(cfg: dict) -> None:
    _require(cfg, ["amplitudes", "n_phase", "n_record_cycles",
                   "n_record_samples", "fourier_kmax", "phase",
                   "forcing_directions", "mixed_pairs"])
    _validate_forced_sampling(cfg)
    amplitudes = [float(value) for value in cfg["amplitudes"]]
    if not amplitudes or any(value <= 0 for value in amplitudes):
        raise ValueError("amplitudes must be a non-empty list of positive values")
    if len(set(amplitudes)) != len(amplitudes):
        raise ValueError("amplitudes must be unique")
    if any(int(direction) not in (0, 1, 2) for direction in cfg["forcing_directions"]):
        raise ValueError("forcing_directions entries must be 0, 1, or 2")
    if set(int(value) for value in cfg["forcing_directions"]) != {0, 1, 2}:
        raise ValueError("response estimation requires forcing_directions [0, 1, 2]")
    valid_pairs = {(0, 1), (0, 2), (1, 2)}
    if any(tuple(pair) not in valid_pairs for pair in cfg["mixed_pairs"]):
        raise ValueError(f"mixed_pairs must be chosen from {sorted(valid_pairs)}")
    if set(tuple(pair) for pair in cfg["mixed_pairs"]) != valid_pairs:
        raise ValueError("response estimation requires all three mixed_pairs")


def _validate_response(cfg: dict) -> None:
    _require(cfg, ["fit_degrees", "selection_metric", "bootstrap_samples",
                   "random_seed", "amplitude_min", "amplitude_max",
                   "fourier_kmax"])
    degrees = [int(value) for value in cfg["fit_degrees"]]
    if degrees != [0, 1, 2]:
        raise ValueError("fit_degrees must explicitly compare [0, 1, 2]")
    if cfg["selection_metric"] not in ("loo_rmse", "bic"):
        raise ValueError("selection_metric must be loo_rmse or bic")
    if int(cfg["bootstrap_samples"]) < 2:
        raise ValueError("bootstrap_samples must be at least two")


def _validate_higher_order(cfg: dict) -> None:
    _require(cfg, ["norms", "harmonics", "bootstrap_samples", "random_seed"])
    required = {"phase_l2", "per_output", "per_harmonic", "overall"}
    if not required.issubset(set(cfg["norms"])):
        raise ValueError(f"higher-order norms must include {sorted(required)}")


def _validate_validation(cfg: dict) -> None:
    _require(cfg, ["signal_to_noise_min", "linear_relative_error_max",
                   "quadratic_relative_error_max", "high_to_quadratic_max",
                   "classification_order"])
    for key in ("signal_to_noise_min", "linear_relative_error_max",
                "quadratic_relative_error_max", "high_to_quadratic_max"):
        if float(cfg[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    valid = {"noise-dominated", "linear-valid", "second-order-detectable",
             "quadratic-truncation-valid", "higher-order-contaminated"}
    if set(cfg["classification_order"]) != valid:
        raise ValueError(f"classification_order must list exactly {sorted(valid)}")


VALIDATORS = {
    "spectrum": _validate_spectrum,
    "sampling-check": _validate_sampling_check,
    "steady": _validate_steady,
    "amplitude-scan": _validate_amplitude_scan,
    "response": _validate_response,
    "higher-order": _validate_higher_order,
    "validate": _validate_validation,
    "frequency-scan": lambda cfg: _require(cfg, ["harmonics", "norm"]),
    "report": lambda cfg: _require(cfg, ["title"]),
}


def load_config(path: str | Path, task: str | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg, sources = _load_with_bases(path)
    configured_task = cfg.get("task")
    expected_task = task or configured_task
    if expected_task not in VALIDATORS:
        raise ValueError(f"unknown or missing task in configuration: {expected_task!r}")
    if configured_task != expected_task:
        raise ValueError(
            f"configuration task is {configured_task!r}, expected {expected_task!r}"
        )
    _validate_common(cfg, require_dynamics=expected_task in {
        "spectrum", "sampling-check", "steady", "amplitude-scan"
    })
    VALIDATORS[expected_task](cfg)
    cfg["config_path"] = _relative_to_root(path)
    cfg["config_sources"] = sources
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def forced_sampling_metadata(omega: float, cfg: dict) -> dict:
    omega = float(omega)
    n_phase = int(cfg["n_phase"])
    n_cycles = int(cfg["n_record_cycles"])
    period = 2.0 * math.pi / omega
    sample_rate = n_phase / period
    n_samples = n_phase * n_cycles
    frequency_resolution = sample_rate / n_samples
    nyquist = sample_rate / 2.0
    fundamental = omega / (2.0 * math.pi)
    return {
        "omega": omega,
        "period": period,
        "n_phase": n_phase,
        "n_record_cycles": n_cycles,
        "n_record_samples": n_samples,
        "sample_rate": sample_rate,
        "frequency_resolution": frequency_resolution,
        "angular_frequency_resolution": 2.0 * math.pi * frequency_resolution,
        "nyquist_frequency": nyquist,
        "nyquist_angular_frequency": 2.0 * math.pi * nyquist,
        "fundamental_bin": n_cycles,
        "second_harmonic_bin": 2 * n_cycles,
        "fundamental_resolved": fundamental >= frequency_resolution,
        "second_harmonic_resolved": 2.0 * fundamental <= nyquist,
        "integer_cycle_aligned": True,
    }


def validate_target_frequencies(omegas, sample_rate: float, fft_length: int,
                                combination_omegas=None) -> list[dict]:
    resolution = float(sample_rate) / int(fft_length)
    nyquist = float(sample_rate) / 2.0
    targets = [("omega", float(value)) for value in omegas]
    targets += [("2omega", 2.0 * float(value)) for value in omegas]
    for value in combination_omegas or []:
        targets.append(("combination", float(value)))
    rows = []
    for kind, omega in targets:
        frequency = omega / (2.0 * math.pi)
        rows.append({
            "kind": kind,
            "omega": omega,
            "frequency": frequency,
            "frequency_resolution": resolution,
            "nyquist_frequency": nyquist,
            "resolved": bool(frequency >= resolution and frequency <= nyquist),
            "nearest_fft_bin": int(round(frequency / resolution)),
        })
    return rows
