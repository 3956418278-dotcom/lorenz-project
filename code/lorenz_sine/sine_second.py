"""Full sine-forcing amplitude/frequency scan with seed-level tensors.

This module deliberately reuses the existing Lorenz RHS, spinup, phase
sampling, and Fourier projection helpers.  It adds a standalone reporting layer
for the ``sine_second/`` experiment requested in the project notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, special

from . import core
from .fourier import fourier_coefficients


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sine_second"
COORDINATES = ("x", "y", "z")
DEFAULT_AMPLITUDES = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
DEFAULT_OMEGAS = tuple(float(x) for x in np.geomspace(0.2, 5.0, 7))
COMPLEX_COMPONENTS = ("cos", "sin")
HARMONICS = (1, 2, 3, 4)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))


@dataclass(frozen=True)
class SineSecondConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    method: str = "DOP853"
    rtol: float = 1e-8
    atol: float = 1e-11
    t_spinup: float = 128.0
    forced_spinup_time: float = 128.0
    n_phase: int = 64
    n_record_cycles: int = 128
    ngrp: int = 64
    seed_offset: int = 3_000_000
    phase: float = 0.0
    confidence_level: float = 0.95
    n_bootstrap: int = 2000
    bootstrap_seed: int = 92821
    significance_alpha: float = 0.05
    noise_half_width_bins: int = 20
    exclude_half_width_bins: int = 1
    primary_forcing_direction: int = 2
    candidate_snr_threshold: float = 3.0
    weak_snr_threshold: float = 1.5
    amplitudes: tuple[float, ...] = DEFAULT_AMPLITUDES
    omegas: tuple[float, ...] = DEFAULT_OMEGAS

    def lorenz_cfg(self) -> dict:
        return {
            "lorenz": {"sigma": self.sigma, "rho": self.rho, "beta": self.beta},
            "solver": {"method": self.method, "rtol": self.rtol, "atol": self.atol},
            "T_spinup": self.t_spinup,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Lorenz sine_second harmonic/tensor scan."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--ngrp", type=int, default=SineSecondConfig.ngrp)
    parser.add_argument("--amplitudes", type=float, nargs="*", default=list(DEFAULT_AMPLITUDES))
    parser.add_argument("--omegas", type=float, nargs="*", default=list(DEFAULT_OMEGAS))
    parser.add_argument(
        "--omega-indices",
        type=int,
        nargs="*",
        help="Optional subset of the omega grid to run; indices refer to --omegas.",
    )
    parser.add_argument("--n-phase", type=int, default=SineSecondConfig.n_phase)
    parser.add_argument("--n-record-cycles", type=int, default=SineSecondConfig.n_record_cycles)
    parser.add_argument("--n-bootstrap", type=int, default=SineSecondConfig.n_bootstrap)
    parser.add_argument("--seed-offset", type=int, default=SineSecondConfig.seed_offset)
    parser.add_argument("--forced-spinup-time", type=float, default=SineSecondConfig.forced_spinup_time)
    parser.add_argument("--noise-half-width-bins", type=int, default=SineSecondConfig.noise_half_width_bins)
    parser.add_argument("--exclude-half-width-bins", type=int, default=SineSecondConfig.exclude_half_width_bins)
    parser.add_argument("--primary-forcing-direction", type=int, default=SineSecondConfig.primary_forcing_direction)
    parser.add_argument("--rtol", type=float, default=SineSecondConfig.rtol)
    parser.add_argument("--atol", type=float, default=SineSecondConfig.atol)
    return parser.parse_args(argv)


def _validate_power_of_two(value: int, name: str) -> None:
    value = int(value)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def config_from_args(args: argparse.Namespace) -> SineSecondConfig:
    amplitudes = tuple(float(x) for x in args.amplitudes)
    omegas_all = tuple(float(x) for x in args.omegas)
    if args.omega_indices:
        bad = [idx for idx in args.omega_indices if idx < 0 or idx >= len(omegas_all)]
        if bad:
            raise ValueError(f"--omega-indices outside omega grid: {bad}")
        omegas = tuple(omegas_all[idx] for idx in args.omega_indices)
    else:
        omegas = omegas_all
    if not amplitudes or any(x <= 0 for x in amplitudes):
        raise ValueError("amplitudes must be positive and non-empty")
    if tuple(sorted(amplitudes)) != amplitudes:
        raise ValueError("amplitudes must be sorted ascending")
    if not omegas or any(x <= 0 for x in omegas):
        raise ValueError("omegas must be positive and non-empty")
    _validate_power_of_two(args.n_phase, "n_phase")
    _validate_power_of_two(args.n_record_cycles, "n_record_cycles")
    if int(args.primary_forcing_direction) not in (0, 1, 2):
        raise ValueError("--primary-forcing-direction must be 0, 1, or 2")
    return SineSecondConfig(
        rtol=float(args.rtol),
        atol=float(args.atol),
        forced_spinup_time=float(args.forced_spinup_time),
        n_phase=int(args.n_phase),
        n_record_cycles=int(args.n_record_cycles),
        ngrp=int(args.ngrp),
        seed_offset=int(args.seed_offset),
        n_bootstrap=int(args.n_bootstrap),
        noise_half_width_bins=int(args.noise_half_width_bins),
        exclude_half_width_bins=int(args.exclude_half_width_bins),
        primary_forcing_direction=int(args.primary_forcing_direction),
        amplitudes=amplitudes,
        omegas=omegas,
    )


def _combo_definitions() -> list[dict]:
    combos = [{"name": "zero", "vector_unit": np.zeros(3)}]
    for direction in range(3):
        for sign, suffix in ((1.0, "pos"), (-1.0, "neg")):
            vector = np.zeros(3)
            vector[direction] = sign
            combos.append({"name": f"axis_{direction}_{suffix}", "vector_unit": vector})
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for s1, n1 in ((1.0, "p"), (-1.0, "m")):
            for s2, n2 in ((1.0, "p"), (-1.0, "m")):
                vector = np.zeros(3)
                vector[first] = s1
                vector[second] = s2
                combos.append({"name": f"pair_{first}_{second}_{n1}{n2}", "vector_unit": vector})
    return combos


def _combo_index(combos: list[dict]) -> dict[str, int]:
    return {str(combo["name"]): idx for idx, combo in enumerate(combos)}


def _raw_fft_from_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    record = np.asarray(samples, dtype=float).reshape(3, -1)
    centered = record - record.mean(axis=-1, keepdims=True)
    coeff = 2.0 * np.fft.rfft(centered, axis=-1) / centered.shape[-1]
    return coeff, np.abs(coeff)


def _local_noise(amplitude: np.ndarray, target_bin: int, half_width: int, exclude: int) -> np.ndarray:
    n_freq = amplitude.shape[-1]
    lo = max(1, int(target_bin) - int(half_width))
    hi = min(n_freq, int(target_bin) + int(half_width) + 1)
    mask = np.ones(hi - lo, dtype=bool)
    ex_lo = max(lo, int(target_bin) - int(exclude))
    ex_hi = min(hi, int(target_bin) + int(exclude) + 1)
    mask[ex_lo - lo:ex_hi - lo] = False
    if not np.any(mask):
        return np.full(amplitude.shape[:-1], np.nan)
    return np.median(amplitude[..., lo:hi][..., mask], axis=-1)


def _forced_skip_cycles(omega: float, cfg: SineSecondConfig) -> int:
    period = 2.0 * math.pi / float(omega)
    return max(1, int(math.ceil(float(cfg.forced_spinup_time) / period)))


def _seed_worker(payload: tuple[int, int, float, SineSecondConfig]) -> dict:
    omega_index, seed_index, omega, cfg = payload
    actual_seed = int(cfg.seed_offset) + int(seed_index)
    initial_state = core.spinup_state(actual_seed, cfg.lorenz_cfg())
    combos = _combo_definitions()
    n_amp = len(cfg.amplitudes)
    n_combo = len(combos)
    n_freq = cfg.n_phase * cfg.n_record_cycles // 2 + 1
    phase_means = np.empty((n_amp, n_combo, 3, cfg.n_phase), dtype=float)
    fourier = np.empty((n_amp, n_combo, 3, 5, 2), dtype=float)
    raw_target_amp = np.empty((n_amp, 3, 3, 4), dtype=float)
    raw_noise = np.empty((n_amp, 3, 3, 4), dtype=float)
    phase_signal_amp = np.empty((n_amp, 3, 3, 4), dtype=float)
    phase_signal_coeff = np.empty((n_amp, 3, 3, 4), dtype=complex)
    raw_spectrum_primary = np.empty((n_amp, 2, 3, n_freq), dtype=float)

    n_skip = _forced_skip_cycles(float(omega), cfg)
    zero_samples = core.simulate_phase_samples(
        actual_seed,
        np.zeros(3),
        float(omega),
        float(cfg.phase),
        n_skip,
        int(cfg.n_record_cycles),
        int(cfg.n_phase),
        cfg.lorenz_cfg(),
        initial_state=initial_state,
    )
    zero_mean = zero_samples.mean(axis=1)
    zero_raw_coeff, zero_raw_amp = _raw_fft_from_samples(zero_samples)
    target_bins = np.asarray([k * cfg.n_record_cycles for k in HARMONICS], dtype=int)
    zero_noise_by_harmonic = np.stack([
        _local_noise(
            zero_raw_amp,
            int(target_bin),
            int(cfg.noise_half_width_bins),
            int(cfg.exclude_half_width_bins),
        )
        for target_bin in target_bins
    ], axis=-1)

    for amp_index, amplitude in enumerate(cfg.amplitudes):
        for combo_index, combo in enumerate(combos):
            if combo["name"] == "zero":
                samples = zero_samples
                phase_mean = zero_mean
            else:
                forcing = float(amplitude) * np.asarray(combo["vector_unit"], dtype=float)
                samples = core.simulate_phase_samples(
                    actual_seed,
                    forcing,
                    float(omega),
                    float(cfg.phase),
                    n_skip,
                    int(cfg.n_record_cycles),
                    int(cfg.n_phase),
                    cfg.lorenz_cfg(),
                    initial_state=initial_state,
                )
                phase_mean = samples.mean(axis=1)
            phase_means[amp_index, combo_index] = phase_mean
            fourier[amp_index, combo_index] = fourier_coefficients(phase_mean, 4)

            name = str(combo["name"])
            if name == "zero":
                raw_spectrum_primary[amp_index, 0] = zero_raw_amp
            elif name == f"axis_{cfg.primary_forcing_direction}_pos":
                _, forced_raw_amp = _raw_fft_from_samples(samples)
                raw_spectrum_primary[amp_index, 1] = forced_raw_amp
            if name.startswith("axis_") and name.endswith("_pos"):
                direction = int(name.split("_")[1])
                raw_coeff, raw_amp = _raw_fft_from_samples(samples)
                for h_index, target_bin in enumerate(target_bins):
                    raw_target_amp[amp_index, direction, :, h_index] = raw_amp[:, int(target_bin)]
                    raw_noise[amp_index, direction, :, h_index] = zero_noise_by_harmonic[:, h_index]
                    coeff = fourier[amp_index, combo_index, :, h_index + 1, 0] + (
                        1j * fourier[amp_index, combo_index, :, h_index + 1, 1]
                    )
                    phase_signal_coeff[amp_index, direction, :, h_index] = coeff
                    phase_signal_amp[amp_index, direction, :, h_index] = np.abs(coeff)

    return {
        "omega_index": int(omega_index),
        "seed_index": int(seed_index),
        "actual_seed": int(actual_seed),
        "n_skip": int(n_skip),
        "phase_means": phase_means,
        "fourier": fourier,
        "raw_target_amp": raw_target_amp,
        "raw_noise": raw_noise,
        "phase_signal_amp": phase_signal_amp,
        "phase_signal_coeff": phase_signal_coeff,
        "raw_spectrum_primary": raw_spectrum_primary,
    }


def _run_parallel(cfg: SineSecondConfig, workers: int) -> dict:
    combos = _combo_definitions()
    n_omega = len(cfg.omegas)
    n_amp = len(cfg.amplitudes)
    n_combo = len(combos)
    n_seed = int(cfg.ngrp)
    n_freq = cfg.n_phase * cfg.n_record_cycles // 2 + 1
    phase_means = np.empty((n_omega, n_amp, n_combo, n_seed, 3, cfg.n_phase), dtype=float)
    fourier = np.empty((n_omega, n_amp, n_combo, n_seed, 3, 5, 2), dtype=float)
    raw_target_amp = np.empty((n_omega, n_amp, 3, n_seed, 3, 4), dtype=float)
    raw_noise = np.empty_like(raw_target_amp)
    phase_signal_amp = np.empty_like(raw_target_amp)
    phase_signal_coeff = np.empty((n_omega, n_amp, 3, n_seed, 3, 4), dtype=complex)
    raw_spectrum_mean = np.zeros((n_omega, n_amp, 2, 3, n_freq), dtype=float)
    raw_spectrum_representative = np.empty((n_omega, n_amp, 2, 3, n_freq), dtype=float)
    actual_seeds = np.empty(n_seed, dtype=int)
    n_skip_by_omega_seed = np.empty((n_omega, n_seed), dtype=int)

    payloads = [
        (omega_index, seed_index, float(omega), cfg)
        for omega_index, omega in enumerate(cfg.omegas)
        for seed_index in range(n_seed)
    ]
    if workers <= 1:
        iterator = (_seed_worker(payload) for payload in payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=int(workers))
        futures = [pool.submit(_seed_worker, payload) for payload in payloads]
        iterator = (future.result() for future in as_completed(futures))
    completed = 0
    try:
        for row in iterator:
            oi = row["omega_index"]
            si = row["seed_index"]
            phase_means[oi, :, :, si] = row["phase_means"]
            fourier[oi, :, :, si] = row["fourier"]
            raw_target_amp[oi, :, :, si] = row["raw_target_amp"]
            raw_noise[oi, :, :, si] = row["raw_noise"]
            phase_signal_amp[oi, :, :, si] = row["phase_signal_amp"]
            phase_signal_coeff[oi, :, :, si] = row["phase_signal_coeff"]
            raw_spectrum_mean[oi] += row["raw_spectrum_primary"] / n_seed
            if si == 0:
                raw_spectrum_representative[oi] = row["raw_spectrum_primary"]
            actual_seeds[si] = row["actual_seed"]
            n_skip_by_omega_seed[oi, si] = row["n_skip"]
            completed += 1
            if completed % max(1, min(16, n_seed)) == 0:
                print(f"completed_seed_points={completed}/{len(payloads)}", flush=True)
    finally:
        if workers > 1:
            pool.shutdown()

    return {
        "combo_names": np.asarray([combo["name"] for combo in combos]),
        "phase_means": phase_means,
        "fourier_by_seed": fourier,
        "raw_target_amp_by_seed": raw_target_amp,
        "raw_noise_by_seed": raw_noise,
        "phase_signal_amp_by_seed": phase_signal_amp,
        "phase_signal_coeff_by_seed": phase_signal_coeff,
        "raw_spectrum_mean": raw_spectrum_mean,
        "raw_spectrum_representative": raw_spectrum_representative,
        "actual_seeds": actual_seeds,
        "n_skip_by_omega_seed": n_skip_by_omega_seed,
    }


def _bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, confidence: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    n = values.shape[0]
    if n == 1:
        mean = values[0]
        return np.asarray(mean), np.asarray(mean)
    boot = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, n, n)
        boot.append(values[indices].mean(axis=0))
    boot = np.asarray(boot)
    alpha = 100.0 * (1.0 - float(confidence)) / 2.0
    return np.percentile(boot, alpha, axis=0), np.percentile(boot, 100.0 - alpha, axis=0)


def _mean_ci_scalar(samples: np.ndarray, cfg: SineSecondConfig,
                    rng: np.random.Generator) -> tuple[float, float, float]:
    samples = np.asarray(samples, dtype=float)
    mean = float(np.nanmean(samples))
    if samples.size <= 1 or int(cfg.n_bootstrap) <= 0:
        return mean, mean, mean
    low, high = _bootstrap_mean_ci(samples, int(cfg.n_bootstrap), cfg.confidence_level, rng)
    return mean, float(low), float(high)


def _hotelling(values: np.ndarray, confidence: float) -> dict:
    values = np.asarray(values, dtype=complex)
    x = np.column_stack([values.real, values.imag])
    n, p = x.shape
    mean = x.mean(axis=0)
    if n <= p:
        return {
            "T2": float("nan"), "F": float("nan"), "p": float("nan"),
            "ellipse_center_cos": float(mean[0]), "ellipse_center_sin": float(mean[1]),
            "ellipse_axis_major": float("nan"), "ellipse_axis_minor": float("nan"),
            "ellipse_angle_rad": float("nan"),
        }
    cov = np.cov(x, rowvar=False) + np.eye(p) * 1e-18
    t2 = float(n * mean @ np.linalg.pinv(cov) @ mean)
    f_stat = float((n - p) * t2 / (p * (n - 1)))
    p_value = float(_f_sf(f_stat, p, n - p))
    c = float(p * (n - 1) / (n - p) * _f_ppf(confidence, p, n - p))
    eigvals, eigvecs = np.linalg.eigh(cov / n)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return {
        "T2": t2,
        "F": f_stat,
        "p": p_value,
        "ellipse_center_cos": float(mean[0]),
        "ellipse_center_sin": float(mean[1]),
        "ellipse_axis_major": float(math.sqrt(max(c * eigvals[0], 0.0))),
        "ellipse_axis_minor": float(math.sqrt(max(c * eigvals[1], 0.0))),
        "ellipse_angle_rad": angle,
    }


def _f_cdf(x: float, dfn: int, dfd: int) -> float:
    if not np.isfinite(x):
        return 1.0 if x > 0 else 0.0
    if x <= 0.0:
        return 0.0
    z = (float(dfn) * float(x)) / (float(dfn) * float(x) + float(dfd))
    return float(special.betainc(0.5 * float(dfn), 0.5 * float(dfd), z))


def _f_sf(x: float, dfn: int, dfd: int) -> float:
    return max(0.0, min(1.0, 1.0 - _f_cdf(x, dfn, dfd)))


def _f_ppf(q: float, dfn: int, dfd: int) -> float:
    q = float(q)
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return float("inf")
    hi = 1.0
    while _f_cdf(hi, dfn, dfd) < q:
        hi *= 2.0
        if hi > 1e12:
            break
    return float(optimize.brentq(lambda x: _f_cdf(x, dfn, dfd) - q, 0.0, hi))


def _complex_phase_ci(samples: np.ndarray, cfg: SineSecondConfig,
                      rng: np.random.Generator) -> tuple[float, float, float, float, float, float, float, float]:
    samples = np.asarray(samples, dtype=complex)
    mean = complex(np.mean(samples))
    amp = float(abs(mean))
    phase = float(math.atan2(-mean.imag, mean.real))
    real_low, real_high = _bootstrap_mean_ci(samples.real, cfg.n_bootstrap, cfg.confidence_level, rng)
    imag_low, imag_high = _bootstrap_mean_ci(samples.imag, cfg.n_bootstrap, cfg.confidence_level, rng)
    amp_samples = np.abs(samples)
    amp_mean, amp_low, amp_high = _mean_ci_scalar(amp_samples, cfg, rng)
    if samples.size <= 1:
        phase_low = phase_high = phase
    else:
        boot_phase = []
        for _ in range(int(cfg.n_bootstrap)):
            indices = rng.integers(0, samples.size, samples.size)
            z = complex(np.mean(samples[indices]))
            boot_phase.append(math.atan2(-z.imag, z.real))
        boot_phase = np.unwrap(np.asarray(boot_phase))
        center = float(np.unwrap(np.asarray([phase, np.mean(boot_phase)]))[0])
        boot_phase = boot_phase + (center - float(np.mean(boot_phase)))
        alpha = 100.0 * (1.0 - cfg.confidence_level) / 2.0
        phase_low = float(np.percentile(boot_phase, alpha))
        phase_high = float(np.percentile(boot_phase, 100.0 - alpha))
    return (
        float(mean.real), float(real_low), float(real_high),
        float(mean.imag), float(imag_low), float(imag_high),
        amp, float(amp_low), float(amp_high), phase, phase_low, phase_high,
    )


def _finite_difference_tensors(data: dict, cfg: SineSecondConfig) -> dict:
    phase = np.asarray(data["phase_means"], dtype=float)
    combos = [str(x) for x in data["combo_names"]]
    index = {name: idx for idx, name in enumerate(combos)}
    amplitudes = np.asarray(cfg.amplitudes, dtype=float)
    n_omega, n_amp, _, n_seed, _, n_phase = phase.shape
    L = np.empty((n_omega, n_amp, n_seed, 3, 3, n_phase), dtype=float)
    H = np.empty((n_omega, n_amp, n_seed, 3, 3, 3, n_phase), dtype=float)
    zero = phase[:, :, index["zero"]]
    amp_shape = (1, n_amp, 1, 1, 1)
    for direction in range(3):
        pos = phase[:, :, index[f"axis_{direction}_pos"]]
        neg = phase[:, :, index[f"axis_{direction}_neg"]]
        L[:, :, :, :, direction] = (pos - neg) / (2.0 * amplitudes.reshape(amp_shape))
        H[:, :, :, :, direction, direction] = (
            pos + neg - 2.0 * zero
        ) / (amplitudes.reshape(amp_shape) ** 2)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        pp = phase[:, :, index[f"pair_{first}_{second}_pp"]]
        pm = phase[:, :, index[f"pair_{first}_{second}_pm"]]
        mp = phase[:, :, index[f"pair_{first}_{second}_mp"]]
        mm = phase[:, :, index[f"pair_{first}_{second}_mm"]]
        mixed = (pp - pm - mp + mm) / (4.0 * amplitudes.reshape(amp_shape) ** 2)
        H[:, :, :, :, first, second] = mixed
        H[:, :, :, :, second, first] = mixed

    L_fourier = fourier_coefficients(L, 4)
    H_fourier = fourier_coefficients(H, 4)
    rng = np.random.default_rng(int(cfg.bootstrap_seed) + 11)
    L_mean = L.mean(axis=2)
    H_mean = H.mean(axis=2)
    L_low = np.empty_like(L_mean)
    L_high = np.empty_like(L_mean)
    H_low = np.empty_like(H_mean)
    H_high = np.empty_like(H_mean)
    for oi in range(n_omega):
        for ai in range(n_amp):
            L_low[oi, ai], L_high[oi, ai] = _bootstrap_mean_ci(
                L[oi, ai], cfg.n_bootstrap, cfg.confidence_level, rng
            )
            H_low[oi, ai], H_high[oi, ai] = _bootstrap_mean_ci(
                H[oi, ai], cfg.n_bootstrap, cfg.confidence_level, rng
            )
    return {
        "L_theta_by_seed": L,
        "H_theta_by_seed": H,
        "L_theta": L_mean,
        "L_theta_lower": L_low,
        "L_theta_upper": L_high,
        "H_theta": H_mean,
        "H_theta_lower": H_low,
        "H_theta_upper": H_high,
        "L_fourier_by_seed": L_fourier,
        "H_fourier_by_seed": H_fourier,
    }


def _write_csv(path: Path, rows: list[dict], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=header, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scalar_tables(data: dict, tensors: dict, cfg: SineSecondConfig) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(int(cfg.bootstrap_seed) + 101)
    rows = []
    hotelling_rows = []
    ratio_rows = []
    signal = data["phase_signal_amp_by_seed"]
    noise = data["raw_noise_by_seed"]
    coeff = data["phase_signal_coeff_by_seed"]
    raw_target = data["raw_target_amp_by_seed"]
    for oi, omega in enumerate(cfg.omegas):
        for ai, amplitude in enumerate(cfg.amplitudes):
            for direction in range(3):
                for output, output_name in enumerate(COORDINATES):
                    r21_seed = signal[oi, ai, direction, :, output, 1] / np.maximum(
                        signal[oi, ai, direction, :, output, 0], 1e-300
                    )
                    r21, r21_low, r21_high = _mean_ci_scalar(r21_seed, cfg, rng)
                    ratio_rows.append({
                        "omega": float(omega),
                        "forcing_amp": float(amplitude),
                        "forcing_direction": COORDINATES[direction],
                        "output": output_name,
                        "R21": r21,
                        "R21_ci_low": r21_low,
                        "R21_ci_high": r21_high,
                    })
                    for h_index, harmonic in enumerate(HARMONICS):
                        sig = signal[oi, ai, direction, :, output, h_index]
                        raw = raw_target[oi, ai, direction, :, output, h_index]
                        noi = noise[oi, ai, direction, :, output, h_index]
                        snr_seed = sig / np.maximum(noi, 1e-300)
                        sig_mean, sig_low, sig_high = _mean_ci_scalar(sig, cfg, rng)
                        raw_mean, raw_low, raw_high = _mean_ci_scalar(raw, cfg, rng)
                        noise_mean, noise_low, noise_high = _mean_ci_scalar(noi, cfg, rng)
                        snr_mean, snr_low, snr_high = _mean_ci_scalar(snr_seed, cfg, rng)
                        hstats = _hotelling(coeff[oi, ai, direction, :, output, h_index], cfg.confidence_level)
                        row = {
                            "omega": float(omega),
                            "forcing_amp": float(amplitude),
                            "forcing_direction": COORDINATES[direction],
                            "output": output_name,
                            "harmonic": int(harmonic),
                            "signal_amp": sig_mean,
                            "signal_amp_ci_low": sig_low,
                            "signal_amp_ci_high": sig_high,
                            "raw_forced_target_amp": raw_mean,
                            "raw_forced_target_amp_ci_low": raw_low,
                            "raw_forced_target_amp_ci_high": raw_high,
                            "noise": noise_mean,
                            "noise_ci_low": noise_low,
                            "noise_ci_high": noise_high,
                            "snr": snr_mean,
                            "snr_ci_low": snr_low,
                            "snr_ci_high": snr_high,
                            "hotelling_T2": hstats["T2"],
                            "hotelling_F": hstats["F"],
                            "hotelling_p": hstats["p"],
                        }
                        rows.append(row)
                        hotelling_rows.append({
                            **row,
                            "ellipse_center_cos": hstats["ellipse_center_cos"],
                            "ellipse_center_sin": hstats["ellipse_center_sin"],
                            "ellipse_axis_major": hstats["ellipse_axis_major"],
                            "ellipse_axis_minor": hstats["ellipse_axis_minor"],
                            "ellipse_angle_rad": hstats["ellipse_angle_rad"],
                        })
    return rows, hotelling_rows, ratio_rows


def _tensor_tables(tensors: dict, cfg: SineSecondConfig) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(int(cfg.bootstrap_seed) + 202)
    first_rows = []
    second_rows = []
    Lf = tensors["L_fourier_by_seed"]
    Hf = tensors["H_fourier_by_seed"]
    for oi, omega in enumerate(cfg.omegas):
        for ai, amplitude in enumerate(cfg.amplitudes):
            for output in range(3):
                for direction in range(3):
                    samples = Lf[oi, ai, :, output, direction, 1, 0] + (
                        1j * Lf[oi, ai, :, output, direction, 1, 1]
                    )
                    vals = _complex_phase_ci(samples, cfg, rng)
                    first_rows.append({
                        "omega": float(omega),
                        "forcing_amp": float(amplitude),
                        "output_i": COORDINATES[output],
                        "forcing_j": COORDINATES[direction],
                        "real": vals[0],
                        "real_ci_low": vals[1],
                        "real_ci_high": vals[2],
                        "imag": vals[3],
                        "imag_ci_low": vals[4],
                        "imag_ci_high": vals[5],
                        "amplitude": vals[6],
                        "amplitude_ci_low": vals[7],
                        "amplitude_ci_high": vals[8],
                        "phase": vals[9],
                        "phase_ci_low": vals[10],
                        "phase_ci_high": vals[11],
                    })
            for output in range(3):
                for j in range(3):
                    for k in range(3):
                        h0 = Hf[oi, ai, :, output, j, k, 0, 0]
                        hc = Hf[oi, ai, :, output, j, k, 2, 0]
                        hs = Hf[oi, ai, :, output, j, k, 2, 1]
                        h_complex = hc + 1j * hs
                        h0_mean, h0_low, h0_high = _mean_ci_scalar(h0, cfg, rng)
                        hc_mean, hc_low, hc_high = _mean_ci_scalar(hc, cfg, rng)
                        hs_mean, hs_low, hs_high = _mean_ci_scalar(hs, cfg, rng)
                        abs_mean, abs_low, abs_high = _mean_ci_scalar(np.abs(h_complex), cfg, rng)
                        phase = float(math.atan2(-float(np.mean(hs)), float(np.mean(hc))))
                        second_rows.append({
                            "omega": float(omega),
                            "forcing_amp": float(amplitude),
                            "output_i": COORDINATES[output],
                            "forcing_j": COORDINATES[j],
                            "forcing_k": COORDINATES[k],
                            "H_mean": h0_mean,
                            "H_mean_ci_low": h0_low,
                            "H_mean_ci_high": h0_high,
                            "H_cos2": hc_mean,
                            "H_cos2_ci_low": hc_low,
                            "H_cos2_ci_high": hc_high,
                            "H_sin2": hs_mean,
                            "H_sin2_ci_low": hs_low,
                            "H_sin2_ci_high": hs_high,
                            "H_abs_2omega": abs_mean,
                            "H_abs_2omega_ci_low": abs_low,
                            "H_abs_2omega_ci_high": abs_high,
                            "H_phase_2omega": phase,
                            "independent_symmetric_entry": bool(j <= k),
                            "nonzero_by_ci": bool((h0_low > 0 or h0_high < 0) or (hc_low > 0 or hc_high < 0) or (hs_low > 0 or hs_high < 0)),
                        })
    return first_rows, second_rows


def _write_figures(out: Path, cfg: SineSecondConfig, scalar_rows: list[dict],
                   ratio_rows: list[dict], hotelling_rows: list[dict],
                   tensor_rows: list[dict], data: dict) -> list[str]:
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    omega_values = [float(x) for x in cfg.omegas]
    primary_dir = COORDINATES[int(cfg.primary_forcing_direction)]

    freq_template = np.arange(cfg.n_phase * cfg.n_record_cycles // 2 + 1, dtype=float)
    for oi, omega in enumerate(cfg.omegas):
        for ai, amplitude in enumerate(cfg.amplitudes):
            freq_axis = freq_template * (float(omega) / (2.0 * math.pi * cfg.n_record_cycles))
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            for output, axis in enumerate(axes):
                axis.plot(freq_axis, data["raw_spectrum_representative"][oi, ai, 0, output],
                          color="0.25", linewidth=0.7, label="unforced representative")
                axis.plot(freq_axis, data["raw_spectrum_representative"][oi, ai, 1, output],
                          color="tab:blue", linewidth=0.7, alpha=0.85,
                          label=f"forced +A {primary_dir} representative")
                axis.plot(freq_axis, data["raw_spectrum_mean"][oi, ai, 0, output],
                          color="black", linewidth=1.1, alpha=0.65, label="unforced ensemble mean")
                axis.plot(freq_axis, data["raw_spectrum_mean"][oi, ai, 1, output],
                          color="tab:orange", linewidth=1.1, alpha=0.75, label="forced ensemble mean")
                for harmonic in HARMONICS:
                    f_target = harmonic * float(omega) / (2.0 * math.pi)
                    axis.axvline(f_target, color="tab:red", linewidth=0.8, alpha=0.75)
                    axis.text(f_target, axis.get_ylim()[1], f"{harmonic}ω",
                              ha="center", va="top", fontsize=7)
                axis.set_ylabel(f"{COORDINATES[output]} |FFT|")
                axis.set_yscale("log")
                axis.grid(True, alpha=0.2)
                if output == 0:
                    axis.legend(frameon=False, fontsize=7, ncol=2)
            axes[-1].set_xlabel("frequency cycles / Lorenz time")
            fig.suptitle(f"Figure 1 raw unsmoothed FFT: omega={omega:.6g}, A={amplitude:.3g}")
            fig.tight_layout()
            path = fig_dir / f"figure1_raw_fft_omega_{oi}_amp_{ai}.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths.append(str(path))

    def _plot_metric(metric: str, low: str, high: str, ylabel: str, stem: str) -> None:
        for omega in omega_values:
            fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            for output, axis in enumerate(axes):
                for harmonic in HARMONICS:
                    rows = [
                        r for r in scalar_rows
                        if math.isclose(float(r["omega"]), omega)
                        and r["forcing_direction"] == primary_dir
                        and r["output"] == COORDINATES[output]
                        and int(r["harmonic"]) == harmonic
                    ]
                    xs = np.asarray([r["forcing_amp"] for r in rows], dtype=float)
                    ys = np.asarray([r[metric] for r in rows], dtype=float)
                    lo = np.asarray([r[low] for r in rows], dtype=float)
                    hi = np.asarray([r[high] for r in rows], dtype=float)
                    axis.errorbar(xs, ys, yerr=[np.maximum(ys - lo, 0.0), np.maximum(hi - ys, 0.0)], marker="o",
                                  capsize=3, label=f"{harmonic}ω")
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set_ylabel(f"{COORDINATES[output]} {ylabel}")
                axis.grid(True, alpha=0.25)
                axis.legend(frameon=False, fontsize=8)
            axes[-1].set_xlabel("forcing amplitude A")
            fig.suptitle(f"{stem}: omega={omega:.6g}, forcing={primary_dir}")
            fig.tight_layout()
            path = fig_dir / f"{stem.lower().replace(' ', '_')}_omega_{omega:.6g}.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths.append(str(path))

    _plot_metric("signal_amp", "signal_amp_ci_low", "signal_amp_ci_high",
                 "coherent amplitude", "Figure 2 harmonic amplitude")
    _plot_metric("snr", "snr_ci_low", "snr_ci_high", "SNR", "Figure 3 harmonic SNR")

    for omega in omega_values:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for output, axis in enumerate(axes):
            rows = [
                r for r in ratio_rows
                if math.isclose(float(r["omega"]), omega)
                and r["forcing_direction"] == primary_dir
                and r["output"] == COORDINATES[output]
            ]
            xs = np.asarray([r["forcing_amp"] for r in rows], dtype=float)
            ys = np.asarray([r["R21"] for r in rows], dtype=float)
            lo = np.asarray([r["R21_ci_low"] for r in rows], dtype=float)
            hi = np.asarray([r["R21_ci_high"] for r in rows], dtype=float)
            axis.errorbar(xs, ys, yerr=[np.maximum(ys - lo, 0.0), np.maximum(hi - ys, 0.0)], marker="o", capsize=3)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_ylabel(f"{COORDINATES[output]} A2/A1")
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("forcing amplitude A")
        fig.suptitle(f"Figure 4 second/first harmonic ratio: omega={omega:.6g}, forcing={primary_dir}")
        fig.tight_layout()
        path = fig_dir / f"figure4_r21_omega_{omega:.6g}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    for omega in omega_values:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for output, axis in enumerate(axes):
            for harmonic in HARMONICS:
                rows = [
                    r for r in hotelling_rows
                    if math.isclose(float(r["omega"]), omega)
                    and r["forcing_direction"] == primary_dir
                    and r["output"] == COORDINATES[output]
                    and int(r["harmonic"]) == harmonic
                ]
                axis.plot([r["forcing_amp"] for r in rows],
                          [r["hotelling_p"] for r in rows],
                          marker="o", label=f"{harmonic}ω")
            axis.axhline(cfg.significance_alpha, color="black", linewidth=0.8, linestyle="--")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_ylabel(f"{COORDINATES[output]} p")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        axes[-1].set_xlabel("forcing amplitude A")
        fig.suptitle(f"Figure 5 Hotelling significance: omega={omega:.6g}, forcing={primary_dir}")
        fig.tight_layout()
        path = fig_dir / f"figure5_hotelling_omega_{omega:.6g}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    first_omega = float(cfg.omegas[0])
    first_amp = float(cfg.amplitudes[0])
    subset = [
        r for r in tensor_rows
        if math.isclose(float(r["omega"]), first_omega)
        and math.isclose(float(r["forcing_amp"]), first_amp)
    ]
    for field in ("H_mean", "H_cos2", "H_sin2", "H_abs_2omega"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for output, axis in enumerate(axes):
            matrix = np.zeros((3, 3))
            for row in subset:
                if row["output_i"] == COORDINATES[output]:
                    j = COORDINATES.index(row["forcing_j"])
                    k = COORDINATES.index(row["forcing_k"])
                    matrix[j, k] = float(row[field])
            im = axis.imshow(matrix, cmap="coolwarm")
            axis.set_xticks(range(3), COORDINATES)
            axis.set_yticks(range(3), COORDINATES)
            axis.set_title(f"output {COORDINATES[output]}")
            fig.colorbar(im, ax=axis, shrink=0.75)
        fig.suptitle(f"Figure 6 tensor coefficients {field}; omega={first_omega:.6g}, A={first_amp:.3g}")
        fig.tight_layout()
        path = fig_dir / f"figure6_tensor_{field}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _candidate_rows(scalar_rows: list[dict], ratio_rows: list[dict], cfg: SineSecondConfig) -> list[dict]:
    ratio_by_key = {
        (r["omega"], r["forcing_amp"], r["forcing_direction"], r["output"]): r
        for r in ratio_rows
    }
    candidates = []
    for omega in cfg.omegas:
        for amplitude in cfg.amplitudes:
            for direction in COORDINATES:
                for output in COORDINATES:
                    rows = {
                        int(r["harmonic"]): r
                        for r in scalar_rows
                        if math.isclose(float(r["omega"]), float(omega))
                        and math.isclose(float(r["forcing_amp"]), float(amplitude))
                        and r["forcing_direction"] == direction
                        and r["output"] == output
                    }
                    if len(rows) != 4:
                        continue
                    low_ok = (
                        rows[1]["snr"] >= cfg.candidate_snr_threshold
                        and rows[2]["snr"] >= cfg.candidate_snr_threshold
                        and rows[1]["hotelling_p"] < cfg.significance_alpha
                        and rows[2]["hotelling_p"] < cfg.significance_alpha
                    )
                    high_weak = (
                        rows[3]["snr"] <= cfg.weak_snr_threshold
                        and rows[4]["snr"] <= cfg.weak_snr_threshold
                        and rows[3]["hotelling_p"] >= cfg.significance_alpha
                        and rows[4]["hotelling_p"] >= cfg.significance_alpha
                    )
                    if low_ok and high_weak:
                        ratio = ratio_by_key[(float(omega), float(amplitude), direction, output)]
                        candidates.append({
                            "omega": float(omega),
                            "forcing_amp": float(amplitude),
                            "forcing_direction": direction,
                            "output": output,
                            "SNR1": rows[1]["snr"],
                            "SNR2": rows[2]["snr"],
                            "SNR3": rows[3]["snr"],
                            "SNR4": rows[4]["snr"],
                            "p1": rows[1]["hotelling_p"],
                            "p2": rows[2]["hotelling_p"],
                            "p3": rows[3]["hotelling_p"],
                            "p4": rows[4]["hotelling_p"],
                            "A2_A1": ratio["R21"],
                            "A2_A1_ci_low": ratio["R21_ci_low"],
                            "A2_A1_ci_high": ratio["R21_ci_high"],
                        })
    return candidates


def _matrix_from_first_rows(rows: list[dict], omega: float, amplitude: float) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=complex)
    for row in rows:
        if math.isclose(float(row["omega"]), omega) and math.isclose(float(row["forcing_amp"]), amplitude):
            i = COORDINATES.index(row["output_i"])
            j = COORDINATES.index(row["forcing_j"])
            matrix[i, j] = float(row["real"]) + 1j * float(row["imag"])
    return matrix


def _tensor_from_second_rows(rows: list[dict], omega: float, amplitude: float, field: str) -> np.ndarray:
    tensor = np.zeros((3, 3, 3), dtype=float)
    for row in rows:
        if math.isclose(float(row["omega"]), omega) and math.isclose(float(row["forcing_amp"]), amplitude):
            i = COORDINATES.index(row["output_i"])
            j = COORDINATES.index(row["forcing_j"])
            k = COORDINATES.index(row["forcing_k"])
            tensor[i, j, k] = float(row[field])
    return tensor


def _format_array(array: np.ndarray) -> str:
    return np.array2string(np.asarray(array), precision=6, suppress_small=False)


def _write_report(out: Path, cfg: SineSecondConfig, scalar_rows: list[dict],
                  first_rows: list[dict], second_rows: list[dict],
                  ratio_rows: list[dict], candidates: list[dict],
                  figures: list[str]) -> None:
    if candidates:
        omega = float(candidates[0]["omega"])
        amplitude = float(candidates[0]["forcing_amp"])
        direction = str(candidates[0]["forcing_direction"])
        output = str(candidates[0]["output"])
    else:
        omega = float(cfg.omegas[0])
        amplitude = float(cfg.amplitudes[0])
        direction = COORDINATES[int(cfg.primary_forcing_direction)]
        output = "x"

    L = _matrix_from_first_rows(first_rows, omega, amplitude)
    H0 = _tensor_from_second_rows(second_rows, omega, amplitude, "H_mean")
    Hc = _tensor_from_second_rows(second_rows, omega, amplitude, "H_cos2")
    Hs = _tensor_from_second_rows(second_rows, omega, amplitude, "H_sin2")
    Ha = _tensor_from_second_rows(second_rows, omega, amplitude, "H_abs_2omega")
    harmonic_lines = []
    for harmonic in HARMONICS:
        row = next(
            r for r in scalar_rows
            if math.isclose(float(r["omega"]), omega)
            and math.isclose(float(r["forcing_amp"]), amplitude)
            and r["forcing_direction"] == direction
            and r["output"] == output
            and int(r["harmonic"]) == harmonic
        )
        harmonic_lines.append(
            f"{harmonic}ω  "
            f"{row['signal_amp']:.6g} [{row['signal_amp_ci_low']:.6g}, {row['signal_amp_ci_high']:.6g}]  "
            f"{row['noise']:.6g} [{row['noise_ci_low']:.6g}, {row['noise_ci_high']:.6g}]  "
            f"{row['snr']:.6g} [{row['snr_ci_low']:.6g}, {row['snr_ci_high']:.6g}]  "
            f"p={row['hotelling_p']:.6g}"
        )
    ratio = next(
        r for r in ratio_rows
        if math.isclose(float(r["omega"]), omega)
        and math.isclose(float(r["forcing_amp"]), amplitude)
        and r["forcing_direction"] == direction
        and r["output"] == output
    )
    candidate_text = "none"
    if candidates:
        candidate_text = "\n".join(
            f"omega={r['omega']:.6g}, A={r['forcing_amp']:.6g}, "
            f"forcing={r['forcing_direction']}, output={r['output']}, "
            f"SNR=({r['SNR1']:.3g},{r['SNR2']:.3g},{r['SNR3']:.3g},{r['SNR4']:.3g}), "
            f"p=({r['p1']:.3g},{r['p2']:.3g},{r['p3']:.3g},{r['p4']:.3g}), "
            f"A2/A1={r['A2_A1']:.3g} [{r['A2_A1_ci_low']:.3g}, {r['A2_A1_ci_high']:.3g}]"
            for r in candidates[:30]
        )

    text = f"""omega = {omega:.12g}
forcing amplitude = {amplitude:.12g}
forcing direction for harmonic summary = {direction}
output for harmonic summary = {output}

First-order response L(ω), complex convention cos + i sin:
{_format_array(L)}

Second-order H_mean:
{_format_array(H0)}

Second-order H_cos(2ω):
{_format_array(Hc)}

Second-order H_sin(2ω):
{_format_array(Hs)}

|H(2ω)|:
{_format_array(Ha)}

Harmonics:
          amplitude [95% CI]      noise [95% CI]      SNR [95% CI]      p
{chr(10).join(harmonic_lines)}

A2/A1 = {ratio['R21']:.6g} [{ratio['R21_ci_low']:.6g}, {ratio['R21_ci_high']:.6g}]

Amplitude grid:
{list(map(float, cfg.amplitudes))}

Omega grid:
{list(map(float, cfg.omegas))}

Seed-level retention:
response_by_seed has shape (n_seed, output, n_phase) for the report-selected omega/amplitude/primary forcing combination.
fourier_by_seed has shape (n_seed, output, harmonic, component[cos,sin]) for the same selected combination.
phase_response_by_seed_all has shape (n_omega, n_amp, n_combo, n_seed, output, n_phase).
fourier_by_seed_all has shape (n_omega, n_amp, n_combo, n_seed, output, harmonic, component[cos,sin]).
L_theta_by_seed and H_theta_by_seed retain matched-seed finite-difference tensors.

Confidence intervals:
All scalar amplitudes, noise floors, SNRs, A2/A1 ratios, L coefficients, and H coefficients use seed-level bootstrap means with n_bootstrap={cfg.n_bootstrap} and percentile 95% intervals.  Lorenz ODEs are not reintegrated during bootstrap.

Tensor definitions:
L_ij(theta) = (M_i(+A e_j, theta) - M_i(-A e_j, theta)) / (2 A).
H_ijj(theta) = (M_i(+A e_j, theta) + M_i(-A e_j, theta) - 2 M_i(0, theta)) / A^2.
H_ijk(theta) = (M_i(++ ) - M_i(+-) - M_i(-+) + M_i(--)) / (4 A^2) for j != k.
The same seed and spinup state are used across (+A, -A, 0) and mixed finite-difference evaluations.

Candidate amplitude intervals:
{candidate_text}

Generated figures:
{chr(10).join(figures)}
"""
    (out / "report.txt").write_text(text, encoding="utf-8")


def run(cfg: SineSecondConfig, output_dir: Path, workers: int) -> dict:
    started = time.time()
    out = Path(output_dir)
    for subdir in ("figures", "data"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    data = _run_parallel(cfg, workers)
    tensors = _finite_difference_tensors(data, cfg)
    scalar_rows, hotelling_rows, ratio_rows = _scalar_tables(data, tensors, cfg)
    first_rows, second_rows = _tensor_tables(tensors, cfg)
    candidates = _candidate_rows(scalar_rows, ratio_rows, cfg)
    figures = _write_figures(out, cfg, scalar_rows, ratio_rows, hotelling_rows, second_rows, data)

    scan_header = [
        "omega", "forcing_amp", "forcing_direction", "output", "harmonic",
        "signal_amp", "signal_amp_ci_low", "signal_amp_ci_high",
        "noise", "noise_ci_low", "noise_ci_high",
        "snr", "snr_ci_low", "snr_ci_high",
        "hotelling_T2", "hotelling_F", "hotelling_p",
        "raw_forced_target_amp", "raw_forced_target_amp_ci_low", "raw_forced_target_amp_ci_high",
    ]
    _write_csv(out / "scan_summary.csv", scalar_rows, scan_header)
    _write_csv(out / "harmonic_statistics.csv", scalar_rows, scan_header)
    hotelling_header = scan_header + [
        "ellipse_center_cos", "ellipse_center_sin",
        "ellipse_axis_major", "ellipse_axis_minor", "ellipse_angle_rad",
    ]
    _write_csv(out / "hotelling_results.csv", hotelling_rows, hotelling_header)
    _write_csv(
        out / "first_order_tensor.csv",
        first_rows,
        [
            "omega", "forcing_amp", "output_i", "forcing_j",
            "real", "real_ci_low", "real_ci_high",
            "imag", "imag_ci_low", "imag_ci_high",
            "amplitude", "amplitude_ci_low", "amplitude_ci_high",
            "phase", "phase_ci_low", "phase_ci_high",
        ],
    )
    second_header = [
        "omega", "forcing_amp", "output_i", "forcing_j", "forcing_k",
        "H_mean", "H_mean_ci_low", "H_mean_ci_high",
        "H_cos2", "H_cos2_ci_low", "H_cos2_ci_high",
        "H_sin2", "H_sin2_ci_low", "H_sin2_ci_high",
        "H_abs_2omega", "H_abs_2omega_ci_low", "H_abs_2omega_ci_high",
        "H_phase_2omega", "independent_symmetric_entry", "nonzero_by_ci",
    ]
    _write_csv(out / "second_order_tensor.csv", second_rows, second_header)
    _write_csv(
        out / "second_order_independent_nonzero.csv",
        [r for r in second_rows if r["independent_symmetric_entry"] and r["nonzero_by_ci"]],
        second_header,
    )
    _write_csv(
        out / "r21_statistics.csv",
        ratio_rows,
        ["omega", "forcing_amp", "forcing_direction", "output", "R21", "R21_ci_low", "R21_ci_high"],
    )
    _write_csv(
        out / "candidate_amplitude_intervals.csv",
        candidates,
        [
            "omega", "forcing_amp", "forcing_direction", "output",
            "SNR1", "SNR2", "SNR3", "SNR4",
            "p1", "p2", "p3", "p4",
            "A2_A1", "A2_A1_ci_low", "A2_A1_ci_high",
        ],
    )

    selected_omega_index = 0
    selected_amp_index = 0
    if candidates:
        selected_omega_index = list(map(float, cfg.omegas)).index(float(candidates[0]["omega"]))
        selected_amp_index = list(map(float, cfg.amplitudes)).index(float(candidates[0]["forcing_amp"]))
    selected_combo_index = list(map(str, data["combo_names"])).index(
        f"axis_{cfg.primary_forcing_direction}_pos"
    )
    np.savez_compressed(
        out / "numerical_response_tensors.npz",
        amplitudes=np.asarray(cfg.amplitudes, dtype=float),
        omegas=np.asarray(cfg.omegas, dtype=float),
        combo_names=data["combo_names"],
        coordinates=np.asarray(COORDINATES),
        component_names=np.asarray(COMPLEX_COMPONENTS),
        actual_seeds=data["actual_seeds"],
        n_skip_by_omega_seed=data["n_skip_by_omega_seed"],
        response_by_seed=data["phase_means"][selected_omega_index, selected_amp_index, selected_combo_index],
        fourier_by_seed=data["fourier_by_seed"][selected_omega_index, selected_amp_index, selected_combo_index],
        phase_response_by_seed_all=data["phase_means"],
        fourier_by_seed_all=data["fourier_by_seed"],
        raw_target_amp_by_seed=data["raw_target_amp_by_seed"],
        raw_noise_by_seed=data["raw_noise_by_seed"],
        phase_signal_amp_by_seed=data["phase_signal_amp_by_seed"],
        phase_signal_coeff_by_seed=data["phase_signal_coeff_by_seed"],
        raw_spectrum_mean=data["raw_spectrum_mean"],
        raw_spectrum_representative=data["raw_spectrum_representative"],
        L_theta_by_seed=tensors["L_theta_by_seed"],
        H_theta_by_seed=tensors["H_theta_by_seed"],
        L_theta_all=tensors["L_theta"],
        L_theta_lower_all=tensors["L_theta_lower"],
        L_theta_upper_all=tensors["L_theta_upper"],
        H_theta_all=tensors["H_theta"],
        H_theta_lower_all=tensors["H_theta_lower"],
        H_theta_upper_all=tensors["H_theta_upper"],
        L_fourier_by_seed=tensors["L_fourier_by_seed"],
        H_fourier_by_seed=tensors["H_fourier_by_seed"],
        H_theta=tensors["H_theta"][selected_omega_index, selected_amp_index],
        H_theta_lower=tensors["H_theta_lower"][selected_omega_index, selected_amp_index],
        H_theta_upper=tensors["H_theta_upper"][selected_omega_index, selected_amp_index],
    )
    metadata = {
        "config": asdict(cfg),
        "output_dir": str(out),
        "elapsed_seconds": time.time() - started,
        "files": [
            "scan_summary.csv",
            "harmonic_statistics.csv",
            "first_order_tensor.csv",
            "second_order_tensor.csv",
            "numerical_response_tensors.npz",
            "hotelling_results.csv",
            "report.txt",
        ],
    }
    (out / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(out, cfg, scalar_rows, first_rows, second_rows, ratio_rows, candidates, figures)
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)
    metadata = run(cfg, args.output_dir, int(args.workers))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
