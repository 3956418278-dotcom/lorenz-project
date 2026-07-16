"""First- and second-order sinusoidal response with paired seed statistics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from . import core, storage
from .config import n_cycle_for_frequency
from .fourier import fourier_coefficients, amplitude_phase


H_NONZERO_INDEPENDENT = [
    (0, 0, 2),
    (0, 1, 2),
    (1, 0, 2),
    (1, 1, 2),
    (2, 0, 0),
    (2, 0, 1),
    (2, 1, 1),
    (2, 2, 2),
]


def second_order_mask():
    """Mask for the 8 independent nonzero H_ijk components, with j-k symmetry."""
    mask = np.zeros((3, 3, 3), dtype=float)
    for i, j, k in H_NONZERO_INDEPENDENT:
        mask[i, j, k] = 1.0
        mask[i, k, j] = 1.0
    return mask


def forcing(direction, amp):
    v = np.zeros(3)
    v[direction] = amp
    return v


def forcing_pair(j, sj, k=None, sk=None, amp=0.0):
    v = np.zeros(3)
    v[j] = sj * amp
    if k is not None:
        v[k] = sk * amp
    return v


def get_spinup_state(seed, cfg, runtime):
    payload = {
        "seed": seed,
        "lorenz": cfg["lorenz"],
        "T_spinup": cfg["T_spinup"],
        "solver": cfg["solver"],
        "kind": "spinup_v1",
    }
    key = storage.cache_key(payload)
    path = storage.spinup_cache_path(key)
    try:
        data = storage.load_npz(path)
        runtime["cache"]["hits"] += 1
        return data["state"]
    except Exception:
        runtime["cache"]["misses"] += 1
    state = core.spinup_state(seed, cfg)
    storage.save_npz_atomic(path, state=state, payload=json.dumps(payload, sort_keys=True))
    return state


def phase_mean_cached(seed, forcing_vector, omega, phase, n_skip, n_cycle, n_phase,
                      cfg, runtime):
    payload = {
        "seed": seed,
        "forcing": np.asarray(forcing_vector, dtype=float).tolist(),
        "omega": omega,
        "phase": phase,
        "n_skip": n_skip,
        "n_cycle": n_cycle,
        "n_phase": n_phase,
        "T_spinup": cfg["T_spinup"],
        "lorenz": cfg["lorenz"],
        "solver": cfg["solver"],
        "kind": "phase_mean_v1",
    }
    key = storage.cache_key(payload)
    path = storage.phase_cache_path(key)
    try:
        data = storage.load_npz(path)
        runtime["cache"]["hits"] += 1
        return data["phase_mean"]
    except Exception:
        runtime["cache"]["misses"] += 1
    state = get_spinup_state(seed, cfg, runtime)
    samples = core.simulate_phase_samples(seed, forcing_vector, omega, phase, n_skip,
                                          n_cycle, n_phase, cfg, initial_state=state)
    phase_mean = samples.mean(1)
    storage.save_npz_atomic(path, phase_mean=phase_mean,
                            payload=json.dumps(payload, sort_keys=True))
    return phase_mean


def phase_group(forcing_vector, omega, n_skip, n_cycle, cfg, runtime):
    arr = []
    for seed in range(int(cfg["n_seed"])):
        arr.append(phase_mean_cached(seed, forcing_vector, omega, cfg.get("phase", 0.0),
                                     n_skip, n_cycle, cfg["n_phase"], cfg,
                                     runtime))
    return np.array(arr)


def mean_se_ci(seed_values):
    mean = seed_values.mean(0)
    n = seed_values.shape[0]
    if n > 1:
        se = seed_values.std(0, ddof=1) / np.sqrt(n)
        ci = stats.t.ppf(0.975, df=n - 1) * se
    else:
        se = np.zeros_like(mean)
        ci = np.zeros_like(mean)
    return mean, se, ci


def compute_seed_responses_for_amp(amp, omega, n_skip, n_cycle, cfg, runtime):
    zero = phase_group(np.zeros(3), omega, n_skip, n_cycle, cfg, runtime)
    L_seed = np.zeros((cfg["n_seed"], 3, 3, cfg["n_phase"]))
    H_seed = np.zeros((cfg["n_seed"], 3, 3, 3, cfg["n_phase"]))

    for j in range(3):
        pos = phase_group(forcing(j, amp), omega, n_skip, n_cycle, cfg, runtime)
        neg = phase_group(forcing(j, -amp), omega, n_skip, n_cycle, cfg, runtime)
        L_seed[:, :, j] = (pos - neg) / (2 * amp)
        H_seed[:, :, j, j] = (pos + neg - 2 * zero) / amp ** 2

    for j, k in [(0, 1), (0, 2), (1, 2)]:
        pp = phase_group(forcing_pair(j, 1, k, 1, amp), omega, n_skip, n_cycle, cfg, runtime)
        pm = phase_group(forcing_pair(j, 1, k, -1, amp), omega, n_skip, n_cycle, cfg, runtime)
        mp = phase_group(forcing_pair(j, -1, k, 1, amp), omega, n_skip, n_cycle, cfg, runtime)
        mm = phase_group(forcing_pair(j, -1, k, -1, amp), omega, n_skip, n_cycle, cfg, runtime)
        mixed = (pp - pm - mp + mm) / (4 * amp ** 2)
        H_seed[:, :, j, k] = mixed
        H_seed[:, :, k, j] = mixed
    H_seed *= second_order_mask()[None, :, :, :, None]
    return L_seed, H_seed


def extrapolate_seed(seed_resp, amplitudes, bootstrap_samples, random_seed):
    """seed_resp shape: (n_amp, n_seed, ...)."""
    amp2 = np.asarray(amplitudes) ** 2
    design = np.vstack([np.ones_like(amp2), amp2]).T
    means = seed_resp.mean(1)
    coef = np.linalg.lstsq(design, means.reshape(len(amplitudes), -1), rcond=None)[0]
    intercept = coef[0].reshape(means.shape[1:])
    slope = coef[1].reshape(means.shape[1:])
    rng = np.random.default_rng(random_seed)
    boots = []
    n_seed = seed_resp.shape[1]
    for _ in range(bootstrap_samples):
        ind = rng.integers(0, n_seed, n_seed)
        wrk = seed_resp[:, ind].mean(1)
        bcoef = np.linalg.lstsq(design, wrk.reshape(len(amplitudes), -1), rcond=None)[0]
        boots.append(bcoef[0].reshape(means.shape[1:]))
    boots = np.array(boots)
    if bootstrap_samples > 1:
        boot_se = boots.std(0, ddof=1)
        boot_lo, boot_hi = np.percentile(boots, [2.5, 97.5], axis=0)
    else:
        boot_se = np.zeros_like(intercept)
        boot_lo = intercept.copy()
        boot_hi = intercept.copy()
    return intercept, slope, boot_se, boot_lo, boot_hi


def compute_response_for_frequency(omega, n_skip, cfg, run_dir, runtime):
    n_cycle = n_cycle_for_frequency(omega, cfg)
    L_all = []
    H_all = []
    for amp in cfg["amplitudes"]:
        L_seed, H_seed = compute_seed_responses_for_amp(
            amp, omega, n_skip, n_cycle, cfg, runtime)
        L_all.append(L_seed)
        H_all.append(H_seed)
    L_seed = np.array(L_all)
    H_seed = np.array(H_all)
    L, L_slope, L_boot_se, L_boot_lo, L_boot_hi = extrapolate_seed(
        L_seed, cfg["amplitudes"], cfg["bootstrap_samples"], cfg["random_seed"])
    H, H_slope, H_boot_se, H_boot_lo, H_boot_hi = extrapolate_seed(
        H_seed, cfg["amplitudes"], cfg["bootstrap_samples"], cfg["random_seed"] + 1)
    kmax = cfg["fourier_kmax"]
    L_fourier = fourier_coefficients(L, kmax)
    H_fourier = fourier_coefficients(H, kmax)
    L_amp, L_phase = amplitude_phase(L_fourier)
    H_amp, H_phase = amplitude_phase(H_fourier)
    out = dict(
        omega=omega, n_skip=n_skip, n_cycle=n_cycle,
        L_seed=L_seed, H_seed=H_seed, L=L, H=H,
        L_slope=L_slope, H_slope=H_slope,
        L_boot_se=L_boot_se, H_boot_se=H_boot_se,
        L_boot_lo=L_boot_lo, L_boot_hi=L_boot_hi,
        H_boot_lo=H_boot_lo, H_boot_hi=H_boot_hi,
        L_fourier=L_fourier, H_fourier=H_fourier,
        L_amp=L_amp, L_phase=L_phase, H_amp=H_amp, H_phase=H_phase,
    )
    ckpt = run_dir / "checkpoints" / f"omega_{omega:.8g}.npz"
    storage.save_npz_atomic(ckpt, **out)
    return out


def response_summary_csv(path, responses):
    with Path(path).open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["omega", "quantity", "i", "j", "k", "harmonic", "cos", "sin", "amplitude", "phase"])
        for r in responses:
            omega = r["omega"]
            Lf = r["L_fourier"]
            Hf = r["H_fourier"]
            La = r["L_amp"]
            Lp = r["L_phase"]
            Ha = r["H_amp"]
            Hp = r["H_phase"]
            for i in range(3):
                for j in range(3):
                    for k in range(Lf.shape[-2]):
                        writer.writerow([omega, "L", i, j, "", k, Lf[i, j, k, 0], Lf[i, j, k, 1], La[i, j, k], Lp[i, j, k]])
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        for k in range(Hf.shape[-2]):
                            writer.writerow([omega, "H", i, j, l, k, Hf[i, j, l, k, 0], Hf[i, j, l, k, 1], Ha[i, j, l, k], Hp[i, j, l, k]])
