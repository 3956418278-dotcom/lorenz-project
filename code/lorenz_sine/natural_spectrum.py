"""Unforced Welch spectrum used to choose candidate forcing frequencies."""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import core


def trapz_compat(y, x, axis=-1):
    return getattr(np, "trapz", np.trapezoid)(y, x, axis=axis)


def spectrum_for_seed(seed, cfg):
    spec = cfg["spectrum"]
    local = dict(cfg)
    local["T_spinup"] = cfg["T_spinup"]
    state = core.spinup_state(seed, local)
    density = int(spec["density"])
    T = float(spec["T"])
    t_eval = np.r_[:int(T * density)] / density
    sol = __import__("scipy.integrate").integrate.solve_ivp(
        lambda t, s: core.lorenz_rhs(t, s, cfg),
        [0, T], state, t_eval=t_eval, **cfg["solver"])
    core.check_solution(sol, (3, t_eval.size))
    nperseg = min(int(spec["segment_time"] * density), sol.y.shape[-1])
    freqs, psd = signal.welch(sol.y, fs=density, window="hann", nperseg=nperseg,
                              noverlap=nperseg // 2, detrend="constant", axis=-1,
                              scaling="density")
    return freqs, psd


def run(cfg):
    psds = []
    freqs = None
    for seed in range(int(cfg["n_seed"])):
        freqs, psd = spectrum_for_seed(seed, cfg)
        psds.append(psd)
    psds = np.array(psds)
    mean = psds.mean(0)
    se = psds.std(0, ddof=1) / np.sqrt(psds.shape[0]) if psds.shape[0] > 1 else np.zeros_like(mean)
    norm = trapz_compat(mean, freqs, axis=-1)
    combined = (mean / np.maximum(norm[:, None], 1e-300)).mean(0)
    work = combined[freqs >= cfg["spectrum"]["f_min"]]
    base = np.flatnonzero(freqs >= cfg["spectrum"]["f_min"])
    peaks, props = signal.find_peaks(work / max(work.max(), 1e-300),
                                     prominence=cfg["spectrum"]["prominence"])
    if peaks.size == 0:
        peaks = np.array([work.argmax()])
        props = {"prominences": np.array([np.nan])}
    order = np.argsort(work[peaks])[::-1][:cfg["spectrum"]["top_n"]]
    rows = []
    for rank, p in enumerate(peaks[order], 1):
        idx = base[p]
        f = freqs[idx]
        omega = 2 * np.pi * f
        delta_omega = 2 * np.pi / cfg["spectrum"]["segment_time"]
        rows.append([rank, f, omega, max(0, omega - 3 * delta_omega),
                     omega + 3 * delta_omega, combined[idx],
                     props["prominences"][order[rank - 1]]])
    rows = np.array(rows, dtype=float)
    return {"freqs": freqs, "psd": mean, "se": se, "combined": combined, "peaks": rows}
