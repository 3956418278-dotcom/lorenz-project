"""Long-term periodic statistical steady-state checks."""

from __future__ import annotations

import numpy as np

from .fourier import fourier_coefficients
from .response import phase_mean_cached


def block_phase_means(omega, n_skip, cfg, runtime):
    steady = cfg["steady"]
    block_cycles = int(steady["block_cycles"])
    n_blocks = int(steady["n_blocks"])
    out = []
    forcing = np.array([steady["check_amplitude"], 0.0, 0.0])
    for block in range(n_blocks):
        means = []
        for seed in range(cfg["n_seed"]):
            means.append(phase_mean_cached(seed, forcing, omega, cfg.get("phase", 0.0),
                                           n_skip + block * block_cycles,
                                           block_cycles, cfg["n_phase"], cfg,
                                           runtime))
        out.append(np.array(means))
    return np.array(out)


def check_frequency(omega, cfg, runtime):
    steady = cfg["steady"]
    candidates = steady["n_skips"]
    results = []
    recommended = None
    converged = False
    for n_skip in candidates:
        blocks = block_phase_means(omega, int(n_skip), cfg, runtime)
        coefs_seed = np.array([[fourier_coefficients(seed_mean, cfg["fourier_kmax"]) for seed_mean in block] for block in blocks])
        coefs = coefs_seed.mean(1)
        se = coefs_seed.std(1, ddof=1) / np.sqrt(cfg["n_seed"]) if cfg["n_seed"] > 1 else np.zeros_like(coefs)
        delta = coefs[1:] - coefs[:-1]
        delta_se = np.sqrt(se[1:] ** 2 + se[:-1] ** 2)
        z = np.abs(delta) / np.maximum(delta_se, 1e-300)
        scale = np.maximum(np.maximum(np.abs(coefs[1:]), np.abs(coefs[:-1])), 1e-300)
        rel_ok = np.abs(delta) <= steady["atol"] + steady["rtol"] * scale
        ok = bool(np.all(z <= steady["z_threshold"]) and np.all(rel_ok))
        results.append((int(n_skip), ok, float(np.nanmax(z))))
    flags = np.array([r[1] for r in results])
    m = int(steady["min_consecutive"])
    for i in range(max(0, len(flags) - m + 1)):
        if np.all(flags[i:i + m]):
            recommended = int(results[i][0])
            converged = True
            break
    return {"omega": omega, "converged": converged,
            "recommended_n_skip": recommended, "rows": results}


def run(omegas, cfg, runtime):
    details = {}
    for omega in omegas:
        res = check_frequency(omega, cfg, runtime)
        details[str(omega)] = res
    return details
