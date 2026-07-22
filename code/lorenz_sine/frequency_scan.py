"""Independent cross-frequency summary of previously estimated responses."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import storage
from .statistics import mean_se_ci


def _component_norm(values, norm):
    values = np.asarray(values, dtype=float)
    component_axes = tuple(range(3, values.ndim))
    if norm == "frobenius":
        return np.sqrt(np.sum(values ** 2, axis=component_axes))
    if norm == "rms":
        return np.sqrt(np.mean(values ** 2, axis=component_axes))
    raise ValueError(f"unsupported frequency-scan norm {norm!r}")


def run(cfg, response_result, run_dir):
    run_dir = Path(run_dir)
    harmonics = np.asarray(cfg["harmonics"], dtype=int)
    L_seed = np.asarray(response_result["L_harmonic_amplitude_seed"])
    H_seed = np.asarray(response_result["H_harmonic_amplitude_seed"])
    # omega, seed, response indices..., harmonic -> select and move harmonic before components
    L_selected = np.moveaxis(L_seed[..., harmonics], -1, 2)
    H_selected = np.moveaxis(H_seed[..., harmonics], -1, 2)
    L_norm_seed = _component_norm(L_selected, cfg["norm"])
    H_norm_seed = _component_norm(H_selected, cfg["norm"])
    confidence = cfg.get("confidence_level", 0.95)
    L_mean, L_se, L_low, L_high = mean_se_ci(L_norm_seed, confidence, axis=1)
    H_mean, H_se, H_low, H_high = mean_se_ci(H_norm_seed, confidence, axis=1)
    result = {
        "task": "frequency-scan",
        "frequencies": np.asarray(response_result["frequencies"]),
        "harmonics": harmonics,
        "norm": cfg["norm"],
        "L_norm_seed": L_norm_seed,
        "L_norm_mean": L_mean,
        "L_norm_se": L_se,
        "L_norm_ci_low": L_low,
        "L_norm_ci_high": L_high,
        "H_norm_seed": H_norm_seed,
        "H_norm_mean": H_mean,
        "H_norm_se": H_se,
        "H_norm_ci_low": H_low,
        "H_norm_ci_high": H_high,
    }
    rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for harmonic_index, harmonic in enumerate(harmonics):
            for quantity in ("L", "H"):
                rows.append([
                    omega, quantity, harmonic, cfg["norm"],
                    result[f"{quantity}_norm_mean"][omega_index, harmonic_index],
                    result[f"{quantity}_norm_se"][omega_index, harmonic_index],
                    result[f"{quantity}_norm_ci_low"][omega_index, harmonic_index],
                    result[f"{quantity}_norm_ci_high"][omega_index, harmonic_index],
                ])
    storage.write_csv(
        run_dir / "tables" / "frequency_response.csv",
        ["omega", "quantity", "harmonic", "norm", "mean", "se", "ci_low", "ci_high"],
        rows,
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for axis, quantity in zip(axes, ("L", "H")):
        for harmonic_index, harmonic in enumerate(harmonics):
            mean = result[f"{quantity}_norm_mean"][:, harmonic_index]
            low = result[f"{quantity}_norm_ci_low"][:, harmonic_index]
            high = result[f"{quantity}_norm_ci_high"][:, harmonic_index]
            axis.plot(result["frequencies"], mean, marker="o", label=f"k={harmonic}")
            axis.fill_between(result["frequencies"], low, high, alpha=0.15)
        axis.set_ylabel(f"{quantity} {cfg['norm']} norm")
        axis.legend(frameon=False)
    axes[-1].set_xlabel("angular forcing frequency omega")
    fig.suptitle("Frequency response with finite-seed confidence bands")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "frequency_response.pdf")
    fig.savefig(run_dir / "figures" / "frequency_response.png", dpi=160)
    plt.close(fig)
    return result
