"""Independent unforced spectrum stage with Welch uncertainty diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import integrate, signal

from . import core, storage
from .config import validate_target_frequencies
from .response import get_spinup_state
from .statistics import mean_se_ci


def spectrum_for_seed(seed, cfg):
    local_runtime = storage.create_runtime()
    state = get_spinup_state(seed, cfg, local_runtime)
    n_samples = int(cfg["n_record_samples"])
    sample_rate = float(cfg["sample_rate"])
    times = np.arange(n_samples, dtype=float) / sample_rate
    sol = integrate.solve_ivp(
        lambda t, state_: core.lorenz_rhs(t, state_, cfg),
        [0.0, float(times[-1])],
        state,
        t_eval=times,
        **cfg["solver"],
    )
    core.check_solution(sol, (3, n_samples))
    freqs, psd = signal.welch(
        sol.y,
        fs=sample_rate,
        window="hann",
        nperseg=int(cfg["welch_segment_length"]),
        noverlap=int(cfg["welch_overlap_samples"]),
        nfft=int(cfg["fft_length"]),
        detrend="constant",
        axis=-1,
        scaling="density",
    )
    return int(seed), freqs, psd, dict(local_runtime["cache"])


def _load_seed_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return int(data["seed"]), data["freqs"], data["psd"], {"hits": 0, "misses": 0}


def _peak_rows(freqs, combined, cfg):
    base = np.flatnonzero(freqs >= float(cfg["f_min"]))
    if base.size == 0:
        raise ValueError("f_min leaves no analyzable Welch frequency bins")
    work = combined[base]
    peaks, properties = signal.find_peaks(
        work / max(float(work.max()), 1e-300),
        prominence=float(cfg["prominence"]),
    )
    if peaks.size == 0:
        peaks = np.array([int(work.argmax())])
        properties = {"prominences": np.array([np.nan])}
    order = np.argsort(work[peaks])[::-1][:int(cfg["top_n"])]
    resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else np.nan
    rows = []
    for rank, order_index in enumerate(order, 1):
        local_index = int(peaks[order_index])
        index = int(base[local_index])
        frequency = float(freqs[index])
        omega = 2.0 * math.pi * frequency
        rows.append([
            rank,
            index,
            frequency,
            omega,
            max(0.0, omega - 3.0 * 2.0 * math.pi * resolution),
            omega + 3.0 * 2.0 * math.pi * resolution,
            float(combined[index]),
            float(properties["prominences"][order_index]),
        ])
    return np.asarray(rows, dtype=float)


def _write_outputs(run_dir: Path, result: dict, cfg: dict) -> None:
    storage.write_csv(
        run_dir / "tables" / "spectrum_peaks.csv",
        ["rank", "bin", "frequency", "omega", "omega_low", "omega_high",
         "combined_psd", "prominence"],
        result["peaks"],
    )
    checks = result["target_frequency_checks"]
    storage.write_csv(
        run_dir / "tables" / "frequency_resolution_checks.csv",
        ["kind", "omega", "frequency", "frequency_resolution",
         "nyquist_frequency", "resolved", "nearest_fft_bin"],
        [[row[key] for key in ("kind", "omega", "frequency",
                              "frequency_resolution", "nyquist_frequency",
                              "resolved", "nearest_fft_bin")] for row in checks],
    )
    storage.write_json(run_dir / "data" / "sampling_metadata.json",
                       result["sampling_metadata"])

    import matplotlib.pyplot as plt

    labels = ("x", "y", "z")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for output, axis in enumerate(axes):
        frequencies = result["freqs"]
        use = frequencies > 0
        axis.loglog(frequencies[use], result["psd_mean"][output, use],
                    label=f"{labels[output]} PSD")
        axis.fill_between(
            frequencies[use],
            np.maximum(result["psd_ci_low"][output, use], 1e-300),
            np.maximum(result["psd_ci_high"][output, use], 1e-300),
            alpha=0.25,
            label=f"{100 * cfg.get('confidence_level', 0.95):.1f}% CI",
        )
        for row in result["peaks"]:
            axis.axvline(row[2], color="tab:red", alpha=0.35, linewidth=0.8)
        axis.set_ylabel("PSD")
        axis.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("frequency")
    fig.suptitle(
        "Natural spectrum\n"
        f"resolution={result['sampling_metadata']['frequency_resolution']:.4g}, "
        f"Nyquist={result['sampling_metadata']['nyquist_frequency']:.4g}, "
        f"Welch={cfg['welch_segment_length']}, overlap={cfg['welch_overlap_samples']}"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "natural_spectrum.pdf")
    fig.savefig(run_dir / "figures" / "natural_spectrum.png", dpi=160)
    plt.close(fig)


def run(cfg, run_dir, runtime, resume=False, client=None):
    run_dir = Path(run_dir)
    completed = storage.completed_parameters(run_dir) if resume else set()
    seed_results = {}
    pending = []
    for seed in range(int(cfg["n_seed"])):
        key = f"seed:{seed}"
        checkpoint = storage.checkpoint_path(run_dir, key)
        if key in completed and storage.valid_npz(checkpoint, ("seed", "freqs", "psd")):
            seed_results[seed] = _load_seed_checkpoint(checkpoint)
        else:
            pending.append(seed)

    def save_seed(seed, value):
        _, freqs, psd, cache_stats = value
        checkpoint = storage.checkpoint_path(run_dir, f"seed:{seed}")
        storage.save_npz_atomic(checkpoint, seed=int(seed), freqs=freqs, psd=psd)
        storage.merge_cache_stats(runtime, cache_stats)
        storage.mark_parameter(run_dir, f"seed:{seed}", runtime)
        seed_results[int(seed)] = value

    if pending:
        from .parallel import execute_tasks

        def mark_failed(seed, exc):
            storage.mark_parameter_failed(run_dir, f"seed:{seed}", exc)

        execute_tasks(
            pending,
            spectrum_for_seed,
            client=client,
            worker_kwargs={"cfg": cfg},
            on_result=save_seed,
            on_error=mark_failed,
        )
    ordered = [seed_results[seed] for seed in range(int(cfg["n_seed"]))]
    freqs = ordered[0][1]
    psd_seed = np.asarray([item[2] for item in ordered])
    psd_mean, psd_se, psd_low, psd_high = mean_se_ci(
        psd_seed, cfg.get("confidence_level", 0.95), axis=0)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    area = trapezoid(psd_mean, freqs, axis=-1)
    combined = (psd_mean / np.maximum(area[:, None], 1e-300)).mean(axis=0)
    peaks = _peak_rows(freqs, combined, cfg)
    sampling_metadata = {
        "n_record_samples": int(cfg["n_record_samples"]),
        "record_duration": int(cfg["n_record_samples"]) / float(cfg["sample_rate"]),
        "sample_rate": float(cfg["sample_rate"]),
        "fft_length": int(cfg["fft_length"]),
        "welch_segment_length": int(cfg["welch_segment_length"]),
        "welch_overlap_samples": int(cfg["welch_overlap_samples"]),
        "frequency_resolution": float(cfg["sample_rate"]) / int(cfg["fft_length"]),
        "nyquist_frequency": float(cfg["sample_rate"]) / 2.0,
    }
    checks = validate_target_frequencies(
        cfg.get("target_omegas", []),
        cfg["sample_rate"],
        cfg["fft_length"],
        cfg.get("combination_omegas", []),
    )
    result = {
        "task": "spectrum",
        "freqs": freqs,
        "psd_seed": psd_seed,
        "psd_mean": psd_mean,
        "psd_se": psd_se,
        "psd_ci_low": psd_low,
        "psd_ci_high": psd_high,
        "combined_psd": combined,
        "peaks": peaks,
        "sampling_metadata": sampling_metadata,
        "target_frequency_checks": checks,
    }
    _write_outputs(run_dir, result, cfg)
    return result
