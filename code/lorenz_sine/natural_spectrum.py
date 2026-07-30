"""Independent unforced spectrum stage with Welch uncertainty diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import integrate, signal, stats

from . import core, storage
from .config import validate_target_frequencies
from .response import get_spinup_state
from .statistics import mean_se_ci


DEFAULT_PEAK_SIGNIFICANCE = {
    "enabled": True,
    "peak_power": "nearest_bin",
    "peak_window_bins": 0,
    "exclude_bins": 2,
    "background_bins": 8,
    "psd_floor": 1e-300,
    "include_detected_peaks": True,
    "include_target_omegas": True,
    "include_target_frequencies": True,
    "target_frequencies": [],
    "target_frequency_bands": [],
    "fundamental_omegas": [],
    "fundamental_frequencies": [],
    "harmonic_count": 0,
    "strict_split": False,
    "discovery_seed_fraction": 0.5,
}


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


def _peak_significance_cfg(cfg):
    options = dict(DEFAULT_PEAK_SIGNIFICANCE)
    options.update(cfg.get("peak_significance", {}))
    options["enabled"] = bool(options.get("enabled", True))
    options["peak_window_bins"] = int(options["peak_window_bins"])
    options["exclude_bins"] = int(options["exclude_bins"])
    options["background_bins"] = int(options["background_bins"])
    options["psd_floor"] = float(options["psd_floor"])
    if options["peak_power"] not in ("nearest_bin", "window_max"):
        raise ValueError("peak_significance.peak_power must be nearest_bin or window_max")
    if options["peak_window_bins"] < 0:
        raise ValueError("peak_significance.peak_window_bins must be non-negative")
    if options["exclude_bins"] < 0:
        raise ValueError("peak_significance.exclude_bins must be non-negative")
    if options["background_bins"] <= 0:
        raise ValueError("peak_significance.background_bins must be positive")
    if options["psd_floor"] <= 0.0:
        raise ValueError("peak_significance.psd_floor must be positive")
    if int(options.get("harmonic_count", 0)) < 0:
        raise ValueError("peak_significance.harmonic_count must be non-negative")
    options["strict_split"] = bool(options.get("strict_split", False))
    options["discovery_seed_fraction"] = float(options["discovery_seed_fraction"])
    if not 0.0 < options["discovery_seed_fraction"] < 1.0:
        raise ValueError(
            "peak_significance.discovery_seed_fraction must lie strictly between zero and one"
        )
    return options


def _candidate_peak_specs(freqs, peaks, cfg, options, reference_psd=None):
    candidates = []
    seen = set()

    def add(source, rank, target_frequency):
        target_frequency = float(target_frequency)
        if target_frequency <= 0.0 or target_frequency > float(freqs[-1]):
            return
        bin_index = int(np.argmin(np.abs(freqs - target_frequency)))
        key = (source, str(rank), bin_index)
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "source": source,
            "peak_rank": rank,
            "target_frequency": target_frequency,
            "nearest_bin": bin_index,
        })

    def add_band(rank, band):
        if isinstance(band, dict):
            band_min = float(band["min"])
            band_max = float(band["max"])
            label = band.get("label", f"band_{rank}")
        else:
            band_min = float(band[0])
            band_max = float(band[1])
            label = f"band_{rank}"
        if band_min <= 0.0 or band_max <= band_min:
            raise ValueError("target frequency bands must have 0 < min < max")
        window = np.flatnonzero((freqs >= band_min) & (freqs <= band_max))
        if window.size == 0:
            return
        if reference_psd is None:
            selected_bin = int(window[window.size // 2])
        else:
            selected_bin = int(window[int(np.argmax(reference_psd[window]))])
        key = ("target_frequency_band", str(label), int(window[0]), int(window[-1]))
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "source": "target_frequency_band",
            "peak_rank": label,
            "target_frequency": float(0.5 * (band_min + band_max)),
            "nearest_bin": selected_bin,
            "window_start_bin": int(window[0]),
            "window_stop_bin": int(window[-1]) + 1,
            "target_band_min": band_min,
            "target_band_max": band_max,
            "select_peak_within_window": False,
        })

    if options.get("include_detected_peaks", True):
        for row in peaks:
            add("detected_peak", int(row[0]), float(row[2]))
    if options.get("include_target_omegas", True):
        for index, omega in enumerate(cfg.get("target_omegas", []), start=1):
            add("target_omega", f"target_{index}", float(omega) / (2.0 * math.pi))
    if options.get("include_target_frequencies", True):
        for index, frequency in enumerate(options.get("target_frequencies", []), start=1):
            add("target_frequency", f"target_f_{index}", float(frequency))
        for index, band in enumerate(options.get("target_frequency_bands", []), start=1):
            add_band(index, band)
    harmonic_count = int(options.get("harmonic_count", 0))
    for base_index, omega in enumerate(options.get("fundamental_omegas", []), start=1):
        fundamental = float(omega) / (2.0 * math.pi)
        for harmonic in range(1, harmonic_count + 1):
            add(
                "configured_harmonic",
                f"omega{base_index}_k{harmonic}",
                harmonic * fundamental,
            )
    for base_index, frequency in enumerate(
        options.get("fundamental_frequencies", []), start=1):
        fundamental = float(frequency)
        for harmonic in range(1, harmonic_count + 1):
            add(
                "configured_frequency_harmonic",
                f"f{base_index}_k{harmonic}",
                harmonic * fundamental,
            )
    return candidates


def _holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order, start=1):
        value = min(1.0, (m - rank + 1) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _one_sided_t_greater(values, confidence_level):
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    mean = float(np.mean(values))
    if n > 1:
        std = float(np.std(values, ddof=1))
        se = std / math.sqrt(n)
        df = n - 1
        if se == 0.0:
            t_statistic = math.inf if mean > 0.0 else 0.0
            p_value = 0.0 if mean > 0.0 else 1.0
        else:
            t_statistic = mean / se
            p_value = float(stats.t.sf(t_statistic, df=df))
        critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=df))
        half = critical * se
        return mean, std, se, t_statistic, df, p_value, mean - half, mean + half
    return mean, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan


def _peak_background_slices(freqs, center_bin, options, window_start=None, window_stop=None):
    exclude = int(options["exclude_bins"])
    width = int(options["background_bins"])
    if window_start is None:
        window_start = int(center_bin)
    if window_stop is None:
        window_stop = int(center_bin) + 1
    left_stop = max(0, int(window_start) - exclude)
    left_start = max(0, left_stop - width)
    right_start = min(len(freqs), int(window_stop) + exclude)
    right_stop = min(len(freqs), right_start + width)
    if left_stop <= left_start or right_stop <= right_start:
        raise ValueError(
            "peak significance background bands are empty; reduce exclude_bins "
            "or background_bins, or avoid edge frequencies"
        )
    return left_start, left_stop, right_start, right_stop


def _log_peak_background_ratios(psd_seed, freqs, candidates, cfg, selection_mode):
    options = _peak_significance_cfg(cfg)
    labels = ("x", "y", "z")
    confidence = float(cfg.get("confidence_level", 0.95))
    alpha = float(cfg.get("significance_alpha", 0.05))
    floor = float(options["psd_floor"])
    rows = []
    d_values = []
    row_coordinates = []
    for candidate in candidates:
        target_bin = int(candidate["nearest_bin"])
        window = int(options["peak_window_bins"])
        if "window_start_bin" in candidate:
            window_start = int(candidate["window_start_bin"])
            window_stop = int(candidate["window_stop_bin"])
        else:
            window_start = max(0, target_bin - window)
            window_stop = min(len(freqs), target_bin + window + 1)
        for coordinate_index, coordinate in enumerate(labels):
            local_mean = np.mean(psd_seed[:, coordinate_index, window_start:window_stop],
                                 axis=0)
            if candidate.get("select_peak_within_window", False):
                tested_bin = window_start + int(np.argmax(local_mean))
            elif "window_start_bin" in candidate:
                tested_bin = target_bin
            elif options["peak_power"] == "window_max":
                tested_bin = window_start + int(np.argmax(local_mean))
            else:
                tested_bin = target_bin
            left_start, left_stop, right_start, right_stop = _peak_background_slices(
                freqs,
                tested_bin,
                options,
                window_start=window_start if "window_start_bin" in candidate else None,
                window_stop=window_stop if "window_stop_bin" in candidate else None,
            )
            peak_power = psd_seed[:, coordinate_index, tested_bin]
            background = np.concatenate([
                psd_seed[:, coordinate_index, left_start:left_stop],
                psd_seed[:, coordinate_index, right_start:right_stop],
            ], axis=1)
            background_power = np.median(background, axis=1)
            d = np.log(np.maximum(peak_power, floor)) - np.log(
                np.maximum(background_power, floor))
            mean, std, se, t_stat, df, p_value, ci_low, ci_high = _one_sided_t_greater(
                d, confidence)
            d_values.append(d)
            row_coordinates.append(coordinate)
            rows.append({
                "coordinate": coordinate,
                "selection_mode": selection_mode,
                "peak_source": candidate["source"],
                "peak_rank": candidate["peak_rank"],
                "target_frequency": float(candidate["target_frequency"]),
                "target_band_min": float(candidate.get("target_band_min", np.nan)),
                "target_band_max": float(candidate.get("target_band_max", np.nan)),
                "tested_frequency": float(freqs[tested_bin]),
                "nearest_bin": int(target_bin),
                "tested_bin": int(tested_bin),
                "peak_power_method": options["peak_power"],
                "peak_window_bins": int(options["peak_window_bins"]),
                "exclude_bins": int(options["exclude_bins"]),
                "background_bins_per_side": int(options["background_bins"]),
                "background_left_min": float(freqs[left_start]),
                "background_left_max": float(freqs[left_stop - 1]),
                "background_right_min": float(freqs[right_start]),
                "background_right_max": float(freqs[right_stop - 1]),
                "n_seed": int(d.size),
                "mean_log_peak_ratio": mean,
                "std_log_peak_ratio": std,
                "standard_error": se,
                "t_statistic": t_stat,
                "degrees_of_freedom": df,
                "p_value_one_sided": p_value,
                "p_value_adjusted": np.nan,
                "confidence_interval_low": ci_low,
                "confidence_interval_high": ci_high,
                "significant": False,
                "significance_alpha": alpha,
                "confidence_level": confidence,
            })
    for coordinate in labels:
        indices = [index for index, value in enumerate(row_coordinates)
                   if value == coordinate]
        adjusted = _holm_adjust([rows[index]["p_value_one_sided"] for index in indices])
        for local_index, row_index in enumerate(indices):
            rows[row_index]["p_value_adjusted"] = float(adjusted[local_index])
            rows[row_index]["significant"] = bool(adjusted[local_index] < alpha)
    return rows, np.asarray(d_values, dtype=float)


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
    significance_rows = result.get("peak_significance_rows", [])
    if significance_rows:
        header = [
            "coordinate", "selection_mode", "peak_source", "peak_rank",
            "target_frequency", "target_band_min", "target_band_max",
            "tested_frequency", "nearest_bin", "tested_bin", "peak_power_method",
            "peak_window_bins", "exclude_bins", "background_bins_per_side",
            "background_left_min",
            "background_left_max", "background_right_min",
            "background_right_max", "n_seed", "mean_log_peak_ratio",
            "std_log_peak_ratio", "standard_error", "t_statistic",
            "degrees_of_freedom", "p_value_one_sided", "p_value_adjusted",
            "confidence_interval_low", "confidence_interval_high",
            "significant", "significance_alpha", "confidence_level",
        ]
        storage.write_csv(
            run_dir / "tables" / "spectrum_peak_significance.csv",
            header,
            [[row[key] for key in header] for row in significance_rows],
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

    if significance_rows:
        _write_peak_significance_figure(run_dir, result, cfg)


def _write_peak_significance_figure(run_dir: Path, result: dict, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    labels = ("x", "y", "z")
    freqs = result["freqs"]
    rows = result["peak_significance_rows"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for coordinate_index, coordinate in enumerate(labels):
        psd_axis = axes[coordinate_index, 0]
        ratio_axis = axes[coordinate_index, 1]
        use = freqs > 0.0
        psd_axis.loglog(freqs[use], result["psd_mean"][coordinate_index, use],
                        color="C0", linewidth=0.9)
        coord_rows = [row for row in rows if row["coordinate"] == coordinate]
        for row in coord_rows:
            color = "tab:green" if row["significant"] else "tab:red"
            psd_axis.axvline(row["tested_frequency"], color=color, alpha=0.45,
                             linewidth=0.8)
        psd_axis.set_ylabel(f"{coordinate} PSD")
        psd_axis.grid(True, which="both", alpha=0.2)

        x = np.arange(len(coord_rows))
        means = np.asarray([row["mean_log_peak_ratio"] for row in coord_rows])
        low = np.asarray([row["confidence_interval_low"] for row in coord_rows])
        high = np.asarray([row["confidence_interval_high"] for row in coord_rows])
        colors = ["tab:green" if row["significant"] else "tab:red"
                  for row in coord_rows]
        ratio_axis.errorbar(
            x,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="none",
            ecolor="0.35",
            capsize=3,
            linewidth=0.8,
        )
        ratio_axis.scatter(x, means, c=colors, zorder=3)
        ratio_axis.axhline(0.0, color="black", linewidth=0.8)
        ratio_axis.set_xticks(
            x,
            [f"{row['peak_rank']}\n{row['tested_frequency']:.3g}Hz"
             for row in coord_rows],
            rotation=0,
            fontsize=7,
        )
        ratio_axis.set_ylabel("mean log peak/background")
        ratio_axis.grid(True, axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("frequency")
    axes[-1, 1].set_xlabel("candidate peak")
    fig.suptitle(
        "Natural spectrum peak significance\n"
        "green: Holm-adjusted significant; red: not significant"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "spectrum_peak_significance.pdf")
    fig.savefig(run_dir / "figures" / "spectrum_peak_significance.png", dpi=160)
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
    significance_options = _peak_significance_cfg(cfg)
    peak_significance_rows = []
    peak_significance_values = np.empty((0, int(cfg["n_seed"])), dtype=float)
    significance_peaks = peaks
    significance_psd_seed = psd_seed
    selection_mode = "exploratory_same_seed_peak_selection"
    if significance_options["enabled"]:
        if significance_options["strict_split"]:
            n_seed = psd_seed.shape[0]
            n_discovery = int(round(
                n_seed * float(significance_options["discovery_seed_fraction"])))
            n_discovery = max(1, min(n_seed - 1, n_discovery))
            discovery_seed = psd_seed[:n_discovery]
            significance_psd_seed = psd_seed[n_discovery:]
            discovery_mean = discovery_seed.mean(axis=0)
            discovery_area = trapezoid(discovery_mean, freqs, axis=-1)
            discovery_combined = (
                discovery_mean / np.maximum(discovery_area[:, None], 1e-300)
            ).mean(axis=0)
            significance_peaks = _peak_rows(freqs, discovery_combined, cfg)
            selection_mode = (
                f"strict_split_discovery_seed_0_to_{n_discovery - 1}"
                f"_test_seed_{n_discovery}_to_{n_seed - 1}"
            )
        reference_combined = discovery_combined if significance_options["strict_split"] else combined
        candidates = _candidate_peak_specs(
            freqs,
            significance_peaks,
            cfg,
            significance_options,
            reference_psd=reference_combined,
        )
        peak_significance_rows, peak_significance_values = (
            _log_peak_background_ratios(
                significance_psd_seed, freqs, candidates, cfg, selection_mode)
        )
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
        "peak_significance_rows": peak_significance_rows,
        "peak_significance_values": peak_significance_values,
        "peak_significance_options": significance_options,
        "sampling_metadata": sampling_metadata,
        "target_frequency_checks": checks,
    }
    _write_outputs(run_dir, result, cfg)
    return result
