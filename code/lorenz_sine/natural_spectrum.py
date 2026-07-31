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


COORDINATES = ("x", "y", "z")


DEFAULT_PEAK_SIGNIFICANCE = {
    "enabled": True,
    "peak_power": "nearest_bin",
    "peak_window_bins": 0,
    "exclude_bins": 2,
    "background_bins": 8,
    "psd_floor": 1e-300,
    "include_detected_peaks": True,
    "predefined_target_frequencies": [],
    "strict_split": True,
    "discovery_seed_fraction": 0.5,
    "seed_split_strategy": "contiguous_seed_index",
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


def _frequency_resolution(freqs):
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2:
        raise ValueError("Welch frequency grid must be one-dimensional with at least two bins")
    differences = np.diff(freqs)
    if np.any(differences <= 0.0):
        raise ValueError("Welch frequency grid must be strictly increasing")
    resolution = float(np.median(differences))
    if not np.allclose(differences, resolution, rtol=1e-8, atol=1e-12):
        raise ValueError("Welch frequency grid must be uniformly spaced")
    return resolution


def _detect_peaks_by_coordinate(freqs, discovery_psd_mean, cfg):
    """Detect and rank peaks independently on each discovery-seed mean PSD."""

    freqs = np.asarray(freqs, dtype=float)
    discovery_psd_mean = np.asarray(discovery_psd_mean, dtype=float)
    if discovery_psd_mean.shape != (len(COORDINATES), freqs.size):
        raise ValueError(
            "discovery mean PSD must have shape "
            f"({len(COORDINATES)}, {freqs.size})"
        )
    base = np.flatnonzero(freqs >= float(cfg["f_min"]))
    if base.size == 0:
        raise ValueError("f_min leaves no analyzable Welch frequency bins")
    resolution = _frequency_resolution(freqs)
    minimum_distance_frequency = float(
        cfg.get("peak_min_distance_frequency", 3.0 * resolution)
    )
    if minimum_distance_frequency <= 0.0:
        raise ValueError("peak_min_distance_frequency must be positive")
    minimum_distance_bins = max(
        1, int(math.ceil(minimum_distance_frequency / resolution))
    )
    relative_prominence = float(cfg["prominence"])
    if relative_prominence <= 0.0:
        raise ValueError("prominence must be positive")
    top_n = int(cfg["top_n"])
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    rows = []
    for coordinate_index, coordinate in enumerate(COORDINATES):
        work = discovery_psd_mean[coordinate_index, base]
        scale = max(float(np.max(work)), 1e-300)
        local_peaks, properties = signal.find_peaks(
            work,
            prominence=relative_prominence * scale,
            width=(None, None),
        )
        if local_peaks.size == 0:
            continue

        # SciPy's distance filter prioritizes peak height.  Here the scientific
        # ranking variable is prominence, so perform the distance suppression
        # in prominence order as well.
        order = np.lexsort((
            base[local_peaks],
            -np.asarray(properties["prominences"], dtype=float),
        ))
        selected = []
        for property_index in order:
            global_bin = int(base[int(local_peaks[property_index])])
            if any(
                abs(global_bin - kept_bin) < minimum_distance_bins
                for _, kept_bin in selected
            ):
                continue
            selected.append((int(property_index), global_bin))
            if len(selected) == top_n:
                break

        for rank, (property_index, global_bin) in enumerate(selected, start=1):
            frequency = float(freqs[global_bin])
            rows.append({
                "coordinate": coordinate,
                "rank": rank,
                "frequency": frequency,
                "omega": 2.0 * math.pi * frequency,
                "bin": global_bin,
                "psd_value": float(discovery_psd_mean[coordinate_index, global_bin]),
                "prominence": float(properties["prominences"][property_index]),
                "width": float(properties["widths"][property_index]),
            })
    return rows


def _legacy_z_peak_array(detected_peaks, frequency_resolution):
    """Keep the historical numeric ``peaks`` key sourced from formal z peaks."""

    rows = []
    for peak in detected_peaks:
        if peak["coordinate"] != "z":
            continue
        half_width = 0.5 * float(peak["width"]) * float(frequency_resolution)
        omega_low = 2.0 * math.pi * max(0.0, float(peak["frequency"]) - half_width)
        omega_high = 2.0 * math.pi * (float(peak["frequency"]) + half_width)
        rows.append([
            int(peak["rank"]),
            int(peak["bin"]),
            float(peak["frequency"]),
            float(peak["omega"]),
            omega_low,
            omega_high,
            float(peak["psd_value"]),
            float(peak["prominence"]),
        ])
    if not rows:
        return np.empty((0, 8), dtype=float)
    return np.asarray(rows, dtype=float)


def _peak_significance_cfg(cfg):
    options = dict(DEFAULT_PEAK_SIGNIFICANCE)
    options.update(cfg.get("peak_significance", {}))
    options["enabled"] = bool(options.get("enabled", True))
    options["peak_window_bins"] = int(options["peak_window_bins"])
    options["exclude_bins"] = int(options["exclude_bins"])
    options["background_bins"] = int(options["background_bins"])
    options["psd_floor"] = float(options["psd_floor"])
    if options["peak_power"] != "nearest_bin":
        raise ValueError(
            "peak_significance.peak_power must be nearest_bin so test seeds "
            "cannot select a peak bin"
        )
    if options["peak_window_bins"] < 0:
        raise ValueError("peak_significance.peak_window_bins must be non-negative")
    if options["exclude_bins"] < 0:
        raise ValueError("peak_significance.exclude_bins must be non-negative")
    if options["background_bins"] <= 0:
        raise ValueError("peak_significance.background_bins must be positive")
    if options["psd_floor"] <= 0.0:
        raise ValueError("peak_significance.psd_floor must be positive")
    targets = [
        float(value)
        for value in options.get("predefined_target_frequencies", [])
    ]
    if any(value <= 0.0 for value in targets):
        raise ValueError(
            "peak_significance.predefined_target_frequencies entries must be positive"
        )
    options["predefined_target_frequencies"] = targets
    # A split is mandatory for this analysis.  Keep the field in saved
    # metadata for compatibility with older configurations, but never permit
    # same-seed discovery and testing.
    options["strict_split"] = True
    options["discovery_seed_fraction"] = float(options["discovery_seed_fraction"])
    if not 0.0 < options["discovery_seed_fraction"] < 1.0:
        raise ValueError(
            "peak_significance.discovery_seed_fraction must lie strictly between zero and one"
        )
    strategy = str(options.get("seed_split_strategy", "contiguous_seed_index"))
    if strategy != "contiguous_seed_index":
        raise ValueError(
            "peak_significance.seed_split_strategy must be contiguous_seed_index"
        )
    options["seed_split_strategy"] = strategy
    return options


def _candidate_peak_specs(freqs, peaks, cfg, options, reference_psd=None):
    del reference_psd  # Combined PSD is intentionally never a candidate source.
    candidates = []
    seen = set()

    def add(coordinate, source, rank, target_frequency, bin_index=None):
        target_frequency = float(target_frequency)
        if target_frequency <= 0.0 or target_frequency > float(freqs[-1]):
            return
        if bin_index is None:
            bin_index = int(np.argmin(np.abs(freqs - target_frequency)))
        else:
            bin_index = int(bin_index)
        key = (coordinate, source, str(rank), bin_index)
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "coordinate": coordinate,
            "source": source,
            "peak_rank": rank,
            "target_frequency": target_frequency,
            "nearest_bin": bin_index,
        })

    if options.get("include_detected_peaks", True):
        for row in peaks:
            if isinstance(row, dict):
                add(
                    str(row["coordinate"]),
                    "automatic_peak",
                    int(row["rank"]),
                    float(row["frequency"]),
                    int(row["bin"]),
                )
            else:
                # Compatibility for callers holding historical global peak
                # arrays.  New formal analyses always pass coordinate rows.
                for coordinate in COORDINATES:
                    add(
                        coordinate,
                        "automatic_peak",
                        int(row[0]),
                        float(row[2]),
                        int(row[1]),
                    )
    for index, frequency in enumerate(
        options.get("predefined_target_frequencies", []), start=1
    ):
        for coordinate in COORDINATES:
            add(
                coordinate,
                "predefined_target",
                f"predefined_{index}",
                float(frequency),
            )
    return candidates


def _fixed_seed_split(n_seed, options):
    n_seed = int(n_seed)
    if n_seed < 2:
        raise ValueError(
            "peak discovery/significance requires at least two seeds for a "
            "disjoint discovery/test split"
        )
    n_discovery = int(round(
        n_seed * float(options["discovery_seed_fraction"])
    ))
    n_discovery = max(1, min(n_seed - 1, n_discovery))
    discovery = np.arange(0, n_discovery, dtype=int)
    test = np.arange(n_discovery, n_seed, dtype=int)
    return {
        "strategy": options["seed_split_strategy"],
        "discovery_seed_fraction": float(options["discovery_seed_fraction"]),
        "discovery_seed_indices": discovery,
        "test_seed_indices": test,
    }


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
    psd_seed = np.asarray(psd_seed, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if psd_seed.ndim != 3 or psd_seed.shape[1:] != (
        len(COORDINATES), freqs.size
    ):
        raise ValueError(
            "test-seed PSD must have shape "
            f"(n_test_seed, {len(COORDINATES)}, {freqs.size})"
        )
    confidence = float(cfg.get("confidence_level", 0.95))
    alpha = float(cfg.get("significance_alpha", 0.05))
    floor = float(options["psd_floor"])
    rows = []
    d_values = []
    row_coordinates = []
    for candidate in candidates:
        target_bin = int(candidate["nearest_bin"])
        window = int(options["peak_window_bins"])
        window_start = max(0, target_bin - window)
        window_stop = min(len(freqs), target_bin + window + 1)
        coordinate = candidate.get("coordinate")
        if coordinate is None:
            coordinate_indices = range(len(COORDINATES))
        else:
            if coordinate not in COORDINATES:
                raise ValueError(f"unknown peak coordinate {coordinate!r}")
            coordinate_indices = (COORDINATES.index(coordinate),)
        for coordinate_index in coordinate_indices:
            coordinate = COORDINATES[coordinate_index]
            # Both automatically discovered peaks and predefined targets are
            # tested at a bin fixed without looking at the test seeds.
            tested_bin = target_bin
            left_start, left_stop, right_start, right_stop = _peak_background_slices(
                freqs,
                tested_bin,
                options,
                window_start=window_start,
                window_stop=window_stop,
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
                "tested_frequency": float(freqs[tested_bin]),
                "tested_omega": 2.0 * math.pi * float(freqs[tested_bin]),
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
    for coordinate in COORDINATES:
        indices = [index for index, value in enumerate(row_coordinates)
                   if value == coordinate]
        adjusted = _holm_adjust([rows[index]["p_value_one_sided"] for index in indices])
        for local_index, row_index in enumerate(indices):
            rows[row_index]["p_value_adjusted"] = float(adjusted[local_index])
            rows[row_index]["significant"] = bool(adjusted[local_index] < alpha)
    if not d_values:
        return rows, np.empty((0, psd_seed.shape[0]), dtype=float)
    return rows, np.asarray(d_values, dtype=float)


def _analyze_saved_spectrum(freqs, psd_seed, psd_mean, cfg):
    """Run discovery and test analysis using only arrays saved by spectrum."""

    freqs = np.asarray(freqs, dtype=float)
    psd_seed = np.asarray(psd_seed, dtype=float)
    psd_mean = np.asarray(psd_mean, dtype=float)
    if psd_seed.ndim != 3 or psd_seed.shape[1:] != (
        len(COORDINATES), freqs.size
    ):
        raise ValueError(
            "saved psd_seed must have shape "
            f"(n_seed, {len(COORDINATES)}, {freqs.size})"
        )
    if psd_mean.shape != psd_seed.shape[1:]:
        raise ValueError("saved psd_mean shape does not match saved psd_seed")

    options = _peak_significance_cfg(cfg)
    seed_split = _fixed_seed_split(psd_seed.shape[0], options)
    discovery_indices = seed_split["discovery_seed_indices"]
    test_indices = seed_split["test_seed_indices"]
    discovery_psd_mean = np.mean(psd_seed[discovery_indices], axis=0)
    detected_peaks = _detect_peaks_by_coordinate(
        freqs, discovery_psd_mean, cfg
    )
    candidates = _candidate_peak_specs(
        freqs, detected_peaks, cfg, options
    )
    significance_rows = []
    significance_values = np.empty((0, test_indices.size), dtype=float)
    if options["enabled"]:
        significance_rows, significance_values = _log_peak_background_ratios(
            psd_seed[test_indices],
            freqs,
            candidates,
            cfg,
            selection_mode="fixed_contiguous_discovery_test_seed_split",
        )
    predefined_rows = [
        row for row in significance_rows
        if row["peak_source"] == "predefined_target"
    ]
    return {
        "detected_peaks_by_coordinate": detected_peaks,
        "discovery_psd_mean": discovery_psd_mean,
        "peak_significance_rows": significance_rows,
        "peak_significance_values": significance_values,
        "predefined_target_tests": predefined_rows,
        "peak_significance_options": options,
        "seed_split": seed_split,
    }


def _write_outputs(run_dir: Path, result: dict, cfg: dict) -> None:
    peak_header = [
        "coordinate", "rank", "frequency", "omega", "bin", "psd_value",
        "prominence", "width",
    ]
    detected_peaks = result.get("detected_peaks_by_coordinate", [])
    peak_rows = [[row[key] for key in peak_header] for row in detected_peaks]
    storage.write_csv(
        run_dir / "tables" / "detected_peaks_by_coordinate.csv",
        peak_header,
        peak_rows,
    )
    # Historical filename retained as an alias of the corrected formal table.
    storage.write_csv(
        run_dir / "tables" / "spectrum_peaks.csv",
        peak_header,
        peak_rows,
    )

    checks = result.get("target_frequency_checks", [])
    check_header = [
        "kind", "omega", "frequency", "frequency_resolution",
        "nyquist_frequency", "resolved", "nearest_fft_bin",
    ]
    storage.write_csv(
        run_dir / "tables" / "frequency_resolution_checks.csv",
        check_header,
        [[row[key] for key in check_header] for row in checks],
    )

    significance_header = [
        "coordinate", "selection_mode", "peak_source", "peak_rank",
        "target_frequency", "tested_frequency", "tested_omega", "nearest_bin",
        "tested_bin", "peak_power_method", "peak_window_bins", "exclude_bins",
        "background_bins_per_side", "background_left_min",
        "background_left_max", "background_right_min",
        "background_right_max", "n_seed", "mean_log_peak_ratio",
        "std_log_peak_ratio", "standard_error", "t_statistic",
        "degrees_of_freedom", "p_value_one_sided", "p_value_adjusted",
        "confidence_interval_low", "confidence_interval_high",
        "significant", "significance_alpha", "confidence_level",
    ]
    significance_rows = result.get("peak_significance_rows", [])
    storage.write_csv(
        run_dir / "tables" / "spectrum_peak_significance.csv",
        significance_header,
        [[row[key] for key in significance_header] for row in significance_rows],
    )
    predefined_rows = result.get("predefined_target_tests", [])
    storage.write_csv(
        run_dir / "tables" / "predefined_target_tests.csv",
        significance_header,
        [[row[key] for key in significance_header] for row in predefined_rows],
    )

    if "sampling_metadata" in result:
        storage.write_json(
            run_dir / "data" / "sampling_metadata.json",
            result["sampling_metadata"],
        )
    if "seed_split" in result:
        storage.write_json(
            run_dir / "data" / "peak_detection_seed_split.json",
            result["seed_split"],
        )

    _write_coordinate_peak_figure(run_dir, result, cfg)
    if significance_rows:
        _write_peak_significance_figure(run_dir, result, cfg)


def _write_coordinate_peak_figure(run_dir: Path, result: dict, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    frequencies = np.asarray(result["freqs"], dtype=float)
    psd_mean = np.asarray(result["psd_mean"], dtype=float)
    use = frequencies > 0.0
    detected = result.get("detected_peaks_by_coordinate", [])
    target_rows = result.get("predefined_target_tests", [])
    for coordinate_index, (coordinate, axis) in enumerate(
        zip(COORDINATES, axes)
    ):
        axis.loglog(
            frequencies[use],
            psd_mean[coordinate_index, use],
            color="0.2",
            linewidth=0.9,
            label=f"{coordinate} mean PSD (all seeds)",
        )
        if "psd_ci_low" in result and "psd_ci_high" in result:
            axis.fill_between(
                frequencies[use],
                np.maximum(result["psd_ci_low"][coordinate_index, use], 1e-300),
                np.maximum(result["psd_ci_high"][coordinate_index, use], 1e-300),
                alpha=0.18,
                color="0.5",
                label=f"{100 * cfg.get('confidence_level', 0.95):.1f}% CI",
            )
        coordinate_peaks = [
            row for row in detected if row["coordinate"] == coordinate
        ]
        if coordinate_peaks:
            bins = np.asarray([row["bin"] for row in coordinate_peaks], dtype=int)
            axis.scatter(
                frequencies[bins],
                psd_mean[coordinate_index, bins],
                marker="o",
                facecolors="none",
                edgecolors="tab:blue",
                s=44,
                linewidths=1.1,
                zorder=4,
                label="automatic peak",
            )
        coordinate_targets = [
            row for row in target_rows if row["coordinate"] == coordinate
        ]
        if coordinate_targets:
            bins = np.asarray(
                [row["tested_bin"] for row in coordinate_targets], dtype=int
            )
            axis.scatter(
                frequencies[bins],
                psd_mean[coordinate_index, bins],
                marker="x",
                color="tab:orange",
                s=52,
                linewidths=1.4,
                zorder=5,
                label="predefined target",
            )
        axis.set_ylabel(f"{coordinate} PSD")
        axis.grid(True, which="both", alpha=0.18)
        axis.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("frequency (Lorenz time unit^-1)")

    metadata = result.get("sampling_metadata", {})
    resolution = metadata.get(
        "frequency_resolution", _frequency_resolution(frequencies)
    )
    discovery_count = len(result["seed_split"]["discovery_seed_indices"])
    test_count = len(result["seed_split"]["test_seed_indices"])
    fig.suptitle(
        "Coordinate-resolved natural spectra\n"
        f"automatic peaks: {discovery_count} discovery seeds; "
        f"significance: {test_count} test seeds; resolution={resolution:.4g}"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "coordinate_psd_peaks.pdf")
    fig.savefig(run_dir / "figures" / "coordinate_psd_peaks.png", dpi=180)
    # Historical names remain aliases of the corrected visualization.
    fig.savefig(run_dir / "figures" / "natural_spectrum.pdf")
    fig.savefig(run_dir / "figures" / "natural_spectrum.png", dpi=180)
    plt.close(fig)


def _write_peak_significance_figure(run_dir: Path, result: dict, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    del cfg
    rows = result["peak_significance_rows"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharey=True)
    for coordinate, axis in zip(COORDINATES, axes):
        coord_rows = [row for row in rows if row["coordinate"] == coordinate]
        x = np.arange(len(coord_rows))
        means = np.asarray([row["mean_log_peak_ratio"] for row in coord_rows])
        low = np.asarray([row["confidence_interval_low"] for row in coord_rows])
        high = np.asarray([row["confidence_interval_high"] for row in coord_rows])
        if coord_rows:
            axis.errorbar(
                x,
                means,
                yerr=np.vstack([means - low, high - means]),
                fmt="none",
                ecolor="0.35",
                capsize=3,
                linewidth=0.8,
            )
        for index, row in enumerate(coord_rows):
            marker = "X" if row["peak_source"] == "predefined_target" else "o"
            color = "tab:green" if row["significant"] else "tab:red"
            axis.scatter(
                [index],
                [row["mean_log_peak_ratio"]],
                marker=marker,
                color=color,
                s=42,
                zorder=3,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(
            x,
            [
                (
                    f"target\n{row['target_frequency']:.3g}"
                    if row["peak_source"] == "predefined_target"
                    else f"auto {row['peak_rank']}\n{row['tested_frequency']:.3g}"
                )
                for row in coord_rows
            ],
            rotation=25,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel(f"{coordinate}: mean $D_r$")
        axis.grid(True, axis="y", alpha=0.2)
    axes[-1].set_xlabel("candidate frequency (Lorenz time unit^-1)")
    fig.suptitle(
        "Test-seed peak/background significance\n"
        "circle: automatic peak; X: predefined target; "
        "green: coordinate-wise Holm significant"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "peak_significance.pdf")
    fig.savefig(run_dir / "figures" / "peak_significance.png", dpi=180)
    # Historical names remain aliases of the corrected figure.
    fig.savefig(run_dir / "figures" / "spectrum_peak_significance.pdf")
    fig.savefig(run_dir / "figures" / "spectrum_peak_significance.png", dpi=180)
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
    analysis = _analyze_saved_spectrum(freqs, psd_seed, psd_mean, cfg)
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
        # Retained only as an auxiliary visualization array.  It is not used
        # for formal peak discovery or candidate construction.
        "combined_psd": combined,
        "peaks": _legacy_z_peak_array(
            analysis["detected_peaks_by_coordinate"],
            sampling_metadata["frequency_resolution"],
        ),
        "legacy_peak_coordinate": "z",
        "sampling_metadata": sampling_metadata,
        "target_frequency_checks": checks,
    }
    result.update(analysis)
    _write_outputs(run_dir, result, cfg)
    return result
