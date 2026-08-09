"""Shared machinery for standalone harmonic-response scans.

The experiment deliberately lives outside the main Lorenz workflow.  It reuses
the repository Lorenz parameters and solver tolerances, but writes all caches,
tables, figures, and intermediate arrays under outputs/harmonic_response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.integrate import solve_ivp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "harmonic_response"
COORDINATES = ("x", "y", "z")
DEFAULT_A_LIST = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
TERMS = ("c1_f", "c2_2f", "c3_f", "c3_3f", "c4_2f", "c4_4f")
HIGHER_TERMS = ("c3_f", "c3_3f", "c4_2f", "c4_4f")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / "matplotlib"))


@dataclass(frozen=True)
class ExperimentConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    method: str = "DOP853"
    rtol: float = 1e-8
    atol: float = 1e-11
    t_spinup: float = 128.0
    sample_rate: float = 32.0
    n_record_samples: int = 4096
    fft_length: int = 4096
    welch_segment_length: int = 2048
    welch_overlap_samples: int = 1024
    n_spectrum_seed: int = 8
    candidate_count: int = 12
    f_min: float = 0.12
    f_max: float = 3.2
    exclusion_half_width: float = 0.06
    peak_prominence_fraction: float = 0.08
    peak_min_distance_frequency: float = 0.08
    alpha: float = 0.05
    a_list: tuple[float, ...] = DEFAULT_A_LIST

    def solver(self) -> dict:
        return {"method": self.method, "rtol": self.rtol, "atol": self.atol}

    def lorenz_cfg(self) -> dict:
        return {
            "lorenz": {"sigma": self.sigma, "rho": self.rho, "beta": self.beta},
            "solver": self.solver(),
            "T_spinup": self.t_spinup,
        }


def parse_common_args(default_n_traj: int, default_run_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--n-traj", type=int, default=default_n_traj)
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--spectrum-run", default="20260731-151340-434542998_spectrum_a93aa73d")
    parser.add_argument("--n-record-samples", type=int, default=4096)
    parser.add_argument("--sample-rate", type=float, default=32.0)
    parser.add_argument("--f-min", type=float, default=0.12)
    parser.add_argument("--f-max", type=float, default=3.2)
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="*",
        help="Optional bin-aligned frequencies to test instead of automatic selection.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    n_record = int(args.n_record_samples)
    if n_record <= 0 or n_record & (n_record - 1):
        raise ValueError("--n-record-samples must be a positive power of two")
    return ExperimentConfig(
        sample_rate=float(args.sample_rate),
        n_record_samples=n_record,
        fft_length=n_record,
        welch_segment_length=max(256, n_record // 2),
        welch_overlap_samples=max(0, n_record // 4),
        candidate_count=int(args.candidate_count),
        f_min=float(args.f_min),
        f_max=float(args.f_max),
    )


def run_directory(run_name: str) -> Path:
    clean = run_name.strip().replace("/", "_")
    if not clean:
        raise ValueError("--run-name cannot be empty")
    path = OUTPUT_ROOT / clean
    for subdir in ("cache", "config", "data", "tables", "figures", "status"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def lorenz_rhs(t: float, state: np.ndarray, cfg: ExperimentConfig, amplitude: float,
               frequency: float) -> list[float]:
    x, y, z = state
    forcing_z = float(amplitude) * math.cos(2.0 * math.pi * float(frequency) * t)
    return [
        cfg.sigma * (y - x),
        cfg.rho * x - y - x * z,
        x * y - cfg.beta * z + forcing_z,
    ]


def spinup_state(seed: int, cfg: ExperimentConfig) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    y0 = rng.random(3)
    sol = solve_ivp(
        lambda t, s: lorenz_rhs(t, s, cfg, 0.0, 0.0),
        [0.0, cfg.t_spinup],
        y0,
        t_eval=[cfg.t_spinup],
        **cfg.solver(),
    )
    if not sol.success or sol.y.shape != (3, 1) or not np.isfinite(sol.y).all():
        raise RuntimeError(f"spinup failed for seed {seed}: {sol.message}")
    return sol.y[:, -1]


def simulate_record(initial_state: np.ndarray, cfg: ExperimentConfig,
                    amplitude: float, frequency: float) -> np.ndarray:
    times = np.arange(cfg.n_record_samples, dtype=float) / cfg.sample_rate
    sol = solve_ivp(
        lambda t, s: lorenz_rhs(t, s, cfg, amplitude, frequency),
        [0.0, float(times[-1])],
        np.asarray(initial_state, dtype=float),
        t_eval=times,
        **cfg.solver(),
    )
    if (
        not sol.success
        or sol.y.shape != (3, cfg.n_record_samples)
        or not np.isfinite(sol.y).all()
    ):
        raise RuntimeError(f"record integration failed: {sol.message}")
    return sol.y


def fft_coefficients(values: np.ndarray, bins: list[int]) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - np.mean(values, axis=-1, keepdims=True)
    fft = np.fft.rfft(centered, axis=-1) / centered.shape[-1]
    coeff = 2.0 * fft[:, np.asarray(bins, dtype=int)]
    return coeff


def natural_spectrum_for_seed(seed: int, cfg: ExperimentConfig) -> tuple[int, np.ndarray, np.ndarray]:
    state = spinup_state(seed, cfg)
    record = simulate_record(state, cfg, 0.0, 0.0)
    freqs, psd = signal.welch(
        record,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=cfg.welch_segment_length,
        noverlap=cfg.welch_overlap_samples,
        nfft=cfg.fft_length,
        detrend="constant",
        axis=-1,
        scaling="density",
    )
    return int(seed), freqs, psd


def natural_spectrum_worker(payload: tuple[int, ExperimentConfig]) -> tuple[int, np.ndarray, np.ndarray]:
    seed, cfg = payload
    return natural_spectrum_for_seed(seed, cfg)


def _run_parallel(items, worker, workers: int):
    if workers <= 1:
        return [worker(item) for item in items]
    results = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def compute_natural_spectrum(cfg: ExperimentConfig, run_dir: Path, workers: int) -> dict:
    cache_path = run_dir / "cache" / "natural_spectrum.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "freqs": data["freqs"],
                "psd_seed": data["psd_seed"],
                "psd_mean": data["psd_mean"],
            }

    rows = _run_parallel(
        [(seed, cfg) for seed in range(cfg.n_spectrum_seed)],
        natural_spectrum_worker,
        workers,
    )
    rows.sort(key=lambda row: row[0])
    freqs = rows[0][1]
    psd_seed = np.asarray([row[2] for row in rows])
    psd_mean = np.mean(psd_seed, axis=0)
    np.savez_compressed(cache_path, freqs=freqs, psd_seed=psd_seed, psd_mean=psd_mean)
    return {"freqs": freqs, "psd_seed": psd_seed, "psd_mean": psd_mean}


def select_candidate_frequencies(natural: dict, cfg: ExperimentConfig) -> dict:
    freqs = np.asarray(natural["freqs"], dtype=float)
    psd_mean = np.asarray(natural["psd_mean"], dtype=float)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    combined = np.mean(
        psd_mean / np.maximum(trapezoid(psd_mean, freqs, axis=-1)[:, None], 1e-300),
        axis=0,
    )
    df = cfg.sample_rate / cfg.fft_length
    valid = (
        (freqs >= cfg.f_min)
        & (freqs <= cfg.f_max)
        & (4.0 * freqs < 0.98 * cfg.sample_rate / 2.0)
    )
    work = combined.copy()
    work[~valid] = 0.0
    peaks, props = signal.find_peaks(
        work,
        prominence=max(np.max(work), 1e-300) * cfg.peak_prominence_fraction,
        distance=max(1, int(math.ceil(cfg.peak_min_distance_frequency / df))),
    )
    excluded = np.zeros_like(freqs, dtype=bool)
    for peak in peaks:
        excluded |= np.abs(freqs - freqs[int(peak)]) <= cfg.exclusion_half_width
    candidate_bins = np.flatnonzero(valid & ~excluded)
    if candidate_bins.size == 0:
        raise RuntimeError("natural-peak exclusion removed all candidate bins")
    order = np.argsort(combined[candidate_bins])
    spread = np.linspace(0, max(0, len(order) - 1), cfg.candidate_count, dtype=int)
    selected_bins = sorted(set(int(candidate_bins[order[index]]) for index in spread))
    return {
        "combined_psd": combined,
        "natural_peak_bins": np.asarray(peaks, dtype=int),
        "excluded_mask": excluded,
        "candidate_bins": np.asarray(selected_bins, dtype=int),
        "candidate_frequencies": freqs[np.asarray(selected_bins, dtype=int)],
    }


def selection_with_explicit_frequencies(natural: dict, cfg: ExperimentConfig,
                                        frequencies: list[float]) -> dict:
    base = select_candidate_frequencies(natural, cfg)
    freqs = np.asarray(natural["freqs"], dtype=float)
    df = cfg.sample_rate / cfg.fft_length
    selected_bins = []
    for frequency in frequencies:
        bin_index = int(round(float(frequency) / df))
        if bin_index <= 0 or bin_index >= freqs.size:
            raise ValueError(f"frequency outside FFT grid: {frequency}")
        aligned = float(freqs[bin_index])
        if not math.isclose(aligned, float(frequency), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"frequency {frequency} is not FFT-bin aligned; nearest bin is {aligned}"
            )
        if 4.0 * aligned >= 0.98 * cfg.sample_rate / 2.0:
            raise ValueError(f"4f is too close to Nyquist for frequency {frequency}")
        selected_bins.append(bin_index)
    return {
        **base,
        "candidate_bins": np.asarray(selected_bins, dtype=int),
        "candidate_frequencies": freqs[np.asarray(selected_bins, dtype=int)],
    }


def response_seed_worker(payload: tuple[int, float, ExperimentConfig, int]) -> dict:
    seed, frequency, cfg, seed_offset = payload
    actual_seed = int(seed) + int(seed_offset)
    state = spinup_state(actual_seed, cfg)
    bins = [int(round(k * frequency / (cfg.sample_rate / cfg.n_record_samples))) for k in (1, 2, 3, 4)]
    zero = simulate_record(state, cfg, 0.0, frequency)
    zero_coeff = fft_coefficients(zero, bins)
    plus_coeff = []
    minus_coeff = []
    for amplitude in cfg.a_list:
        plus = simulate_record(state, cfg, float(amplitude), frequency)
        minus = simulate_record(state, cfg, -float(amplitude), frequency)
        plus_coeff.append(fft_coefficients(plus, bins))
        minus_coeff.append(fft_coefficients(minus, bins))
    plus_coeff = np.asarray(plus_coeff)
    minus_coeff = np.asarray(minus_coeff)
    odd = (plus_coeff - minus_coeff) / 2.0
    even = (plus_coeff + minus_coeff - 2.0 * zero_coeff[None, :, :]) / 2.0
    return {
        "seed": int(seed),
        "actual_seed": int(actual_seed),
        "frequency": float(frequency),
        "bins": np.asarray(bins, dtype=int),
        "zero_coeff": zero_coeff,
        "odd": odd,
        "even": even,
    }


def load_or_compute_response(frequency: float, cfg: ExperimentConfig, n_traj: int,
                             run_dir: Path, workers: int, seed_offset: int = 0) -> dict:
    key_payload = {
        "frequency": float(frequency),
        "cfg": asdict(cfg),
        "n_traj": int(n_traj),
        "seed_offset": int(seed_offset),
        "version": 2,
    }
    key = _cache_key(key_payload)
    cache_path = run_dir / "cache" / f"response_{key}.npz"
    if not cache_path.exists() and int(seed_offset) == 0:
        legacy_key = _cache_key({
            "frequency": float(frequency),
            "cfg": asdict(cfg),
            "n_traj": int(n_traj),
            "version": 1,
        })
        legacy_path = run_dir / "cache" / f"response_{legacy_key}.npz"
        if legacy_path.exists():
            cache_path = legacy_path
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "frequency": float(data["frequency"]),
                "seed": data["seed"],
                "actual_seed": data["actual_seed"] if "actual_seed" in data else data["seed"],
                "bins": data["bins"],
                "zero_coeff": data["zero_coeff"],
                "odd": data["odd"],
                "even": data["even"],
            }
    payloads = [(seed, float(frequency), cfg, int(seed_offset)) for seed in range(int(n_traj))]
    rows = _run_parallel(payloads, response_seed_worker, workers)
    rows.sort(key=lambda row: row["seed"])
    result = {
        "frequency": float(frequency),
        "seed": np.asarray([row["seed"] for row in rows], dtype=int),
        "actual_seed": np.asarray([row["actual_seed"] for row in rows], dtype=int),
        "bins": rows[0]["bins"],
        "zero_coeff": np.asarray([row["zero_coeff"] for row in rows]),
        "odd": np.asarray([row["odd"] for row in rows]),
        "even": np.asarray([row["even"] for row in rows]),
    }
    np.savez_compressed(cache_path, **result)
    return result


def _fit_complex(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeff


def fit_seed_coefficients(response: dict, cfg: ExperimentConfig) -> dict:
    amplitudes = np.asarray(cfg.a_list, dtype=float)
    design_odd = np.column_stack([amplitudes, amplitudes ** 3])
    design_even = np.column_stack([amplitudes ** 2, amplitudes ** 4])
    design_c3 = (amplitudes ** 3)[:, None]
    design_c4 = (amplitudes ** 4)[:, None]
    n_seed = response["odd"].shape[0]
    coeffs = {
        "c1_f": np.zeros((n_seed, 3), dtype=complex),
        "c3_f": np.zeros((n_seed, 3), dtype=complex),
        "c2_2f": np.zeros((n_seed, 3), dtype=complex),
        "c4_2f": np.zeros((n_seed, 3), dtype=complex),
        "c3_3f": np.zeros((n_seed, 3), dtype=complex),
        "c4_4f": np.zeros((n_seed, 3), dtype=complex),
    }
    for seed_index in range(n_seed):
        for output in range(3):
            c_odd_f = _fit_complex(response["odd"][seed_index, :, output, 0], design_odd)
            c_even_2f = _fit_complex(response["even"][seed_index, :, output, 1], design_even)
            c_odd_3f = _fit_complex(response["odd"][seed_index, :, output, 2], design_c3)
            c_even_4f = _fit_complex(response["even"][seed_index, :, output, 3], design_c4)
            coeffs["c1_f"][seed_index, output] = c_odd_f[0]
            coeffs["c3_f"][seed_index, output] = c_odd_f[1]
            coeffs["c2_2f"][seed_index, output] = c_even_2f[0]
            coeffs["c4_2f"][seed_index, output] = c_even_2f[1]
            coeffs["c3_3f"][seed_index, output] = c_odd_3f[0]
            coeffs["c4_4f"][seed_index, output] = c_even_4f[0]
    return coeffs


def hotelling_p_value(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=complex)
    x = np.column_stack([values.real, values.imag])
    n, p = x.shape
    mean = np.mean(x, axis=0)
    if n <= p:
        return float("nan"), float("nan"), n
    cov = np.cov(x, rowvar=False)
    cov = np.asarray(cov, dtype=float) + np.eye(p) * 1e-18
    t2 = float(n * mean @ np.linalg.pinv(cov) @ mean)
    f_stat = (n - p) * t2 / (p * (n - 1))
    p_value = float(stats.f.sf(f_stat, p, n - p))
    return t2, p_value, n


def complex_mean_and_se(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=complex)
    n = int(values.size)
    mean = complex(np.mean(values))
    if n > 1:
        real_se = float(stats.sem(values.real))
        imag_se = float(stats.sem(values.imag))
    else:
        real_se = float("nan")
        imag_se = float("nan")
    return {
        "mean_real": float(mean.real),
        "mean_imag": float(mean.imag),
        "mean_abs": float(abs(mean)),
        "se_real": real_se,
        "se_imag": imag_se,
    }


def _distance_to_formal_natural_peak(frequency: float, formal_peaks: list[dict] | None) -> dict:
    if not formal_peaks:
        return {
            "formal_natural_reference": "missing",
            "nearest_formal_peak_frequency": float("nan"),
            "nearest_formal_peak_coordinate": "",
            "distance_to_nearest_formal_peak": float("nan"),
            "inside_formal_exclusion_band": "",
        }
    nearest = min(formal_peaks, key=lambda row: abs(float(row["frequency"]) - float(frequency)))
    distance = abs(float(nearest["frequency"]) - float(frequency))
    half_width = float(nearest.get("exclusion_half_width", 0.06))
    return {
        "formal_natural_reference": "formal_spectrum",
        "nearest_formal_peak_frequency": float(nearest["frequency"]),
        "nearest_formal_peak_coordinate": nearest["coordinate"],
        "distance_to_nearest_formal_peak": float(distance),
        "inside_formal_exclusion_band": bool(distance <= half_width),
    }


def load_formal_natural_peaks(
    spectrum_run: str | None = "20260731-151340-434542998_spectrum_a93aa73d",
    exclusion_half_width: float = 0.06,
) -> list[dict]:
    """Load significant automatic natural peaks from the formal spectrum result.

    The standalone harmonic scan's n_spectrum_seed is intentionally too small to
    be treated as a final natural-peak reference.  This helper prefers the
    high-seed repository spectrum table and returns an empty list if it is absent.
    """
    if not spectrum_run:
        return []
    path = PROJECT_ROOT / "results" / "runs" / spectrum_run / "tables" / "spectrum_peak_significance.csv"
    if not path.exists():
        return []
    peaks = []
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("peak_source") != "automatic_peak":
                continue
            if str(row.get("significant", "")).lower() != "true":
                continue
            peaks.append({
                "coordinate": row["coordinate"],
                "frequency": float(row["tested_frequency"]),
                "omega": float(row["tested_omega"]),
                "peak_rank": row["peak_rank"],
                "n_seed": int(row["n_seed"]),
                "p_value_adjusted": float(row["p_value_adjusted"]),
                "exclusion_half_width": float(exclusion_half_width),
            })
    return peaks


def analyze_frequency(
    response: dict,
    cfg: ExperimentConfig,
    formal_peaks: list[dict] | None = None,
) -> list[dict]:
    coeffs = fit_seed_coefficients(response, cfg)
    rows = []
    a_ref = float(max(cfg.a_list))
    for output_index, coordinate in enumerate(COORDINATES):
        stats_by_term = {}
        for term, values in coeffs.items():
            samples = values[:, output_index]
            t2, p_value, n = hotelling_p_value(samples)
            summary = complex_mean_and_se(samples)
            stats_by_term[term] = {
                "t2": t2,
                "p_value": p_value,
                "n": n,
                "significant": bool(np.isfinite(p_value) and p_value < cfg.alpha),
                **summary,
            }
        first = stats_by_term["c1_f"]["significant"]
        second = stats_by_term["c2_2f"]["significant"]
        higher_clear = all(not stats_by_term[term]["significant"] for term in HIGHER_TERMS)
        contributions = {
            "R1_c1_f": stats_by_term["c1_f"]["mean_abs"] * a_ref,
            "R2_c2_2f": stats_by_term["c2_2f"]["mean_abs"] * a_ref ** 2,
            "R3_c3_f": stats_by_term["c3_f"]["mean_abs"] * a_ref ** 3,
            "R3_c3_3f": stats_by_term["c3_3f"]["mean_abs"] * a_ref ** 3,
            "R4_c4_2f": stats_by_term["c4_2f"]["mean_abs"] * a_ref ** 4,
            "R4_c4_4f": stats_by_term["c4_4f"]["mean_abs"] * a_ref ** 4,
        }
        low_sum = contributions["R1_c1_f"] + contributions["R2_c2_2f"]
        high_sum = (
            contributions["R3_c3_f"] + contributions["R3_c3_3f"]
            + contributions["R4_c4_2f"] + contributions["R4_c4_4f"]
        )
        p_values = {term: stats_by_term[term]["p_value"] for term in TERMS}
        low_order_p_max = max(p_values["c1_f"], p_values["c2_2f"])
        higher_p_min = min(p_values[term] for term in HIGHER_TERMS)
        rows.append({
            "frequency": float(response["frequency"]),
            "omega": 2.0 * math.pi * float(response["frequency"]),
            "coordinate": coordinate,
            "n_seed": int(response["seed"].size),
            "seed_min": int(np.min(response.get("actual_seed", response["seed"]))),
            "seed_max": int(np.max(response.get("actual_seed", response["seed"]))),
            "A_ref": a_ref,
            "pass_candidate": bool(first and second and higher_clear),
            "failure_reason": _failure_reason(stats_by_term, cfg.alpha),
            "low_order_p_max": float(low_order_p_max),
            "higher_p_min": float(higher_p_min),
            "higher_significant_count": int(sum(p_values[term] < cfg.alpha for term in HIGHER_TERMS)),
            "lower_contribution_sum": float(low_sum),
            "higher_contribution_sum": float(high_sum),
            "higher_ratio": float(high_sum / max(low_sum, 1e-300)),
            **_distance_to_formal_natural_peak(float(response["frequency"]), formal_peaks),
            **{f"{term}_p": p_values[term] for term in TERMS},
            **{f"{term}_mean_real": stats_by_term[term]["mean_real"] for term in TERMS},
            **{f"{term}_mean_imag": stats_by_term[term]["mean_imag"] for term in TERMS},
            **{f"{term}_mean_abs": stats_by_term[term]["mean_abs"] for term in TERMS},
            **contributions,
        })
    return rows


def _failure_reason(stats_by_term: dict, alpha: float) -> str:
    failures = []
    if not stats_by_term["c1_f"]["p_value"] < alpha:
        failures.append("c1_f_not_significant")
    if not stats_by_term["c2_2f"]["p_value"] < alpha:
        failures.append("c2_2f_not_significant")
    for term in ("c3_f", "c3_3f", "c4_2f", "c4_4f"):
        if stats_by_term[term]["p_value"] < alpha:
            failures.append(f"{term}_significant")
    return "pass" if not failures else ";".join(failures)


def response_score(row: dict) -> float:
    low = max(row["c1_f_p"], row["c2_2f_p"])
    high = min(row["c3_f_p"], row["c3_3f_p"], row["c4_2f_p"], row["c4_4f_p"])
    low = max(low, 1e-300)
    high = max(high, 1e-300)
    return float(-math.log10(low) + math.log10(high))


def candidate_sort_key(row: dict) -> tuple:
    inside = row.get("inside_formal_exclusion_band")
    inside_penalty = 1 if inside is True else 0
    return (
        inside_penalty,
        int(row.get("higher_significant_count", 0)),
        float(row.get("low_order_p_max", max(row["c1_f_p"], row["c2_2f_p"]))),
        float(row.get("higher_ratio", float("inf"))),
        -float(row.get("higher_p_min", min(row["c3_f_p"], row["c3_3f_p"], row["c4_2f_p"], row["c4_4f_p"]))),
    )


def select_validation_frequencies(rows: list[dict], max_count: int = 4) -> list[float]:
    ordered = sorted(rows, key=candidate_sort_key)
    selected = []
    for row in ordered:
        frequency = float(row["frequency"])
        if frequency not in selected:
            selected.append(frequency)
        if len(selected) >= max_count:
            break
    return selected


def write_analysis_tables(run_dir: Path, cfg: ExperimentConfig, natural: dict,
                          selection: dict, rows: list[dict]) -> None:
    write_csv(
        run_dir / "tables" / "scanned_frequencies.csv",
        ["frequency", "omega", "bin"],
        [
            [float(f), 2.0 * math.pi * float(f), int(b)]
            for f, b in zip(selection["candidate_frequencies"], selection["candidate_bins"])
        ],
    )
    peak_header = ["coordinate", "frequency", "omega", "bin", "combined_psd"]
    peak_rows = []
    freqs = natural["freqs"]
    for peak_bin in selection["natural_peak_bins"]:
        for coordinate in COORDINATES:
            peak_rows.append([
                coordinate,
                float(freqs[int(peak_bin)]),
                2.0 * math.pi * float(freqs[int(peak_bin)]),
                int(peak_bin),
                float(selection["combined_psd"][int(peak_bin)]),
            ])
    write_csv(run_dir / "tables" / "natural_peaks.csv", peak_header, peak_rows)
    header = [
        "rank", "frequency", "omega", "coordinate", "n_seed", "seed_min", "seed_max",
        "A_ref", "pass_candidate", "failure_reason", "low_order_p_max",
        "higher_p_min", "higher_significant_count", "lower_contribution_sum",
        "higher_contribution_sum", "higher_ratio", "formal_natural_reference",
        "nearest_formal_peak_frequency", "nearest_formal_peak_coordinate",
        "distance_to_nearest_formal_peak", "inside_formal_exclusion_band",
        "c1_f_p", "c2_2f_p", "c3_f_p", "c3_3f_p", "c4_2f_p", "c4_4f_p",
        "c1_f_mean_real", "c1_f_mean_imag", "c1_f_mean_abs",
        "c2_2f_mean_real", "c2_2f_mean_imag", "c2_2f_mean_abs",
        "c3_f_mean_real", "c3_f_mean_imag", "c3_f_mean_abs",
        "c3_3f_mean_real", "c3_3f_mean_imag", "c3_3f_mean_abs",
        "c4_2f_mean_real", "c4_2f_mean_imag", "c4_2f_mean_abs",
        "c4_4f_mean_real", "c4_4f_mean_imag", "c4_4f_mean_abs",
        "R1_c1_f", "R2_c2_2f", "R3_c3_f", "R3_c3_3f", "R4_c4_2f", "R4_c4_4f",
        "deprecated_score",
    ]
    ranked_all = sorted(rows, key=candidate_sort_key)
    rank_by_key = {
        (row["frequency"], row["coordinate"]): rank
        for rank, row in enumerate(ranked_all, start=1)
    }
    def row_values(row: dict) -> list:
        enriched = {
            **row,
            "rank": rank_by_key[(row["frequency"], row["coordinate"])],
            "deprecated_score": response_score(row),
        }
        return [enriched.get(col, "") for col in header]
    write_csv(
        run_dir / "tables" / "candidate_statistics.csv",
        header,
        [row_values(row) for row in ranked_all],
    )
    selected_freqs = select_validation_frequencies(rows, max_count=4) if rows else []
    ranked = [row for row in ranked_all if float(row["frequency"]) in selected_freqs]
    write_csv(
        run_dir / "tables" / "selected_candidates.csv",
        header,
        [row_values(row) for row in ranked],
    )


def make_figures(run_dir: Path, cfg: ExperimentConfig, natural: dict, selection: dict,
                 responses: list[dict], rows: list[dict]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig_paths = {}
    freqs = natural["freqs"]
    final_frequencies = select_validation_frequencies(rows, max_count=4) if rows else []

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    excluded = selection["excluded_mask"]
    for output, axis in enumerate(axes):
        axis.semilogy(freqs, natural["psd_mean"][output], color="0.25", linewidth=0.9)
        for peak_bin in selection["natural_peak_bins"]:
            axis.axvline(freqs[int(peak_bin)], color="tab:red", alpha=0.35, linewidth=0.8)
        axis.fill_between(freqs, axis.get_ylim()[0], axis.get_ylim()[1],
                          where=excluded, color="tab:red", alpha=0.08)
        for frequency in selection["candidate_frequencies"]:
            axis.axvline(float(frequency), color="tab:blue", alpha=0.25, linewidth=0.8)
        for frequency in final_frequencies:
            axis.axvline(float(frequency), color="tab:green", linewidth=1.5)
        axis.set_ylabel(f"{COORDINATES[output]} PSD")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("frequency (Lorenz time unit^-1)")
    fig.suptitle("Unforced natural spectrum, excluded peaks, and tested frequencies")
    fig.tight_layout()
    path = run_dir / "figures" / "natural_spectrum_candidates.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    fig_paths["natural_spectrum"] = str(path)

    by_frequency = {float(response["frequency"]): response for response in responses}
    row_by_key = {(row["frequency"], row["coordinate"]): row for row in rows}
    for frequency, response in by_frequency.items():
        _plot_fft_markers(run_dir, cfg, response)
        _plot_response_scaling(run_dir, cfg, response)
        _plot_high_order(run_dir, cfg, response, row_by_key)

    _plot_summary(run_dir, cfg, natural, selection, rows, by_frequency)
    fig_paths["summary"] = str(run_dir / "figures" / "summary.png")
    return fig_paths


def _plot_fft_markers(run_dir: Path, cfg: ExperimentConfig, response: dict) -> None:
    import matplotlib.pyplot as plt

    frequency = float(response["frequency"])
    bins = response["bins"]
    freq_axis = np.fft.rfftfreq(cfg.n_record_samples, d=1.0 / cfg.sample_rate)
    mean_amp = np.mean(np.abs(response["zero_coeff"]), axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for output, axis in enumerate(axes):
        axis.plot(freq_axis[bins], mean_amp[output], marker="o", linestyle="none")
        for harmonic, bin_index in enumerate(bins, start=1):
            axis.axvline(freq_axis[int(bin_index)], color="tab:orange", linewidth=0.9)
            axis.text(freq_axis[int(bin_index)], axis.get_ylim()[1], f"{harmonic}f",
                      va="top", ha="center", fontsize=8)
        axis.set_ylabel(f"{COORDINATES[output]} |FFT|")
    axes[-1].set_xlabel("frequency (Lorenz time unit^-1)")
    fig.suptitle(f"FFT harmonic bins for f={frequency:.6g}")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / f"fft_markers_f_{frequency:.6f}.png", dpi=160)
    plt.close(fig)


def _plot_response_scaling(run_dir: Path, cfg: ExperimentConfig, response: dict) -> None:
    import matplotlib.pyplot as plt

    amplitudes = np.asarray(cfg.a_list)
    a2 = amplitudes ** 2
    frequency = float(response["frequency"])
    odd_f = response["odd"][:, :, :, 0] / amplitudes[None, :, None]
    even_2f = response["even"][:, :, :, 1] / (amplitudes[None, :, None] ** 2)
    for name, values, ylabel in (
        ("first_order", odd_f, "O_f(A) / A"),
        ("second_order", even_2f, "E_2f(A) / A^2"),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        for output in range(3):
            samples = values[:, :, output]
            mean = np.mean(samples, axis=0)
            real_se = stats.sem(samples.real, axis=0)
            imag_se = stats.sem(samples.imag, axis=0)
            axes[output, 0].errorbar(a2, mean.real, yerr=real_se, marker="o", capsize=3)
            axes[output, 0].set_ylabel(f"{COORDINATES[output]} Re")
            axes[output, 0].grid(True, alpha=0.2)
            axes[output, 1].errorbar(a2, mean.imag, yerr=imag_se, marker="o", capsize=3)
            axes[output, 1].set_ylabel(f"{COORDINATES[output]} Im")
            axes[output, 1].grid(True, alpha=0.2)
        axes[-1, 0].set_xlabel("A^2")
        axes[-1, 1].set_xlabel("A^2")
        fig.suptitle(f"Coherent ensemble mean {ylabel} vs A^2, f={frequency:.6g}")
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / f"{name}_scaling_f_{frequency:.6f}.png", dpi=160)
        plt.close(fig)


def _plot_high_order(run_dir: Path, cfg: ExperimentConfig, response: dict,
                     row_by_key: dict) -> None:
    import matplotlib.pyplot as plt

    amplitudes = np.asarray(cfg.a_list)
    frequency = float(response["frequency"])
    odd_3f = response["odd"][:, :, :, 2]
    even_4f = response["even"][:, :, :, 3]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for output, axis in enumerate(axes):
        mean_3 = np.mean(odd_3f[:, :, output], axis=0)
        mean_4 = np.mean(even_4f[:, :, output], axis=0)
        se_3 = stats.sem(np.abs(odd_3f[:, :, output] - mean_3[None, :]), axis=0)
        se_4 = stats.sem(np.abs(even_4f[:, :, output] - mean_4[None, :]), axis=0)
        axis.errorbar(amplitudes, np.abs(mean_3), yerr=se_3, marker="o", capsize=3,
                      label="|coherent mean 3f odd|")
        axis.errorbar(amplitudes, np.abs(mean_4), yerr=se_4, marker="s", capsize=3,
                      label="|coherent mean 4f even|")
        axis.set_yscale("log")
        row = row_by_key.get((frequency, COORDINATES[output]))
        if row:
            axis.set_title(
                f"{COORDINATES[output]}: p(c3_3f)={row['c3_3f_p']:.3g}, "
                f"p(c4_4f)={row['c4_4f_p']:.3g}"
            )
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("A")
    fig.suptitle(f"High-order harmonic diagnostics, f={frequency:.6g}")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / f"high_order_f_{frequency:.6f}.png", dpi=160)
    plt.close(fig)


def _plot_summary(run_dir: Path, cfg: ExperimentConfig, natural: dict, selection: dict,
                  rows: list[dict], by_frequency: dict[float, dict]) -> None:
    import matplotlib.pyplot as plt

    best = sorted(rows, key=candidate_sort_key)[0]
    frequency = float(best["frequency"])
    response = by_frequency[frequency]
    amplitudes = np.asarray(cfg.a_list)
    a2 = amplitudes ** 2
    output = COORDINATES.index(best["coordinate"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].semilogy(natural["freqs"], natural["psd_mean"][output], color="0.25")
    axes[0, 0].axvline(frequency, color="tab:green", linewidth=1.5)
    axes[0, 0].set_title(f"natural PSD ({best['coordinate']})")
    axes[0, 0].set_xlabel("frequency (Lorenz time unit^-1)")
    axes[0, 0].set_ylabel("PSD")

    first = response["odd"][:, :, output, 0] / amplitudes[None, :]
    second = response["even"][:, :, output, 1] / (amplitudes[None, :] ** 2)
    third = response["odd"][:, :, output, 2]
    fourth = response["even"][:, :, output, 3]
    first_mean = np.mean(first, axis=0)
    second_mean = np.mean(second, axis=0)
    axes[0, 1].errorbar(a2, first_mean.real, yerr=stats.sem(first.real, axis=0),
                        marker="o", capsize=3, label="Re")
    axes[0, 1].errorbar(a2, first_mean.imag, yerr=stats.sem(first.imag, axis=0),
                        marker="s", capsize=3, label="Im")
    axes[0, 1].set_title("coherent first-order diagnostic")
    axes[0, 1].set_xlabel("A^2")
    axes[0, 1].set_ylabel("O_f(A) / A")
    axes[1, 0].errorbar(a2, second_mean.real, yerr=stats.sem(second.real, axis=0),
                        marker="o", capsize=3, label="Re")
    axes[1, 0].errorbar(a2, second_mean.imag, yerr=stats.sem(second.imag, axis=0),
                        marker="s", capsize=3, label="Im")
    axes[1, 0].set_title("coherent second-order diagnostic")
    axes[1, 0].set_xlabel("A^2")
    axes[1, 0].set_ylabel("E_2f(A) / A^2")
    axes[1, 1].plot(amplitudes, np.abs(np.mean(third, axis=0)), marker="o", label="|mean 3f|")
    axes[1, 1].plot(amplitudes, np.abs(np.mean(fourth, axis=0)), marker="s", label="|mean 4f|")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(
        f"higher-order p: {best['c3_3f_p']:.3g}, {best['c4_4f_p']:.3g}"
    )
    axes[1, 1].set_xlabel("A")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(True, alpha=0.2)
    fig.suptitle(f"Summary for f={frequency:.6g}, output={best['coordinate']}")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "summary.png", dpi=170)
    plt.close(fig)


def execute_experiment(args: argparse.Namespace, detailed: bool) -> dict:
    started = time.time()
    cfg = make_config(args)
    run_dir = run_directory(args.run_name)
    config_payload = {
        "mode": "detailed" if detailed else "scan",
        "n_traj": int(args.n_traj),
        "workers": int(args.workers),
        "seed_offset": int(args.seed_offset),
        **asdict(cfg),
    }
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "config" / "config.json", config_payload)
    natural = compute_natural_spectrum(cfg, run_dir, int(args.workers))
    formal_peaks = load_formal_natural_peaks(args.spectrum_run, cfg.exclusion_half_width)
    if args.frequencies:
        selection = selection_with_explicit_frequencies(natural, cfg, args.frequencies)
    else:
        selection = select_candidate_frequencies(natural, cfg)
    responses = []
    rows = []
    for frequency in selection["candidate_frequencies"]:
        response = load_or_compute_response(float(frequency), cfg, int(args.n_traj),
                                            run_dir, int(args.workers), int(args.seed_offset))
        responses.append(response)
        rows.extend(analyze_frequency(response, cfg, formal_peaks))
    write_analysis_tables(run_dir, cfg, natural, selection, rows)
    figures = make_figures(run_dir, cfg, natural, selection, responses, rows)
    status = {
        "run_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
        "scanned_frequencies": [float(value) for value in selection["candidate_frequencies"]],
        "formal_natural_spectrum_run": args.spectrum_run if formal_peaks else None,
        "formal_natural_peak_count": len(formal_peaks),
        "candidate_rows": [row for row in rows if row["pass_candidate"]],
        "recommended_validation_frequencies": select_validation_frequencies(rows, max_count=4),
        "best_rows": sorted(rows, key=candidate_sort_key)[:6],
        "figures": figures,
    }
    write_json(run_dir / "status.json", status)
    write_json(run_dir / "status" / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return status
