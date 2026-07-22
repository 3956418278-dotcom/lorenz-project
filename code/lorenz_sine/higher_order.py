"""Explicit cubic, quartic, and total higher-order contamination diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import storage
from .fourier import amplitude_phase, fourier_coefficients
from .statistics import mean_se_ci, phase_l2, safe_ratio


def _axis_raw(amplitude_result):
    raw = np.asarray(amplitude_result["phase_means"], dtype=float)
    combinations = {
        item["name"]: index
        for index, item in enumerate(amplitude_result["combinations"])
    }
    zero = raw[:, :, combinations["zero"]]
    positive = np.stack([
        raw[:, :, combinations[f"axis_{direction}_pos"]]
        for direction in range(3)
    ], axis=4)
    negative = np.stack([
        raw[:, :, combinations[f"axis_{direction}_neg"]]
        for direction in range(3)
    ], axis=4)
    # positive/negative: omega, amplitude, seed, output, direction, phase
    return zero[:, :, :, :, None, :], positive, negative


def _slope_along_amplitude(values, amplitudes):
    values = np.maximum(np.asarray(values, dtype=float), 1e-300)
    denominator = np.diff(np.log(np.asarray(amplitudes, dtype=float)))
    reshape = (1, denominator.size) + (1,) * (values.ndim - 2)
    return np.diff(np.log(values), axis=1) / denominator.reshape(reshape)


def _signal_statistics(signals, amplitudes, confidence, bootstrap_samples,
                       random_seed):
    result = {}
    first = next(iter(signals.values()))
    n_seed = first.shape[2]
    rng = np.random.default_rng(int(random_seed))
    bootstrap_indices = [
        rng.integers(0, n_seed, n_seed)
        for _ in range(int(bootstrap_samples))
    ]
    alpha = 100.0 * (1.0 - float(confidence)) / 2.0
    for name, values in signals.items():
        norm_seed = phase_l2(values, axis=-1)
        mean, se, low, high = mean_se_ci(norm_seed, confidence, axis=2)
        slopes_seed = _slope_along_amplitude(norm_seed, amplitudes)
        slope_mean, slope_se, slope_low, slope_high = mean_se_ci(
            slopes_seed, confidence, axis=2)
        bootstrap_slopes = np.asarray([
            slopes_seed[:, :, indices].mean(axis=2)
            for indices in bootstrap_indices
        ])
        bootstrap_low = np.percentile(bootstrap_slopes, alpha, axis=0)
        bootstrap_high = np.percentile(
            bootstrap_slopes, 100.0 - alpha, axis=0)
        overall_seed = np.sqrt(np.mean(values ** 2, axis=(3, 4, 5)))
        overall_mean, overall_se, overall_low, overall_high = mean_se_ci(
            overall_seed, confidence, axis=2)
        result[name] = {
            "phase_l2_seed": norm_seed,
            "phase_l2_mean": mean,
            "phase_l2_se": se,
            "phase_l2_ci_low": low,
            "phase_l2_ci_high": high,
            "local_slope_seed": slopes_seed,
            "local_slope_mean": slope_mean,
            "local_slope_se": slope_se,
            "local_slope_ci_low": slope_low,
            "local_slope_ci_high": slope_high,
            "local_slope_bootstrap_ci_low": bootstrap_low,
            "local_slope_bootstrap_ci_high": bootstrap_high,
            "overall_seed": overall_seed,
            "overall_mean": overall_mean,
            "overall_se": overall_se,
            "overall_ci_low": overall_low,
            "overall_ci_high": overall_high,
        }
    return result


def _ratio_statistics(numerator, denominator, confidence):
    numerator_norm = phase_l2(numerator, axis=-1)
    denominator_norm = phase_l2(denominator, axis=-1)
    seed = safe_ratio(numerator_norm, denominator_norm)
    mean, se, low, high = mean_se_ci(seed, confidence, axis=2)
    return {
        "seed": seed,
        "mean": mean,
        "se": se,
        "ci_low": low,
        "ci_high": high,
    }


def _harmonic_statistics(values, kmax, confidence):
    coefficients = fourier_coefficients(values, kmax)
    amplitude = amplitude_phase(coefficients)[0]
    mean, se, low, high = mean_se_ci(amplitude, confidence, axis=2)
    return {
        "coefficient_seed": coefficients,
        "amplitude_seed": amplitude,
        "amplitude_mean": mean,
        "amplitude_se": se,
        "amplitude_ci_low": low,
        "amplitude_ci_high": high,
    }


def _write_tables(run_dir, result):
    rows = []
    slope_rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for amplitude_index, amplitude in enumerate(result["amplitudes"]):
            for output in range(3):
                for direction in range(3):
                    for name, diagnostics in result["signals"].items():
                        rows.append([
                            omega, amplitude, output, direction, name,
                            diagnostics["phase_l2_mean"][omega_index, amplitude_index,
                                                          output, direction],
                            diagnostics["phase_l2_se"][omega_index, amplitude_index,
                                                        output, direction],
                            diagnostics["phase_l2_ci_low"][omega_index, amplitude_index,
                                                            output, direction],
                            diagnostics["phase_l2_ci_high"][omega_index, amplitude_index,
                                                             output, direction],
                        ])
        for interval in range(len(result["amplitudes"]) - 1):
            for output in range(3):
                for direction in range(3):
                    for name, diagnostics in result["signals"].items():
                        slope_rows.append([
                            omega,
                            result["amplitudes"][interval],
                            result["amplitudes"][interval + 1],
                            output, direction, name,
                            diagnostics["local_slope_mean"][omega_index, interval,
                                                            output, direction],
                            diagnostics["local_slope_se"][omega_index, interval,
                                                          output, direction],
                            diagnostics["local_slope_ci_low"][omega_index, interval,
                                                              output, direction],
                            diagnostics["local_slope_ci_high"][omega_index, interval,
                                                               output, direction],
                        ])
    storage.write_csv(
        run_dir / "tables" / "higher_order_scaling.csv",
        ["omega", "amplitude", "output", "forcing_direction", "signal",
         "phase_l2_mean", "se", "ci_low", "ci_high"],
        rows,
    )
    storage.write_csv(
        run_dir / "tables" / "local_power_slopes.csv",
        ["omega", "amplitude_1", "amplitude_2", "output", "forcing_direction",
         "signal", "slope_mean", "se", "ci_low", "ci_high"],
        slope_rows,
    )


def _write_figures(run_dir, result):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    amplitudes = np.asarray(result["amplitudes"])
    labels = ("x", "y", "z")
    expected = {
        "linear_component": 1.0,
        "quadratic_component": 2.0,
        "odd": 1.0,
        "even": 2.0,
        "odd_ge3": 3.0,
        "even_ge4": 4.0,
    }
    with PdfPages(run_dir / "figures" / "higher_order_scaling.pdf") as pdf:
        for omega_index, omega in enumerate(result["frequencies"]):
            for direction in range(3):
                fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
                for name in ("linear_component", "quadratic_component", "odd", "even"):
                    values = result["signals"][name]["overall_mean"][omega_index]
                    axes[0].loglog(amplitudes, values, marker="o", label=name)
                for name in ("odd_ge3", "even_ge4", "quadratic_remainder"):
                    diagnostics = result["signals"][name]
                    for output in range(3):
                        mean = diagnostics["phase_l2_mean"][omega_index, :, output, direction]
                        low = diagnostics["phase_l2_ci_low"][omega_index, :, output, direction]
                        high = diagnostics["phase_l2_ci_high"][omega_index, :, output, direction]
                        axes[1].loglog(amplitudes, mean, marker="o",
                                       label=f"{name}:{labels[output]}")
                        axes[1].fill_between(amplitudes, np.maximum(low, 1e-300),
                                             np.maximum(high, 1e-300), alpha=0.1)
                for name, expected_slope in expected.items():
                    diagnostics = result["signals"][name]
                    midpoint = np.sqrt(amplitudes[:-1] * amplitudes[1:])
                    slope = diagnostics["local_slope_mean"][omega_index, :, :, direction]
                    low = diagnostics["local_slope_ci_low"][omega_index, :, :, direction]
                    high = diagnostics["local_slope_ci_high"][omega_index, :, :, direction]
                    axes[2].plot(midpoint, slope.mean(axis=-1), marker="o", label=name)
                    axes[2].fill_between(
                        midpoint,
                        low.mean(axis=-1),
                        high.mean(axis=-1),
                        alpha=0.1,
                    )
                    axes[2].axhline(expected_slope, linewidth=0.6, alpha=0.25)
                axes[0].set_ylabel("overall L2")
                axes[0].set_title("leading signals")
                axes[1].set_ylabel("phase L2")
                axes[1].set_title("explicit higher-order remainders with t confidence bands")
                axes[2].set_ylabel("local log-log slope")
                axes[2].set_xlabel("geometric midpoint amplitude")
                axes[2].set_xscale("log")
                for axis in axes:
                    axis.legend(frameon=False, fontsize=7, ncol=2)
                fig.suptitle(f"omega={omega:.6g}, forcing direction={direction}")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    with PdfPages(run_dir / "figures" / "higher_order_harmonics.pdf") as pdf:
        for omega_index, omega in enumerate(result["frequencies"]):
            for direction in range(3):
                fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
                for output, axis in enumerate(axes):
                    for harmonic in range(result["harmonics"]["quadratic_remainder"]
                                          ["amplitude_mean"].shape[-1]):
                        axis.loglog(
                            amplitudes,
                            result["harmonics"]["quadratic_remainder"]["amplitude_mean"]
                            [omega_index, :, output, direction, harmonic],
                            marker="o", label=f"k={harmonic}",
                        )
                    axis.set_ylabel(f"{labels[output]} amplitude")
                    axis.legend(frameon=False, fontsize=7, ncol=3)
                axes[-1].set_xlabel("forcing amplitude")
                fig.suptitle(
                    f"omega={omega:.6g}, direction={direction}: high-order residual harmonics\n"
                    "k=0 is offset, k=1 fundamental, k=2 second harmonic; adjacent bins shown"
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def _concatenate_nested(points, section):
    keys = points[0][section].keys()
    combined = {}
    for key in keys:
        first = points[0][section][key]
        if isinstance(first, dict):
            combined[key] = {
                field: np.concatenate([point[section][key][field] for point in points], axis=0)
                for field in first
            }
        else:
            combined[key] = np.concatenate(
                [point[section][key] for point in points], axis=0)
    return combined


def run(cfg, response_result, amplitude_result, run_dir, runtime, resume=False):
    run_dir = Path(run_dir)
    amplitudes = np.asarray(amplitude_result["amplitudes"], dtype=float)
    zero, positive, negative = _axis_raw(amplitude_result)
    odd = (positive - negative) / 2.0
    even = (positive + negative - 2.0 * zero) / 2.0
    delta = positive - zero
    L = np.asarray(response_result["L_seed"], dtype=float)
    H = np.asarray(response_result["H_seed"], dtype=float)
    n_omega, n_amp, n_seed = delta.shape[:3]
    L_axis = np.stack([L[:, :, :, direction] for direction in range(3)], axis=3)
    H_axis = np.stack([
        H[:, :, :, direction, direction] for direction in range(3)
    ], axis=3)
    # L_axis/H_axis: omega, seed, output, direction, phase
    amplitude_shape = (1, n_amp, 1, 1, 1, 1)
    amplitude_array = amplitudes.reshape(amplitude_shape)
    confidence = cfg.get("confidence_level", 0.95)
    completed = storage.completed_parameters(run_dir) if resume else set()
    points = []
    for omega_index in range(n_omega):
        key = f"omega:{omega_index}"
        checkpoint = storage.checkpoint_path(run_dir, key)
        if key in completed and storage.valid_npz(checkpoint, ("result",)):
            with np.load(checkpoint, allow_pickle=True) as data:
                point = data["result"].item()
        else:
            linear_component = amplitude_array * L_axis[omega_index:omega_index + 1, None]
            quadratic_component = (
                0.5 * amplitude_array ** 2
                * H_axis[omega_index:omega_index + 1, None]
            )
            point_delta = delta[omega_index:omega_index + 1]
            point_odd = odd[omega_index:omega_index + 1]
            point_even = even[omega_index:omega_index + 1]
            odd_ge3 = point_odd - linear_component
            even_ge4 = point_even - quadratic_component
            linear_remainder = point_delta - linear_component
            quadratic_remainder = point_delta - linear_component - quadratic_component
            raw_signals = {
                "delta": point_delta,
                "linear_component": linear_component,
                "quadratic_component": quadratic_component,
                "odd": point_odd,
                "even": point_even,
                "odd_ge3": odd_ge3,
                "even_ge4": even_ge4,
                "linear_remainder": linear_remainder,
                "quadratic_remainder": quadratic_remainder,
            }
            point = {
                "signals": _signal_statistics(
                    raw_signals,
                    amplitudes,
                    confidence,
                    cfg["bootstrap_samples"],
                    cfg["random_seed"] + omega_index,
                ),
                "ratios": {
                    "high_to_quadratic": _ratio_statistics(
                        quadratic_remainder, quadratic_component, confidence),
                    "linear_relative": _ratio_statistics(
                        linear_remainder, point_delta, confidence),
                    "quadratic_relative": _ratio_statistics(
                        quadratic_remainder, point_delta, confidence),
                },
                "harmonics": {
                    name: _harmonic_statistics(
                        values, int(max(cfg["harmonics"])), confidence)
                    for name, values in raw_signals.items()
                },
                "raw_signals": raw_signals,
            }
            storage.save_npz_atomic(checkpoint, result=np.array(point, dtype=object))
            storage.mark_parameter(run_dir, key, runtime)
        points.append(point)
    signal_diagnostics = _concatenate_nested(points, "signals")
    ratios = _concatenate_nested(points, "ratios")
    harmonics = _concatenate_nested(points, "harmonics")
    signals = _concatenate_nested(points, "raw_signals")
    result = {
        "task": "higher-order",
        "frequencies": np.asarray(response_result["frequencies"]),
        "amplitudes": amplitudes,
        "signals": signal_diagnostics,
        "ratios": ratios,
        "harmonics": harmonics,
        "raw_signals": signals,
        "expected_power_slopes": {
            "linear_component": 1.0,
            "quadratic_component": 2.0,
            "odd_ge3": 3.0,
            "even_ge4": 4.0,
        },
        "sampling_metadata": response_result["sampling_metadata"],
    }
    _write_tables(run_dir, result)
    _write_figures(run_dir, result)
    return result
