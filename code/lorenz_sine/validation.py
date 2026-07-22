"""Statistical detectability and finite-amplitude truncation validity windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import storage


CATEGORY_CODES = {
    "noise-dominated": 0,
    "linear-valid": 1,
    "second-order-detectable": 2,
    "quadratic-truncation-valid": 3,
    "higher-order-contaminated": 4,
}


def _detectability(signal_diagnostics, snr_min):
    mean = np.asarray(signal_diagnostics["phase_l2_mean"])
    se = np.asarray(signal_diagnostics["phase_l2_se"])
    low = np.asarray(signal_diagnostics["phase_l2_ci_low"])
    snr = mean / np.maximum(se, 1e-300)
    detectable = (low > 0.0) & (snr >= float(snr_min))
    return detectable, snr


def _intervals(mask, amplitudes):
    values = np.asarray(amplitudes)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return {"amplitude_min": None, "amplitude_max": None, "values": []}
    return {
        "amplitude_min": float(values.min()),
        "amplitude_max": float(values.max()),
        "values": [float(value) for value in values],
    }


def _write_tables(run_dir, result):
    rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for amplitude_index, amplitude in enumerate(result["amplitudes"]):
            for output in range(3):
                for direction in range(3):
                    rows.append([
                        omega, amplitude, output, direction,
                        result["linear_snr"][omega_index, amplitude_index, output, direction],
                        result["quadratic_snr"][omega_index, amplitude_index, output, direction],
                        result["linear_detectable"][omega_index, amplitude_index,
                                                    output, direction],
                        result["second_order_detectable"][omega_index, amplitude_index,
                                                          output, direction],
                        result["linear_valid"][omega_index, amplitude_index,
                                               output, direction],
                        result["quadratic_valid"][omega_index, amplitude_index,
                                                  output, direction],
                        result["linear_relative_ci_high"][omega_index, amplitude_index,
                                                          output, direction],
                        result["quadratic_relative_ci_high"][omega_index, amplitude_index,
                                                             output, direction],
                        result["high_to_quadratic_ci_high"][omega_index, amplitude_index,
                                                            output, direction],
                        result["category"][omega_index, amplitude_index, output, direction],
                    ])
    storage.write_csv(
        run_dir / "tables" / "validation_by_amplitude.csv",
        ["omega", "amplitude", "output", "forcing_direction", "linear_snr",
         "quadratic_snr", "linear_detectable", "second_order_detectable",
         "linear_valid", "quadratic_valid", "linear_relative_ci_high",
         "quadratic_relative_ci_high", "high_to_quadratic_ci_high", "category"],
        rows,
    )
    interval_rows = []
    for key, value in result["windows"].items():
        omega, output, direction = key.split(":")
        for name, interval in value.items():
            interval_rows.append([
                omega, output, direction, name, interval["amplitude_min"],
                interval["amplitude_max"], "|".join(str(item) for item in interval["values"]),
            ])
    storage.write_csv(
        run_dir / "tables" / "validation_windows.csv",
        ["omega", "output", "forcing_direction", "window", "amplitude_min",
         "amplitude_max", "tested_amplitudes"],
        interval_rows,
    )


def _write_figures(run_dir, result):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    amplitudes = np.asarray(result["amplitudes"])
    thresholds = result["thresholds"]
    labels = ("x", "y", "z")
    with PdfPages(run_dir / "figures" / "validation_windows.pdf") as pdf:
        for omega_index, omega in enumerate(result["frequencies"]):
            for direction in range(3):
                fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
                for output in range(3):
                    axes[0].semilogx(
                        amplitudes,
                        result["linear_snr"][omega_index, :, output, direction],
                        marker="o", label=f"linear {labels[output]}",
                    )
                    axes[0].semilogx(
                        amplitudes,
                        result["quadratic_snr"][omega_index, :, output, direction],
                        marker="s", linestyle="--", label=f"quadratic {labels[output]}",
                    )
                    axes[1].semilogx(
                        amplitudes,
                        result["linear_relative_ci_high"][omega_index, :, output, direction],
                        marker="o", label=f"linear remainder / delta {labels[output]}",
                    )
                    axes[1].semilogx(
                        amplitudes,
                        result["quadratic_relative_ci_high"][omega_index, :, output, direction],
                        marker="s", label=f"quadratic remainder / delta {labels[output]}",
                    )
                    axes[1].semilogx(
                        amplitudes,
                        result["high_to_quadratic_ci_high"][omega_index, :, output, direction],
                        marker="^", label=f"high / quadratic {labels[output]}",
                    )
                    axes[2].scatter(
                        amplitudes,
                        result["category_code"][omega_index, :, output, direction]
                        + 0.06 * output,
                        label=labels[output],
                    )
                axes[0].axhline(thresholds["signal_to_noise_min"], color="black",
                                linestyle="--", label="configured detection threshold")
                axes[1].axhline(thresholds["linear_relative_error_max"],
                                color="tab:blue", linestyle=":")
                axes[1].axhline(thresholds["quadratic_relative_error_max"],
                                color="tab:orange", linestyle=":")
                axes[1].axhline(thresholds["high_to_quadratic_max"],
                                color="tab:green", linestyle=":")
                axes[0].set_ylabel("signal / standard error")
                axes[1].set_ylabel("upper confidence bound on ratio")
                axes[1].set_yscale("log")
                axes[2].set_ylabel("classification")
                axes[2].set_yticks(list(CATEGORY_CODES.values()))
                axes[2].set_yticklabels(list(CATEGORY_CODES.keys()))
                axes[2].set_xlabel("amplitude")
                for axis in axes:
                    axis.legend(frameon=False, fontsize=7, ncol=2)
                fig.suptitle(
                    f"omega={omega:.6g}, forcing direction={direction}\n"
                    "Thresholds are configured decision rules, not physical constants"
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def run(cfg, response_result, higher_result, run_dir):
    run_dir = Path(run_dir)
    signals = higher_result["signals"]
    ratios = higher_result["ratios"]
    linear_detectable, linear_snr = _detectability(
        signals["linear_component"], cfg["signal_to_noise_min"])
    quadratic_detectable, quadratic_snr = _detectability(
        signals["quadratic_component"], cfg["signal_to_noise_min"])
    linear_ratio_high = np.asarray(ratios["linear_relative"]["ci_high"])
    quadratic_ratio_high = np.asarray(ratios["quadratic_relative"]["ci_high"])
    high_to_quadratic_high = np.asarray(ratios["high_to_quadratic"]["ci_high"])
    linear_valid = linear_detectable & (
        linear_ratio_high <= float(cfg["linear_relative_error_max"]))
    quadratic_valid = (
        quadratic_detectable
        & (quadratic_ratio_high <= float(cfg["quadratic_relative_error_max"]))
        & (high_to_quadratic_high <= float(cfg["high_to_quadratic_max"]))
    )
    higher_contaminated = quadratic_detectable & ~quadratic_valid
    noise_dominated = ~linear_detectable
    shape = linear_valid.shape
    masks = {
        "noise-dominated": noise_dominated,
        "linear-valid": linear_valid,
        "second-order-detectable": quadratic_detectable,
        "quadratic-truncation-valid": quadratic_valid,
        "higher-order-contaminated": higher_contaminated,
    }
    category = np.full(shape, "noise-dominated", dtype="U32")
    for name in cfg["classification_order"]:
        category[masks[name]] = name
    category_code = np.vectorize(CATEGORY_CODES.__getitem__)(category)
    amplitudes = np.asarray(higher_result["amplitudes"], dtype=float)
    windows = {}
    for omega_index, omega in enumerate(higher_result["frequencies"]):
        for output in range(3):
            for direction in range(3):
                key = f"{float(omega)}:{output}:{direction}"
                windows[key] = {
                    "noise-dominated range": _intervals(
                        noise_dominated[omega_index, :, output, direction], amplitudes),
                    "linear-valid range": _intervals(
                        linear_valid[omega_index, :, output, direction], amplitudes),
                    "second-order-detectable range": _intervals(
                        quadratic_detectable[omega_index, :, output, direction], amplitudes),
                    "quadratic-truncation-valid range": _intervals(
                        quadratic_valid[omega_index, :, output, direction], amplitudes),
                    "higher-order-contaminated range": _intervals(
                        higher_contaminated[omega_index, :, output, direction], amplitudes),
                }
    result = {
        "task": "validate",
        "frequencies": np.asarray(higher_result["frequencies"]),
        "amplitudes": amplitudes,
        "thresholds": {
            "signal_to_noise_min": float(cfg["signal_to_noise_min"]),
            "linear_relative_error_max": float(cfg["linear_relative_error_max"]),
            "quadratic_relative_error_max": float(cfg["quadratic_relative_error_max"]),
            "high_to_quadratic_max": float(cfg["high_to_quadratic_max"]),
            "interpretation": "configured decision thresholds; not natural constants",
        },
        "linear_snr": linear_snr,
        "quadratic_snr": quadratic_snr,
        "linear_detectable": linear_detectable,
        "second_order_detectable": quadratic_detectable,
        "linear_valid": linear_valid,
        "quadratic_valid": quadratic_valid,
        "higher_order_contaminated": higher_contaminated,
        "noise_dominated": noise_dominated,
        "linear_relative_ci_high": linear_ratio_high,
        "quadratic_relative_ci_high": quadratic_ratio_high,
        "high_to_quadratic_ci_high": high_to_quadratic_high,
        "category": category,
        "category_code": category_code,
        "windows": windows,
        "response_model_degrees": {
            "L": np.asarray(response_result["selected_L_degree"]),
            "H": np.asarray(response_result["selected_H_degree"]),
        },
    }
    _write_tables(run_dir, result)
    _write_figures(run_dir, result)
    return result
