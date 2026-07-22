"""Independent consolidated research report assembled from recorded run ancestry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from . import storage
from .statistics import phase_l2


def _text_page(pdf, title, payload):
    fig, axis = plt.subplots(figsize=(8.5, 11))
    axis.axis("off")
    text = json.dumps(payload, indent=2, default=storage.json_default)
    axis.text(0.02, 0.98, title + "\n\n" + text, va="top",
              family="monospace", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def _spectrum_pages(pdf, spectrum):
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    frequencies = spectrum["freqs"]
    use = frequencies > 0
    for output, axis in enumerate(axes):
        axis.loglog(frequencies[use], spectrum["psd_mean"][output, use],
                    label=labels[output])
        axis.fill_between(
            frequencies[use],
            np.maximum(spectrum["psd_ci_low"][output, use], 1e-300),
            np.maximum(spectrum["psd_ci_high"][output, use], 1e-300),
            alpha=0.2,
        )
        for row in spectrum["peaks"]:
            axis.axvline(row[2], color="tab:red", alpha=0.3)
        axis.set_ylabel("PSD")
    axes[-1].set_xlabel("frequency")
    metadata = spectrum["sampling_metadata"]
    fig.suptitle(
        "Natural spectrum with finite-seed confidence bands\n"
        f"resolution={metadata['frequency_resolution']:.5g}, "
        f"Nyquist={metadata['nyquist_frequency']:.5g}"
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _sampling_page(pdf, sampling):
    cases = [case for case in sampling["cases"] if case["parameter"] != "reference"]
    indices = [index for index, case in enumerate(sampling["cases"])
               if case["parameter"] != "reference"]
    fig, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(indices))
    mean = sampling["relative_error_mean"][indices]
    low = sampling["relative_error_ci_low"][indices]
    high = sampling["relative_error_ci_high"][indices]
    axis.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), fmt="o")
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{case['parameter']}={case['value']}" for case in cases],
                         rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("relative error")
    axis.set_title("Sampling/solver convergence scan (recommendations remain advisory)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _steady_pages(pdf, steady):
    _text_page(pdf, "Steady-state platform recommendations", steady["recommendations"])
    for omega_index, omega in enumerate(steady["frequencies"]):
        fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
        skips = steady["candidate_n_skip"]
        for output in range(3):
            for component, name in enumerate(("cos", "sin")):
                axis = axes[output, component]
                for harmonic in range(min(3, steady["coefficient_mean"].shape[-2])):
                    mean = steady["coefficient_mean"][omega_index, :, -1,
                                                        output, harmonic, component]
                    low = steady["coefficient_ci_low"][omega_index, :, -1,
                                                         output, harmonic, component]
                    high = steady["coefficient_ci_high"][omega_index, :, -1,
                                                          output, harmonic, component]
                    axis.plot(skips, mean, marker="o", label=f"k={harmonic}")
                    axis.fill_between(skips, low, high, alpha=0.12)
                axis.set_title(f"output={output}, {name}, final block")
                axis.legend(frameon=False, fontsize=7)
        axes[-1, 0].set_xlabel("n_skip")
        axes[-1, 1].set_xlabel("n_skip")
        fig.suptitle(f"omega={omega:.6g}: steady Fourier coefficients and t-CIs")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _amplitude_pages(pdf, amplitude):
    amplitudes = amplitude["amplitudes"]
    for omega_index, omega in enumerate(amplitude["frequencies"]):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        for direction in range(3):
            for axis, prefix, title in zip(
                axes.ravel(),
                ("odd_norm", "even_norm", "odd_scaled_norm", "even_scaled_norm"),
                ("odd", "even", "odd/A", "2 even/A^2"),
            ):
                mean = amplitude[f"{prefix}_mean"][omega_index, :, :, direction].mean(axis=-1)
                low = amplitude[f"{prefix}_ci_low"][omega_index, :, :, direction].mean(axis=-1)
                high = amplitude[f"{prefix}_ci_high"][omega_index, :, :, direction].mean(axis=-1)
                axis.loglog(amplitudes, mean, marker="o", label=f"direction {direction}")
                axis.fill_between(amplitudes, np.maximum(low, 1e-300),
                                  np.maximum(high, 1e-300), alpha=0.1)
                axis.set_title(title)
                axis.legend(frameon=False)
        fig.suptitle(f"omega={omega:.6g}: paired-amplitude odd/even scaling")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _response_pages(pdf, response):
    labels = ("x", "y", "z")
    model_summary = []
    for omega_index, omega in enumerate(response["frequencies"]):
        for quantity, models, selected in (
            ("L", response["L_models"], response["selected_L_degree"]),
            ("H", response["H_models"], response["selected_H_degree"]),
        ):
            for degree, model in models[omega_index].items():
                model_summary.append({
                    "omega": float(omega),
                    "quantity": quantity,
                    "degree_in_A2": int(degree),
                    "selected": int(degree) == int(selected[omega_index]),
                    "rss": model["rss"],
                    "mean_r_squared": float(np.nanmean(model["r_squared"])),
                    "bic": model["bic"],
                    "loo_rmse": model["loo_rmse"],
                    "drop_largest_relative_change": model["amplitude_sensitivity"]
                    ["drop_largest"]["relative_change_l2"],
                    "drop_smallest_relative_change": model["amplitude_sensitivity"]
                    ["drop_smallest"]["relative_change_l2"],
                })
    _text_page(pdf, "Response extrapolation model comparison", model_summary)
    for omega_index, omega in enumerate(response["frequencies"]):
        theta = np.linspace(0.0, 2.0 * np.pi, response["L_mean"].shape[-1], endpoint=False)
        fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
        for output in range(3):
            for direction in range(3):
                axis = axes[output, direction]
                axis.plot(theta, response["L_mean"][omega_index, output, direction])
                axis.fill_between(
                    theta,
                    response["L_ci_low"][omega_index, output, direction],
                    response["L_ci_high"][omega_index, output, direction],
                    alpha=0.2,
                )
                axis.set_title(f"L[{labels[output]},{labels[direction]}]")
        fig.suptitle(
            f"omega={omega:.6g}: all first-order response components, "
            f"selected degree={response['selected_L_degree'][omega_index]}"
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for output in range(3):
            fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True)
            for axis, (first, second) in zip(axes.ravel(),
                                             ((0, 0), (0, 1), (0, 2),
                                              (1, 1), (1, 2), (2, 2))):
                axis.plot(theta, response["H_mean"][omega_index, output, first, second])
                axis.fill_between(
                    theta,
                    response["H_ci_low"][omega_index, output, first, second],
                    response["H_ci_high"][omega_index, output, first, second],
                    alpha=0.2,
                )
                axis.set_title(f"H[{labels[output]},{labels[first]},{labels[second]}]")
            fig.suptitle(
                f"omega={omega:.6g}: independent symmetric H components, output {labels[output]}, "
                f"selected degree={response['selected_H_degree'][omega_index]}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _higher_pages(pdf, higher):
    amplitudes = higher["amplitudes"]
    expected = higher["expected_power_slopes"]
    for omega_index, omega in enumerate(higher["frequencies"]):
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
        for name in ("linear_component", "quadratic_component", "odd_ge3", "even_ge4"):
            diagnostics = higher["signals"][name]
            axes[0].loglog(amplitudes, diagnostics["overall_mean"][omega_index],
                           marker="o", label=name)
            slope = diagnostics["local_slope_mean"][omega_index].mean(axis=(1, 2))
            axes[1].plot(np.sqrt(amplitudes[:-1] * amplitudes[1:]), slope,
                         marker="o", label=f"{name} (expected {expected.get(name)})")
        axes[0].set_ylabel("overall L2 norm")
        axes[1].set_ylabel("local log-log slope")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("amplitude midpoint")
        for axis in axes:
            axis.legend(frameon=False)
        fig.suptitle(f"omega={omega:.6g}: leading and explicit higher-order scaling")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _validation_pages(pdf, validation):
    _text_page(pdf, "Configured validation thresholds", validation["thresholds"])
    for omega_index, omega in enumerate(validation["frequencies"]):
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        amplitudes = validation["amplitudes"]
        for direction in range(3):
            quadratic_valid_fraction = validation["quadratic_valid"][
                omega_index, :, :, direction].mean(axis=1)
            linear_valid_fraction = validation["linear_valid"][
                omega_index, :, :, direction].mean(axis=1)
            axes[0].plot(amplitudes, linear_valid_fraction, marker="o",
                         label=f"linear valid direction {direction}")
            axes[0].plot(amplitudes, quadratic_valid_fraction, marker="s",
                         label=f"quadratic valid direction {direction}")
            axes[1].plot(
                amplitudes,
                validation["high_to_quadratic_ci_high"][omega_index, :, :, direction]
                .mean(axis=1),
                marker="o", label=f"direction {direction}",
            )
        axes[0].set_ylabel("fraction of outputs passing")
        axes[1].set_ylabel("mean upper CI: high / quadratic")
        axes[1].set_xlabel("amplitude")
        axes[1].set_xscale("log")
        for axis in axes:
            axis.legend(frameon=False, fontsize=7)
        fig.suptitle(f"omega={omega:.6g}: detectability and truncation windows")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def run(cfg, source_runs: dict, run_dir):
    run_dir = Path(run_dir)
    report_path = run_dir / "report.pdf"
    with PdfPages(report_path) as pdf:
        _text_page(pdf, cfg["title"], {
            "source_runs": {key: value[0] for key, value in source_runs.items()},
            "statement": (
                "Detectability and validity thresholds are configured decision rules. "
                "They are not inferred physical constants."
            ),
        })
        if "spectrum" in source_runs:
            _spectrum_pages(pdf, source_runs["spectrum"][1])
        if "sampling-check" in source_runs:
            _sampling_page(pdf, source_runs["sampling-check"][1])
        if "steady" in source_runs:
            _steady_pages(pdf, source_runs["steady"][1])
        if "amplitude-scan" in source_runs:
            _amplitude_pages(pdf, source_runs["amplitude-scan"][1])
        if "response" in source_runs:
            _response_pages(pdf, source_runs["response"][1])
        if "higher-order" in source_runs:
            _higher_pages(pdf, source_runs["higher-order"][1])
        if "validate" in source_runs:
            _validation_pages(pdf, source_runs["validate"][1])
    shutil.copyfile(report_path, run_dir / "figures" / "report.pdf")
    provenance = {
        task: {"run_id": run_id, "manifest": manifest}
        for task, (run_id, _, manifest) in source_runs.items()
    }
    storage.write_json(run_dir / "data" / "report_sources.json", provenance)
    storage.write_csv(
        run_dir / "tables" / "report_sources.csv",
        ["task", "run_id", "config_hash"],
        [[task, data["run_id"], data["manifest"]["config_hash"]]
         for task, data in provenance.items()],
    )
    return {
        "task": "report",
        "report_path": "report.pdf",
        "source_run_ids": {task: data[0] for task, data in source_runs.items()},
    }
