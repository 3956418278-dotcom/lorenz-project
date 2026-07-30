"""Paired finite-amplitude simulations and odd/even decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import storage
from .config import forced_sampling_metadata
from .fourier import fourier_coefficients
from .response import phase_mean_cached
from .statistics import mean_se_ci, one_sample_t_test, phase_l2, safe_ratio


def forcing_combinations(cfg):
    combinations = [{"name": "zero", "vector_unit": [0.0, 0.0, 0.0]}]
    for direction in cfg["forcing_directions"]:
        for sign, suffix in ((1, "pos"), (-1, "neg")):
            vector = np.zeros(3)
            vector[int(direction)] = sign
            combinations.append({
                "name": f"axis_{int(direction)}_{suffix}",
                "vector_unit": vector.tolist(),
            })
    for first, second in cfg["mixed_pairs"]:
        for first_sign, first_name in ((1, "p"), (-1, "m")):
            for second_sign, second_name in ((1, "p"), (-1, "m")):
                vector = np.zeros(3)
                vector[int(first)] = first_sign
                vector[int(second)] = second_sign
                combinations.append({
                    "name": f"pair_{int(first)}_{int(second)}_{first_name}{second_name}",
                    "vector_unit": vector.tolist(),
                })
    return combinations


def amplitude_seed_worker(seed, omega, amplitude, n_skip, cfg, combinations):
    local_runtime = storage.create_runtime()
    values = []
    for combination in combinations:
        vector = float(amplitude) * np.asarray(combination["vector_unit"], dtype=float)
        values.append(phase_mean_cached(
            int(seed),
            vector,
            float(omega),
            cfg.get("phase", 0.0),
            int(n_skip),
            int(cfg["n_record_cycles"]),
            int(cfg["n_phase"]),
            cfg,
            local_runtime,
        ))
    return int(seed), np.asarray(values), dict(local_runtime["cache"])


def _point_key(omega_index, amplitude_index, seed):
    return f"omega:{omega_index}:amplitude:{amplitude_index}:seed:{int(seed)}"


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return int(data["seed"]), data["phase_means"], {"hits": 0, "misses": 0}


def _combo_index(combinations):
    return {item["name"]: index for index, item in enumerate(combinations)}


def _decompose(phase_means, amplitudes, combinations, cfg):
    index = _combo_index(combinations)
    n_omega, n_amp, _, n_seed, _, n_phase = phase_means.shape
    odd = np.full((n_omega, n_amp, n_seed, 3, 3, n_phase), np.nan)
    even = np.full_like(odd, np.nan)
    zero = phase_means[:, :, index["zero"]]
    for direction in cfg["forcing_directions"]:
        direction = int(direction)
        positive = phase_means[:, :, index[f"axis_{direction}_pos"]]
        negative = phase_means[:, :, index[f"axis_{direction}_neg"]]
        odd[:, :, :, :, direction] = (positive - negative) / 2.0
        even[:, :, :, :, direction] = (positive + negative - 2.0 * zero) / 2.0
    amplitude_shape = (1, n_amp, 1, 1, 1, 1)
    amplitude_array = np.asarray(amplitudes).reshape(amplitude_shape)
    odd_scaled = odd / amplitude_array
    even_scaled = 2.0 * even / amplitude_array ** 2
    return odd, even, odd_scaled, even_scaled


def _leakage_ratio(coefficients, expected_harmonics):
    amplitude = np.sqrt(np.sum(np.asarray(coefficients) ** 2, axis=-1))
    total = np.sqrt(np.sum(amplitude ** 2, axis=-1))
    keep = np.zeros(amplitude.shape[-1], dtype=bool)
    keep[np.asarray(expected_harmonics, dtype=int)] = True
    outside = np.sqrt(np.sum(amplitude[..., ~keep] ** 2, axis=-1))
    return safe_ratio(outside, total)


def _target_harmonics(group: str, kmax: int):
    if group == "odd":
        return [k for k in range(1, kmax + 1) if k % 2 == 1]
    if group == "even":
        return [k for k in range(0, kmax + 1) if k % 2 == 0]
    raise ValueError(f"unknown response group {group!r}")


def _component_names(harmonic: int):
    if int(harmonic) == 0:
        return [(0, "dc")]
    return [(0, "cos"), (1, "sin")]


def _harmonic_class(group: str, harmonic: int) -> str:
    harmonic = int(harmonic)
    if group == "odd" and harmonic == 1:
        return "first_order_fundamental"
    if group == "even" and harmonic in (0, 2):
        return "second_order_dc_or_second_harmonic"
    return "tested_higher_harmonic"


def _signed_fourier_tests(coefficients, cfg):
    statistics = one_sample_t_test(
        coefficients, cfg.get("confidence_level", 0.95), axis=2)
    alpha = float(cfg.get("significance_alpha", 0.05))
    statistics["significant"] = statistics["p_value"] < alpha
    statistics["alpha"] = alpha
    statistics["confidence_level"] = float(cfg.get("confidence_level", 0.95))
    return statistics


def _write_tables(run_dir, result, cfg):
    rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for amplitude_index, amplitude in enumerate(result["amplitudes"]):
            for output in range(3):
                for direction in cfg["forcing_directions"]:
                    direction = int(direction)
                    rows.append([
                        omega, amplitude, output, direction,
                        result["odd_norm_mean"][omega_index, amplitude_index,
                                                 output, direction],
                        result["even_norm_mean"][omega_index, amplitude_index,
                                                  output, direction],
                        result["odd_scaled_norm_mean"][omega_index, amplitude_index,
                                                        output, direction],
                        result["even_scaled_norm_mean"][omega_index, amplitude_index,
                                                         output, direction],
                        result["odd_leakage_mean"][omega_index, amplitude_index,
                                                    output, direction],
                        result["even_leakage_mean"][omega_index, amplitude_index,
                                                     output, direction],
                    ])
    storage.write_csv(
        run_dir / "tables" / "amplitude_scaling.csv",
        ["omega", "amplitude", "output", "forcing_direction", "odd_l2",
         "even_l2", "odd_over_a_l2", "two_even_over_a2_l2",
         "odd_unexpected_harmonic_ratio", "even_unexpected_harmonic_ratio"],
        rows,
    )
    significance_rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for amplitude_index, amplitude in enumerate(result["amplitudes"]):
            for output in range(3):
                for direction in cfg["forcing_directions"]:
                    direction = int(direction)
                    for group in ("odd", "even"):
                        tests = result[f"{group}_fourier_t_test"]
                        for harmonic in _target_harmonics(
                                group, int(cfg["fourier_kmax"])):
                            for component, component_name in _component_names(harmonic):
                                significance_rows.append([
                                    omega,
                                    amplitude,
                                    output,
                                    direction,
                                    group,
                                    _harmonic_class(group, harmonic),
                                    harmonic,
                                    component_name,
                                    tests["n"],
                                    tests["mean"][omega_index, amplitude_index,
                                                  output, direction,
                                                  harmonic, component],
                                    tests["std"][omega_index, amplitude_index,
                                                 output, direction,
                                                 harmonic, component],
                                    tests["se"][omega_index, amplitude_index,
                                                output, direction,
                                                harmonic, component],
                                    tests["t_statistic"][omega_index, amplitude_index,
                                                         output, direction,
                                                         harmonic, component],
                                    tests["df"],
                                    tests["p_value"][omega_index, amplitude_index,
                                                     output, direction,
                                                     harmonic, component],
                                    tests["ci_low"][omega_index, amplitude_index,
                                                    output, direction,
                                                    harmonic, component],
                                    tests["ci_high"][omega_index, amplitude_index,
                                                     output, direction,
                                                     harmonic, component],
                                    bool(tests["significant"][omega_index,
                                                              amplitude_index,
                                                              output, direction,
                                                              harmonic, component]),
                                    tests["alpha"],
                                    tests["confidence_level"],
                                ])
    storage.write_csv(
        run_dir / "tables" / "signed_fourier_t_tests.csv",
        ["omega", "amplitude", "output", "forcing_direction", "response_group",
         "harmonic_class", "harmonic", "component", "n_seed", "sample_mean",
         "sample_standard_deviation", "standard_error", "t_statistic",
         "degrees_of_freedom", "two_sided_p_value", "ci_low", "ci_high",
         "significant", "significance_alpha", "confidence_level"],
        significance_rows,
    )
    storage.write_csv(
        run_dir / "tables" / "signed_fft_t_tests.csv",
        ["omega", "amplitude", "output", "forcing_direction", "response_group",
         "harmonic_class", "harmonic", "component", "n_seed", "sample_mean",
         "sample_standard_deviation", "standard_error", "t_statistic",
         "degrees_of_freedom", "two_sided_p_value", "ci_low", "ci_high",
         "significant", "significance_alpha", "confidence_level"],
        significance_rows,
    )
    storage.write_json(run_dir / "data" / "sampling_metadata.json",
                       result["sampling_metadata"])


def _write_figures(run_dir, result, cfg):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    amplitudes = np.asarray(result["amplitudes"])
    labels = ("x", "y", "z")
    path = run_dir / "figures" / "amplitude_odd_even_scaling.pdf"
    with PdfPages(path) as pdf:
        for omega_index, omega in enumerate(result["frequencies"]):
            for direction in cfg["forcing_directions"]:
                direction = int(direction)
                fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
                quantities = (
                    ("odd_norm", "odd signal", 1),
                    ("even_norm", "even signal", 2),
                    ("odd_scaled_norm", "odd / A", 0),
                    ("even_scaled_norm", "2 even / A^2", 0),
                )
                for axis, (prefix, title, expected_slope) in zip(axes.ravel(), quantities):
                    for output in range(3):
                        mean = result[f"{prefix}_mean"][omega_index, :, output, direction]
                        low = result[f"{prefix}_ci_low"][omega_index, :, output, direction]
                        high = result[f"{prefix}_ci_high"][omega_index, :, output, direction]
                        axis.loglog(amplitudes, mean, marker="o", label=labels[output])
                        axis.fill_between(amplitudes, np.maximum(low, 1e-300),
                                          np.maximum(high, 1e-300), alpha=0.12)
                    axis.set_title(f"{title}; expected leading slope {expected_slope}")
                    axis.set_xlabel("amplitude")
                    axis.set_ylabel("phase L2 norm")
                    axis.legend(frameon=False)
                fig.suptitle(f"omega={omega:.6g}, forcing direction={direction}")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    signed_paths = (
        run_dir / "figures" / "signed_fourier_t_tests.pdf",
        run_dir / "figures" / "signed_fft_t_tests.pdf",
    )
    for signed_path in signed_paths:
        with PdfPages(signed_path) as pdf:
            _write_signed_fft_pages(pdf, result, cfg, amplitudes, labels)

    harmonic_path = run_dir / "figures" / "amplitude_harmonics.pdf"
    with PdfPages(harmonic_path) as pdf:
        for omega_index, omega in enumerate(result["frequencies"]):
            for direction in cfg["forcing_directions"]:
                direction = int(direction)
                fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
                for output in range(3):
                    for column, (key, title) in enumerate(
                            (("odd_harmonic_amplitude_mean", "odd"),
                             ("even_harmonic_amplitude_mean", "even"))):
                        axis = axes[output, column]
                        for harmonic in range(min(4, int(cfg["fourier_kmax"]) + 1)):
                            axis.loglog(
                                amplitudes,
                                result[key][omega_index, :, output, direction, harmonic],
                                marker="o",
                                label=f"k={harmonic}",
                            )
                            prefix = "odd" if title == "odd" else "even"
                            low = result[f"{prefix}_harmonic_amplitude_ci_low"][
                                omega_index, :, output, direction, harmonic]
                            high = result[f"{prefix}_harmonic_amplitude_ci_high"][
                                omega_index, :, output, direction, harmonic]
                            axis.fill_between(amplitudes, np.maximum(low, 1e-300),
                                              np.maximum(high, 1e-300), alpha=0.08)
                        axis.set_title(f"{title}, output {labels[output]}")
                        axis.set_ylabel("harmonic amplitude")
                        axis.legend(frameon=False, fontsize=8)
                axes[-1, 0].set_xlabel("amplitude")
                axes[-1, 1].set_xlabel("amplitude")
                fig.suptitle(f"omega={omega:.6g}, direction={direction}: aligned harmonics")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def _write_signed_fft_pages(pdf, result, cfg, amplitudes, labels):
    import matplotlib.pyplot as plt

    confidence = 100.0 * float(cfg.get("confidence_level", 0.95))
    for omega_index, omega in enumerate(result["frequencies"]):
        for direction in cfg["forcing_directions"]:
            direction = int(direction)
            for group in ("odd", "even"):
                tests = result[f"{group}_fourier_t_test"]
                targets = _target_harmonics(group, int(cfg["fourier_kmax"]))
                if not targets:
                    continue
                fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
                for output, axis in enumerate(axes):
                    for harmonic in targets:
                        for component, component_name in _component_names(harmonic):
                            mean = tests["mean"][omega_index, :, output,
                                                 direction, harmonic, component]
                            low = tests["ci_low"][omega_index, :, output,
                                                  direction, harmonic, component]
                            high = tests["ci_high"][omega_index, :, output,
                                                   direction, harmonic, component]
                            if group == "odd" and harmonic == 1:
                                linewidth = 1.8
                            elif group == "even" and harmonic in (0, 2):
                                linewidth = 1.8
                            else:
                                linewidth = 0.9
                            axis.plot(
                                amplitudes,
                                mean,
                                marker="o",
                                linewidth=linewidth,
                                label=f"k={harmonic} {component_name}",
                            )
                            axis.fill_between(amplitudes, low, high, alpha=0.08)
                    axis.axhline(0.0, color="black", linewidth=0.8)
                    axis.set_ylabel(f"{labels[output]} coefficient")
                    axis.legend(frameon=False, fontsize=7, ncol=3)
                axes[-1].set_xlabel("forcing amplitude")
                title = (
                    f"omega={omega:.6g}, forcing direction={direction}, {group} response\n"
                    f"signed FFT/Fourier coefficient seed mean and {confidence:.1f}% Student-t CI"
                )
                fig.suptitle(title)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def run(cfg, run_dir, runtime, resume=False, client=None):
    run_dir = Path(run_dir)
    combinations = forcing_combinations(cfg)
    frequencies = np.asarray(cfg["frequencies"], dtype=float)
    amplitudes = np.asarray(cfg["amplitudes"], dtype=float)
    n_seed = int(cfg["n_seed"])
    phase_means = np.empty((
        len(frequencies), len(amplitudes), len(combinations), n_seed, 3,
        int(cfg["n_phase"]),
    ))
    completed = storage.completed_parameters(run_dir) if resume else set()
    for omega_index, omega in enumerate(frequencies):
        n_skip = int(cfg["n_skip_by_frequency"][str(float(omega))])
        for amplitude_index, amplitude in enumerate(amplitudes):
            values = {}
            pending = []
            for seed in range(n_seed):
                key = _point_key(omega_index, amplitude_index, seed)
                checkpoint = storage.checkpoint_path(run_dir, key)
                if key in completed and storage.valid_npz(
                        checkpoint, ("seed", "phase_means")):
                    values[seed] = _load_checkpoint(checkpoint)
                else:
                    pending.append(seed)

            def save_seed(seed, value):
                _, seed_values, cache_stats = value
                key = _point_key(omega_index, amplitude_index, seed)
                storage.save_npz_atomic(
                    storage.checkpoint_path(run_dir, key),
                    seed=int(seed), phase_means=seed_values,
                )
                storage.merge_cache_stats(runtime, cache_stats)
                storage.mark_parameter(run_dir, key, runtime)
                values[int(seed)] = value

            if pending:
                from .parallel import execute_tasks

                def mark_failed(seed, exc):
                    storage.mark_parameter_failed(
                        run_dir, _point_key(omega_index, amplitude_index, seed), exc)

                execute_tasks(
                    pending,
                    amplitude_seed_worker,
                    client=client,
                    worker_kwargs={
                        "omega": float(omega),
                        "amplitude": float(amplitude),
                        "n_skip": n_skip,
                        "cfg": cfg,
                        "combinations": combinations,
                    },
                    on_result=save_seed,
                    on_error=mark_failed,
                )
            for seed in range(n_seed):
                phase_means[omega_index, amplitude_index, :, seed] = values[seed][1]

    odd, even, odd_scaled, even_scaled = _decompose(
        phase_means, amplitudes, combinations, cfg)
    confidence = cfg.get("confidence_level", 0.95)
    odd_mean, odd_se, odd_low, odd_high = mean_se_ci(odd, confidence, axis=2)
    even_mean, even_se, even_low, even_high = mean_se_ci(even, confidence, axis=2)
    odd_scaled_mean, odd_scaled_se, odd_scaled_low, odd_scaled_high = mean_se_ci(
        odd_scaled, confidence, axis=2)
    even_scaled_mean, even_scaled_se, even_scaled_low, even_scaled_high = mean_se_ci(
        even_scaled, confidence, axis=2)
    norm_statistics = {}
    for prefix, values in (
        ("odd_norm", odd),
        ("even_norm", even),
        ("odd_scaled_norm", odd_scaled),
        ("even_scaled_norm", even_scaled),
    ):
        norm_seed = phase_l2(values, axis=-1)
        norm_mean, norm_se, norm_low, norm_high = mean_se_ci(
            norm_seed, confidence, axis=2)
        norm_statistics.update({
            f"{prefix}_seed": norm_seed,
            f"{prefix}_mean": norm_mean,
            f"{prefix}_se": norm_se,
            f"{prefix}_ci_low": norm_low,
            f"{prefix}_ci_high": norm_high,
        })
    odd_fourier_seed = fourier_coefficients(odd, int(cfg["fourier_kmax"]))
    even_fourier_seed = fourier_coefficients(even, int(cfg["fourier_kmax"]))
    odd_fourier_t_test = _signed_fourier_tests(odd_fourier_seed, cfg)
    even_fourier_t_test = _signed_fourier_tests(even_fourier_seed, cfg)
    odd_harmonic_seed = np.sqrt(np.sum(odd_fourier_seed ** 2, axis=-1))
    even_harmonic_seed = np.sqrt(np.sum(even_fourier_seed ** 2, axis=-1))
    odd_harmonic_mean, odd_harmonic_se, odd_harmonic_low, odd_harmonic_high = mean_se_ci(
        odd_harmonic_seed, confidence, axis=2)
    even_harmonic_mean, even_harmonic_se, even_harmonic_low, even_harmonic_high = mean_se_ci(
        even_harmonic_seed, confidence, axis=2)
    odd_leakage_seed = _leakage_ratio(odd_fourier_seed, [1])
    even_leakage_seed = _leakage_ratio(even_fourier_seed, [0, 2])
    odd_leakage_mean, odd_leakage_se, odd_leakage_low, odd_leakage_high = mean_se_ci(
        odd_leakage_seed, confidence, axis=2)
    even_leakage_mean, even_leakage_se, even_leakage_low, even_leakage_high = mean_se_ci(
        even_leakage_seed, confidence, axis=2)
    result = {
        "task": "amplitude-scan",
        "frequencies": frequencies,
        "amplitudes": amplitudes,
        "n_skip_by_frequency": cfg["n_skip_by_frequency"],
        "combinations": combinations,
        "phase_means": phase_means,
        "odd_seed": odd,
        "even_seed": even,
        "odd_scaled_seed": odd_scaled,
        "even_scaled_seed": even_scaled,
        "odd_mean": odd_mean,
        "odd_se": odd_se,
        "odd_ci_low": odd_low,
        "odd_ci_high": odd_high,
        "even_mean": even_mean,
        "even_se": even_se,
        "even_ci_low": even_low,
        "even_ci_high": even_high,
        "odd_scaled_mean": odd_scaled_mean,
        "odd_scaled_se": odd_scaled_se,
        "odd_scaled_ci_low": odd_scaled_low,
        "odd_scaled_ci_high": odd_scaled_high,
        "even_scaled_mean": even_scaled_mean,
        "even_scaled_se": even_scaled_se,
        "even_scaled_ci_low": even_scaled_low,
        "even_scaled_ci_high": even_scaled_high,
        **norm_statistics,
        "odd_fourier_seed": odd_fourier_seed,
        "even_fourier_seed": even_fourier_seed,
        "odd_fourier_t_test": odd_fourier_t_test,
        "even_fourier_t_test": even_fourier_t_test,
        "odd_fft_t_test": odd_fourier_t_test,
        "even_fft_t_test": even_fourier_t_test,
        "odd_harmonic_amplitude_seed": odd_harmonic_seed,
        "even_harmonic_amplitude_seed": even_harmonic_seed,
        "odd_harmonic_amplitude_mean": odd_harmonic_mean,
        "odd_harmonic_amplitude_se": odd_harmonic_se,
        "odd_harmonic_amplitude_ci_low": odd_harmonic_low,
        "odd_harmonic_amplitude_ci_high": odd_harmonic_high,
        "even_harmonic_amplitude_mean": even_harmonic_mean,
        "even_harmonic_amplitude_se": even_harmonic_se,
        "even_harmonic_amplitude_ci_low": even_harmonic_low,
        "even_harmonic_amplitude_ci_high": even_harmonic_high,
        "odd_leakage_seed": odd_leakage_seed,
        "odd_leakage_mean": odd_leakage_mean,
        "odd_leakage_se": odd_leakage_se,
        "odd_leakage_ci_low": odd_leakage_low,
        "odd_leakage_ci_high": odd_leakage_high,
        "even_leakage_seed": even_leakage_seed,
        "even_leakage_mean": even_leakage_mean,
        "even_leakage_se": even_leakage_se,
        "even_leakage_ci_low": even_leakage_low,
        "even_leakage_ci_high": even_leakage_high,
        "sampling_metadata": [forced_sampling_metadata(omega, cfg) for omega in frequencies],
    }
    _write_tables(run_dir, result, cfg)
    _write_figures(run_dir, result, cfg)
    return result
