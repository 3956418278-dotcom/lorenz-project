"""Independent periodic steady-state analysis with finite-seed confidence bands."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import storage
from .fourier import fourier_coefficients
from .response import phase_mean_cached
from .statistics import mean_se_ci


def steady_seed_worker(seed, omega, n_skip, cfg):
    local_runtime = storage.create_runtime()
    forcing = np.asarray(cfg["check_forcing"], dtype=float)
    block_cycles = int(cfg["block_cycles"])
    phase_means = []
    coefficients = []
    for block in range(int(cfg["n_blocks"])):
        value = phase_mean_cached(
            seed,
            forcing,
            omega,
            cfg.get("phase", 0.0),
            int(n_skip) + block * block_cycles,
            block_cycles,
            int(cfg["n_phase"]),
            cfg,
            local_runtime,
        )
        phase_means.append(value)
        coefficients.append(fourier_coefficients(value, int(cfg["fourier_kmax"])))
    return (
        int(seed),
        np.asarray(phase_means),
        np.asarray(coefficients),
        dict(local_runtime["cache"]),
    )


def _load_seed_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return int(data["seed"]), data["phase_means"], data["coefficients"], {
            "hits": 0, "misses": 0
        }


def _point_key(omega_index, n_skip, seed):
    return f"omega:{omega_index}:nskip:{int(n_skip)}:seed:{int(seed)}"


def _recommend_platform(coef_mean, diff_mean, diff_low, diff_high, cfg):
    harmonics = np.asarray(cfg["target_harmonics"], dtype=int)
    convergence = cfg["convergence"]
    stable = []
    max_relative_delta = []
    for skip_index in range(coef_mean.shape[0]):
        selected_mean = np.take(coef_mean[skip_index], harmonics, axis=2)
        selected_diff = np.take(diff_mean[skip_index], harmonics, axis=2)
        selected_low = np.take(diff_low[skip_index], harmonics, axis=2)
        selected_high = np.take(diff_high[skip_index], harmonics, axis=2)
        scale = np.maximum(
            np.maximum(np.abs(selected_mean[1:]), np.abs(selected_mean[:-1])),
            float(convergence.get("scale_floor", 1e-12)),
        )
        bound = float(convergence["atol"]) + float(convergence["rtol"]) * scale
        contains_zero = (selected_low <= 0.0) & (selected_high >= 0.0)
        confidence_magnitude = np.maximum(np.abs(selected_low), np.abs(selected_high))
        within_tolerance = confidence_magnitude <= bound
        require_zero = bool(convergence.get("require_ci_contains_zero", True))
        stable.append(bool(
            (np.all(contains_zero) or not require_zero)
            and np.all(within_tolerance)
        ))
        max_relative_delta.append(float(np.max(np.abs(selected_diff) / scale)))
    minimum = int(convergence["min_consecutive"])
    recommended = None
    interval = None
    for start in range(max(0, len(stable) - minimum + 1)):
        if all(stable[start:start + minimum]):
            recommended = start
            end = start + minimum - 1
            while end + 1 < len(stable) and stable[end + 1]:
                end += 1
            interval = (start, end)
            break
    return stable, max_relative_delta, recommended, interval


def _write_tables(run_dir, result, cfg):
    coefficient_rows = []
    difference_rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for skip_index, n_skip in enumerate(result["candidate_n_skip"]):
            for block in range(int(cfg["n_blocks"])):
                for output in range(3):
                    for harmonic in range(int(cfg["fourier_kmax"]) + 1):
                        for component, name in enumerate(("cos", "sin")):
                            coefficient_rows.append([
                                omega, n_skip, block, output, harmonic, name,
                                result["coefficient_mean"][omega_index, skip_index,
                                                           block, output, harmonic, component],
                                result["coefficient_se"][omega_index, skip_index,
                                                         block, output, harmonic, component],
                                result["coefficient_ci_low"][omega_index, skip_index,
                                                             block, output, harmonic, component],
                                result["coefficient_ci_high"][omega_index, skip_index,
                                                              block, output, harmonic, component],
                            ])
            for difference in range(int(cfg["n_blocks"]) - 1):
                for output in range(3):
                    for harmonic in range(int(cfg["fourier_kmax"]) + 1):
                        for component, name in enumerate(("cos", "sin")):
                            difference_rows.append([
                                omega, n_skip, difference, difference + 1,
                                output, harmonic, name,
                                result["block_difference_mean"][omega_index, skip_index,
                                                                 difference, output,
                                                                 harmonic, component],
                                result["block_difference_se"][omega_index, skip_index,
                                                               difference, output,
                                                               harmonic, component],
                                result["block_difference_ci_low"][omega_index, skip_index,
                                                                   difference, output,
                                                                   harmonic, component],
                                result["block_difference_ci_high"][omega_index, skip_index,
                                                                    difference, output,
                                                                    harmonic, component],
                            ])
    storage.write_csv(
        run_dir / "tables" / "steady_fourier_coefficients.csv",
        ["omega", "n_skip", "block", "output", "harmonic", "component",
         "mean", "se", "ci_low", "ci_high"],
        coefficient_rows,
    )
    storage.write_csv(
        run_dir / "tables" / "steady_block_differences.csv",
        ["omega", "n_skip", "block_from", "block_to", "output", "harmonic",
         "component", "mean", "se", "ci_low", "ci_high"],
        difference_rows,
    )
    summary_rows = []
    for index, omega in enumerate(result["frequencies"]):
        recommendation = result["recommendations"][str(float(omega))]
        summary_rows.append([
            omega,
            recommendation["converged"],
            recommendation["recommended_n_skip"],
            recommendation["platform_n_skip_min"],
            recommendation["platform_n_skip_max"],
            "|".join(str(value) for value in recommendation["stable_flags"]),
            "|".join(f"{value:.8g}" for value in recommendation["max_relative_delta"]),
        ])
    storage.write_csv(
        run_dir / "tables" / "steady_summary.csv",
        ["omega", "converged", "recommended_n_skip", "platform_n_skip_min",
         "platform_n_skip_max", "stable_flags", "max_relative_delta"],
        summary_rows,
    )


def _write_figures(run_dir, result, cfg):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    labels = ("x", "y", "z")
    confidence = 100.0 * float(cfg.get("confidence_level", 0.95))
    x = np.asarray(result["candidate_n_skip"], dtype=float)
    for omega_index, omega in enumerate(result["frequencies"]):
        coefficient_path = run_dir / "figures" / f"steady_coefficients_omega_{omega_index}.pdf"
        with PdfPages(coefficient_path) as pdf:
            for output in range(3):
                for harmonic in cfg["target_harmonics"]:
                    for component, name in enumerate(("cos", "sin")):
                        fig, axis = plt.subplots(figsize=(8, 5))
                        for block in range(int(cfg["n_blocks"])):
                            mean = result["coefficient_mean"][omega_index, :, block,
                                                                output, harmonic, component]
                            low = result["coefficient_ci_low"][omega_index, :, block,
                                                                 output, harmonic, component]
                            high = result["coefficient_ci_high"][omega_index, :, block,
                                                                  output, harmonic, component]
                            axis.plot(x, mean, marker="o", label=f"block {block}")
                            axis.fill_between(x, low, high, alpha=0.15)
                        axis.set_title(
                            f"omega={omega:.6g}, {labels[output]}, k={harmonic} {name}\n"
                            f"seed mean and {confidence:.1f}% t confidence bands"
                        )
                        axis.set_xlabel("n_skip")
                        axis.set_ylabel("Fourier coefficient")
                        axis.legend(frameon=False)
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
        difference_path = run_dir / "figures" / f"steady_block_differences_omega_{omega_index}.pdf"
        with PdfPages(difference_path) as pdf:
            for output in range(3):
                for harmonic in cfg["target_harmonics"]:
                    for component, name in enumerate(("cos", "sin")):
                        fig, axis = plt.subplots(figsize=(8, 5))
                        for difference in range(int(cfg["n_blocks"]) - 1):
                            mean = result["block_difference_mean"][
                                omega_index, :, difference, output, harmonic, component]
                            low = result["block_difference_ci_low"][
                                omega_index, :, difference, output, harmonic, component]
                            high = result["block_difference_ci_high"][
                                omega_index, :, difference, output, harmonic, component]
                            axis.plot(x, mean, marker="o",
                                      label=f"block {difference + 1} - {difference}")
                            axis.fill_between(x, low, high, alpha=0.15)
                        axis.axhline(0.0, color="black", linewidth=0.8)
                        axis.set_title(
                            f"omega={omega:.6g}, adjacent-block difference, "
                            f"{labels[output]}, k={harmonic} {name}"
                        )
                        axis.set_xlabel("n_skip")
                        axis.set_ylabel("coefficient difference")
                        axis.legend(frameon=False)
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)


def run(cfg, run_dir, runtime, resume=False, client=None):
    run_dir = Path(run_dir)
    frequencies = np.asarray(cfg["frequencies"], dtype=float)
    candidate_n_skip = np.asarray(cfg["candidate_n_skip"], dtype=int)
    n_omega = len(frequencies)
    n_skip = len(candidate_n_skip)
    n_blocks = int(cfg["n_blocks"])
    n_seed = int(cfg["n_seed"])
    n_phase = int(cfg["n_phase"])
    k_count = int(cfg["fourier_kmax"]) + 1
    phase_means = np.empty((n_omega, n_skip, n_blocks, n_seed, 3, n_phase))
    coefficient_seed = np.empty((n_omega, n_skip, n_blocks, n_seed, 3, k_count, 2))
    completed = storage.completed_parameters(run_dir) if resume else set()

    for omega_index, omega in enumerate(frequencies):
        for skip_index, skip in enumerate(candidate_n_skip):
            values = {}
            pending = []
            for seed in range(n_seed):
                key = _point_key(omega_index, skip, seed)
                checkpoint = storage.checkpoint_path(run_dir, key)
                if key in completed and storage.valid_npz(
                        checkpoint, ("seed", "phase_means", "coefficients")):
                    values[seed] = _load_seed_checkpoint(checkpoint)
                else:
                    pending.append(seed)

            def save_seed(seed, value):
                _, seed_phase, seed_coef, cache_stats = value
                key = _point_key(omega_index, skip, seed)
                checkpoint = storage.checkpoint_path(run_dir, key)
                storage.save_npz_atomic(
                    checkpoint,
                    seed=int(seed),
                    phase_means=seed_phase,
                    coefficients=seed_coef,
                )
                storage.merge_cache_stats(runtime, cache_stats)
                storage.mark_parameter(run_dir, key, runtime)
                values[int(seed)] = value

            if pending:
                from .parallel import execute_tasks

                def mark_failed(seed, exc):
                    storage.mark_parameter_failed(
                        run_dir, _point_key(omega_index, skip, seed), exc)

                execute_tasks(
                    pending,
                    steady_seed_worker,
                    client=client,
                    worker_kwargs={"omega": float(omega), "n_skip": int(skip), "cfg": cfg},
                    on_result=save_seed,
                    on_error=mark_failed,
                )
            for seed in range(n_seed):
                phase_means[omega_index, skip_index, :, seed] = values[seed][1]
                coefficient_seed[omega_index, skip_index, :, seed] = values[seed][2]

    coef_mean, coef_se, coef_low, coef_high = mean_se_ci(
        coefficient_seed, cfg.get("confidence_level", 0.95), axis=3)
    difference_seed = np.diff(coefficient_seed, axis=2)
    diff_mean, diff_se, diff_low, diff_high = mean_se_ci(
        difference_seed, cfg.get("confidence_level", 0.95), axis=3)
    recommendations = {}
    for omega_index, omega in enumerate(frequencies):
        stable, max_delta, recommended_index, interval = _recommend_platform(
            coef_mean[omega_index], diff_mean[omega_index], diff_low[omega_index],
            diff_high[omega_index], cfg)
        recommendations[str(float(omega))] = {
            "converged": recommended_index is not None,
            "recommended_n_skip": (
                int(candidate_n_skip[recommended_index])
                if recommended_index is not None else None
            ),
            "platform_n_skip_min": (
                int(candidate_n_skip[interval[0]]) if interval is not None else None
            ),
            "platform_n_skip_max": (
                int(candidate_n_skip[interval[1]]) if interval is not None else None
            ),
            "stable_flags": stable,
            "max_relative_delta": max_delta,
            "confidence_level": float(cfg.get("confidence_level", 0.95)),
        }
    result = {
        "task": "steady",
        "frequencies": frequencies,
        "candidate_n_skip": candidate_n_skip,
        "phase_means": phase_means,
        "coefficient_seed": coefficient_seed,
        "coefficient_mean": coef_mean,
        "coefficient_se": coef_se,
        "coefficient_ci_low": coef_low,
        "coefficient_ci_high": coef_high,
        "block_difference_seed": difference_seed,
        "block_difference_mean": diff_mean,
        "block_difference_se": diff_se,
        "block_difference_ci_low": diff_low,
        "block_difference_ci_high": diff_high,
        "recommendations": recommendations,
        "sampling_metadata": cfg["sampling_metadata"],
    }
    _write_tables(run_dir, result, cfg)
    _write_figures(run_dir, result, cfg)
    return result
