"""Cached phase sampling and independent multi-model L/H extrapolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import core, storage
from .fourier import amplitude_phase, fourier_coefficients
from .statistics import mean_se_ci, phase_l2


H_INDEPENDENT_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def get_spinup_state(seed, cfg, runtime):
    payload = {
        "seed": int(seed),
        "lorenz": cfg["lorenz"],
        "T_spinup": cfg["T_spinup"],
        "solver": cfg["solver"],
        "kind": "spinup_v1",
    }
    key = storage.cache_key(payload)
    path = storage.spinup_cache_path(key)
    try:
        with storage.load_npz(path) as data:
            state = data["state"].copy()
        runtime["cache"]["hits"] += 1
        return state
    except Exception:
        runtime["cache"]["misses"] += 1
    state = core.spinup_state(int(seed), cfg)
    storage.save_npz_atomic(path, state=state,
                            payload=json.dumps(payload, sort_keys=True))
    return state


def phase_mean_cached(seed, forcing_vector, omega, phase, n_skip, n_cycle,
                      n_phase, cfg, runtime):
    payload = {
        "seed": int(seed),
        "forcing": np.asarray(forcing_vector, dtype=float).tolist(),
        "omega": float(omega),
        "phase": float(phase),
        "n_skip": int(n_skip),
        "n_cycle": int(n_cycle),
        "n_phase": int(n_phase),
        "T_spinup": cfg["T_spinup"],
        "lorenz": cfg["lorenz"],
        "solver": cfg["solver"],
        "kind": "phase_mean_v1",
    }
    key = storage.cache_key(payload)
    path = storage.phase_cache_path(key)
    try:
        with storage.load_npz(path) as data:
            phase_mean = data["phase_mean"].copy()
        runtime["cache"]["hits"] += 1
        return phase_mean
    except Exception:
        runtime["cache"]["misses"] += 1
    state = get_spinup_state(seed, cfg, runtime)
    samples = core.simulate_phase_samples(
        int(seed), forcing_vector, float(omega), float(phase), int(n_skip),
        int(n_cycle), int(n_phase), cfg, initial_state=state)
    phase_mean = samples.mean(axis=1)
    storage.save_npz_atomic(path, phase_mean=phase_mean,
                            payload=json.dumps(payload, sort_keys=True))
    return phase_mean


def _combination_index(combinations):
    return {item["name"]: index for index, item in enumerate(combinations)}


def finite_difference_estimates(amplitude_result):
    amplitudes = np.asarray(amplitude_result["amplitudes"], dtype=float)
    linear = np.asarray(amplitude_result["odd_scaled_seed"], dtype=float)
    raw = np.asarray(amplitude_result["phase_means"], dtype=float)
    combinations = amplitude_result["combinations"]
    index = _combination_index(combinations)
    n_omega, n_amp, _, n_seed, _, n_phase = raw.shape
    quadratic = np.full((n_omega, n_amp, n_seed, 3, 3, 3, n_phase), np.nan)
    for direction in range(3):
        quadratic[:, :, :, :, direction, direction] = np.asarray(
            amplitude_result["even_scaled_seed"][:, :, :, :, direction])
    for first, second in ((0, 1), (0, 2), (1, 2)):
        pp = raw[:, :, index[f"pair_{first}_{second}_pp"]]
        pm = raw[:, :, index[f"pair_{first}_{second}_pm"]]
        mp = raw[:, :, index[f"pair_{first}_{second}_mp"]]
        mm = raw[:, :, index[f"pair_{first}_{second}_mm"]]
        shape = (1, n_amp, 1, 1, 1)
        mixed = (pp - pm - mp + mm) / (
            4.0 * amplitudes.reshape(shape) ** 2)
        quadratic[:, :, :, :, first, second] = mixed
        quadratic[:, :, :, :, second, first] = mixed
    if not np.isfinite(linear).all() or not np.isfinite(quadratic).all():
        raise ValueError(
            "amplitude run does not contain all three forcing directions and mixed pairs"
        )
    return linear, quadratic


def _design(amplitudes, degree):
    x = np.asarray(amplitudes, dtype=float) ** 2
    return np.stack([x ** power for power in range(int(degree) + 1)], axis=1)


def _fit_coefficients(values, design):
    """values shape (amplitude, ..., phase); return (parameter, ..., phase)."""
    flat = np.asarray(values).reshape(values.shape[0], -1)
    coefficients = np.linalg.lstsq(design, flat, rcond=None)[0]
    return coefficients.reshape((design.shape[1],) + values.shape[1:])


def _fit_one_model(seed_values, amplitudes, degree, cfg):
    design = _design(amplitudes, degree)
    n_amplitude, n_seed = seed_values.shape[:2]
    parameter_count = design.shape[1]
    if n_amplitude < parameter_count:
        raise ValueError(f"degree {degree} needs at least {parameter_count} amplitudes")
    seed_coefficients = np.asarray([
        _fit_coefficients(seed_values[:, seed], design)
        for seed in range(n_seed)
    ])
    coefficient_mean, coefficient_se, coefficient_low, coefficient_high = mean_se_ci(
        seed_coefficients, cfg.get("confidence_level", 0.95), axis=0)
    mean_values = seed_values.mean(axis=1)
    fitted_mean = np.tensordot(design, coefficient_mean, axes=(1, 0))
    residual_mean = mean_values - fitted_mean
    rss_by_element = np.sum(residual_mean ** 2, axis=0)
    centered = mean_values - mean_values.mean(axis=0, keepdims=True)
    tss_by_element = np.sum(centered ** 2, axis=0)
    r_squared = 1.0 - rss_by_element / np.maximum(tss_by_element, 1e-300)
    rss = float(np.sum(residual_mean ** 2))
    observation_count = int(np.prod(mean_values.shape))
    bic = observation_count * np.log(max(rss / observation_count, 1e-300))
    bic += parameter_count * np.log(observation_count)

    loo_prediction = np.full_like(mean_values, np.nan)
    if n_amplitude - 1 >= parameter_count:
        for held_out in range(n_amplitude):
            keep = np.arange(n_amplitude) != held_out
            coefficients = _fit_coefficients(mean_values[keep], design[keep])
            loo_prediction[held_out] = np.tensordot(
                design[held_out], coefficients, axes=(0, 0))
        loo_residual = mean_values - loo_prediction
        loo_rmse = float(np.sqrt(np.mean(loo_residual ** 2)))
    else:
        loo_residual = np.full_like(mean_values, np.nan)
        loo_rmse = float("inf")

    rng = np.random.default_rng(int(cfg["random_seed"]) + 1009 * int(degree))
    bootstrap_coefficients = []
    for _ in range(int(cfg["bootstrap_samples"])):
        indices = rng.integers(0, n_seed, n_seed)
        bootstrap_coefficients.append(seed_coefficients[indices].mean(axis=0))
    bootstrap_coefficients = np.asarray(bootstrap_coefficients)
    alpha = 100.0 * (1.0 - float(cfg.get("confidence_level", 0.95))) / 2.0
    bootstrap_low = np.percentile(bootstrap_coefficients, alpha, axis=0)
    bootstrap_high = np.percentile(bootstrap_coefficients, 100.0 - alpha, axis=0)

    largest = int(np.argmax(amplitudes))
    smallest = int(np.argmin(amplitudes))
    sensitivity = {}
    for label, removed in (("drop_largest", largest), ("drop_smallest", smallest)):
        keep = np.arange(n_amplitude) != removed
        if int(np.sum(keep)) >= parameter_count:
            reduced = _fit_coefficients(mean_values[keep], design[keep])
            delta = reduced[0] - coefficient_mean[0]
            sensitivity[label] = {
                "removed_amplitude": float(amplitudes[removed]),
                "intercept": reduced[0],
                "relative_change_l2": float(
                    np.linalg.norm(delta) /
                    max(float(np.linalg.norm(coefficient_mean[0])), 1e-300)
                ),
            }
        else:
            sensitivity[label] = {
                "removed_amplitude": float(amplitudes[removed]),
                "intercept": None,
                "relative_change_l2": None,
                "reason": "insufficient remaining amplitudes for this model",
            }
    return {
        "degree": int(degree),
        "powers": [2 * power for power in range(parameter_count)],
        "seed_coefficients": seed_coefficients,
        "coefficient_mean": coefficient_mean,
        "coefficient_se": coefficient_se,
        "coefficient_ci_low": coefficient_low,
        "coefficient_ci_high": coefficient_high,
        "bootstrap_ci_low": bootstrap_low,
        "bootstrap_ci_high": bootstrap_high,
        "fitted_mean": fitted_mean,
        "residual_mean": residual_mean,
        "rss_by_element": rss_by_element,
        "r_squared": r_squared,
        "rss": rss,
        "bic": float(bic),
        "loo_prediction": loo_prediction,
        "loo_residual": loo_residual,
        "loo_rmse": loo_rmse,
        "amplitude_sensitivity": sensitivity,
    }


def fit_models(seed_values, amplitudes, cfg):
    if len(amplitudes) < 4:
        raise ValueError(
            "at least four included amplitudes are required to compare degree-0/1/2 "
            "models and perform leave-one-amplitude diagnostics"
        )
    models = {
        int(degree): _fit_one_model(seed_values, amplitudes, int(degree), cfg)
        for degree in cfg["fit_degrees"]
    }
    metric = cfg["selection_metric"]
    selected = min(models, key=lambda degree: float(models[degree][metric]))
    return models, int(selected)


def _amplitude_selection(amplitudes, cfg):
    amplitudes = np.asarray(amplitudes, dtype=float)
    minimum = float(cfg["amplitude_min"])
    maximum = float(cfg["amplitude_max"])
    used = np.flatnonzero((amplitudes >= minimum) & (amplitudes <= maximum))
    excluded = []
    for index, amplitude in enumerate(amplitudes):
        if index not in used:
            reason = "below amplitude_min" if amplitude < minimum else "above amplitude_max"
            excluded.append({"index": index, "amplitude": float(amplitude), "reason": reason})
    if len(used) < 4:
        raise ValueError(
            f"response amplitude selection retained {len(used)} points; at least four required"
        )
    return used, excluded


def _write_tables(run_dir, result):
    rows = []
    for omega_index, omega in enumerate(result["frequencies"]):
        for quantity, all_models, selected in (
            ("L", result["L_models"], result["selected_L_degree"]),
            ("H", result["H_models"], result["selected_H_degree"]),
        ):
            for degree, model in all_models[omega_index].items():
                rows.append([
                    omega, quantity, degree, int(degree) == int(selected[omega_index]),
                    model["rss"], float(np.nanmean(model["r_squared"])),
                    model["bic"], model["loo_rmse"],
                    model["amplitude_sensitivity"]["drop_largest"]["relative_change_l2"],
                    model["amplitude_sensitivity"]["drop_smallest"]["relative_change_l2"],
                ])
    storage.write_csv(
        run_dir / "tables" / "response_model_comparison.csv",
        ["omega", "quantity", "degree_in_a2", "selected", "rss", "mean_r2",
         "bic", "loo_rmse", "drop_largest_relative_change",
         "drop_smallest_relative_change"],
        rows,
    )
    exclusion_rows = [[item["index"], item["amplitude"], item["reason"]]
                      for item in result["excluded_amplitudes"]]
    storage.write_csv(
        run_dir / "tables" / "amplitude_selection.csv",
        ["index", "amplitude", "reason"],
        exclusion_rows,
    )


def _component_page(pdf, amplitudes, finite_values, models, selected_degree,
                    response_mean, response_low, response_high, title, harmonics):
    import matplotlib.pyplot as plt

    x = amplitudes ** 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    finite_norm = phase_l2(finite_values, axis=-1)
    axes[0, 0].plot(x, finite_norm, "o-", label="finite difference")
    for degree, model in models.items():
        axes[0, 0].plot(x, phase_l2(model["fitted_mean"], axis=-1),
                        label=f"degree {degree}")
        axes[0, 1].plot(x, phase_l2(model["residual_mean"], axis=-1),
                        marker="o", label=f"degree {degree}")
    axes[0, 0].set_title("finite-amplitude estimate and fitted models")
    axes[0, 1].set_title("fit residual L2")
    theta = np.linspace(0.0, 2.0 * np.pi, response_mean.size, endpoint=False)
    axes[1, 0].plot(theta, response_mean, label=f"selected degree {selected_degree}")
    axes[1, 0].fill_between(theta, response_low, response_high, alpha=0.25)
    axes[1, 0].set_title("extrapolated phase response with seed t-CI")
    axes[1, 1].bar(np.arange(len(harmonics)), harmonics)
    axes[1, 1].set_title("extrapolated harmonic amplitudes")
    for axis in axes.ravel():
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _write_figures(run_dir, result):
    from matplotlib.backends.backend_pdf import PdfPages

    amplitudes = np.asarray(result["used_amplitudes"])
    labels = ("x", "y", "z")
    for omega_index, omega in enumerate(result["frequencies"]):
        with PdfPages(run_dir / "figures" / f"response_L_omega_{omega_index}.pdf") as pdf:
            for output in range(3):
                for direction in range(3):
                    models = {
                        degree: {
                            **model,
                            "fitted_mean": model["fitted_mean"][:, output, direction],
                            "residual_mean": model["residual_mean"][:, output, direction],
                        }
                        for degree, model in result["L_models"][omega_index].items()
                    }
                    _component_page(
                        pdf, amplitudes,
                        result["L_finite_mean"][omega_index, :, output, direction],
                        models, result["selected_L_degree"][omega_index],
                        result["L_mean"][omega_index, output, direction],
                        result["L_ci_low"][omega_index, output, direction],
                        result["L_ci_high"][omega_index, output, direction],
                        f"omega={omega:.6g}: L[{labels[output]},{labels[direction]}]",
                        result["L_harmonic_amplitude_mean"][omega_index, output, direction],
                    )
        with PdfPages(run_dir / "figures" / f"response_H_omega_{omega_index}.pdf") as pdf:
            for output in range(3):
                for first, second in H_INDEPENDENT_PAIRS:
                    models = {
                        degree: {
                            **model,
                            "fitted_mean": model["fitted_mean"][:, output, first, second],
                            "residual_mean": model["residual_mean"][:, output, first, second],
                        }
                        for degree, model in result["H_models"][omega_index].items()
                    }
                    _component_page(
                        pdf, amplitudes,
                        result["H_finite_mean"][omega_index, :, output, first, second],
                        models, result["selected_H_degree"][omega_index],
                        result["H_mean"][omega_index, output, first, second],
                        result["H_ci_low"][omega_index, output, first, second],
                        result["H_ci_high"][omega_index, output, first, second],
                        f"omega={omega:.6g}: H[{labels[output]},{labels[first]},{labels[second]}]",
                        result["H_harmonic_amplitude_mean"][omega_index, output, first, second],
                    )


def run(cfg, amplitude_result, run_dir, runtime, resume=False):
    run_dir = Path(run_dir)
    amplitudes = np.asarray(amplitude_result["amplitudes"], dtype=float)
    used_indices, excluded = _amplitude_selection(amplitudes, cfg)
    used_amplitudes = amplitudes[used_indices]
    linear_finite, quadratic_finite = finite_difference_estimates(amplitude_result)
    linear_used = linear_finite[:, used_indices]
    quadratic_used = quadratic_finite[:, used_indices]
    confidence = cfg.get("confidence_level", 0.95)
    linear_mean, linear_se, linear_low, linear_high = mean_se_ci(
        linear_used, confidence, axis=2)
    quadratic_mean, quadratic_se, quadratic_low, quadratic_high = mean_se_ci(
        quadratic_used, confidence, axis=2)
    L_models = []
    H_models = []
    selected_L = []
    selected_H = []
    completed = storage.completed_parameters(run_dir) if resume else set()
    for omega_index in range(len(amplitude_result["frequencies"])):
        key = f"omega:{omega_index}"
        checkpoint = storage.checkpoint_path(run_dir, key)
        if key in completed and storage.valid_npz(checkpoint, ("result",)):
            with np.load(checkpoint, allow_pickle=True) as data:
                point = data["result"].item()
        else:
            models, selected = fit_models(
                linear_used[omega_index], used_amplitudes, cfg)
            linear_models = {degree: model for degree, model in models.items()}
            models, selected_quadratic = fit_models(
                quadratic_used[omega_index], used_amplitudes, cfg)
            point = {
                "L_models": linear_models,
                "H_models": {degree: model for degree, model in models.items()},
                "selected_L": int(selected),
                "selected_H": int(selected_quadratic),
            }
            storage.save_npz_atomic(checkpoint, result=np.array(point, dtype=object))
            storage.mark_parameter(run_dir, key, runtime)
        L_models.append(point["L_models"])
        H_models.append(point["H_models"])
        selected_L.append(point["selected_L"])
        selected_H.append(point["selected_H"])
    L_seed = np.asarray([
        L_models[index][selected_L[index]]["seed_coefficients"][:, 0]
        for index in range(len(L_models))
    ])
    H_seed = np.asarray([
        H_models[index][selected_H[index]]["seed_coefficients"][:, 0]
        for index in range(len(H_models))
    ])
    L_mean, L_se, L_low, L_high = mean_se_ci(L_seed, confidence, axis=1)
    H_mean, H_se, H_low, H_high = mean_se_ci(H_seed, confidence, axis=1)
    L_fourier_seed = fourier_coefficients(L_seed, int(cfg["fourier_kmax"]))
    H_fourier_seed = fourier_coefficients(H_seed, int(cfg["fourier_kmax"]))
    L_harmonic_seed = amplitude_phase(L_fourier_seed)[0]
    H_harmonic_seed = amplitude_phase(H_fourier_seed)[0]
    L_harmonic_mean, L_harmonic_se, L_harmonic_low, L_harmonic_high = mean_se_ci(
        L_harmonic_seed, confidence, axis=1)
    H_harmonic_mean, H_harmonic_se, H_harmonic_low, H_harmonic_high = mean_se_ci(
        H_harmonic_seed, confidence, axis=1)
    result = {
        "task": "response",
        "frequencies": np.asarray(amplitude_result["frequencies"]),
        "all_amplitudes": amplitudes,
        "used_amplitude_indices": used_indices,
        "used_amplitudes": used_amplitudes,
        "excluded_amplitudes": excluded,
        "L_finite_seed": linear_used,
        "H_finite_seed": quadratic_used,
        "L_finite_mean": linear_mean,
        "L_finite_se": linear_se,
        "L_finite_ci_low": linear_low,
        "L_finite_ci_high": linear_high,
        "H_finite_mean": quadratic_mean,
        "H_finite_se": quadratic_se,
        "H_finite_ci_low": quadratic_low,
        "H_finite_ci_high": quadratic_high,
        "L_models": L_models,
        "H_models": H_models,
        "selected_L_degree": np.asarray(selected_L, dtype=int),
        "selected_H_degree": np.asarray(selected_H, dtype=int),
        "L_seed": L_seed,
        "L_mean": L_mean,
        "L_se": L_se,
        "L_ci_low": L_low,
        "L_ci_high": L_high,
        "H_seed": H_seed,
        "H_mean": H_mean,
        "H_se": H_se,
        "H_ci_low": H_low,
        "H_ci_high": H_high,
        "L_fourier_seed": L_fourier_seed,
        "H_fourier_seed": H_fourier_seed,
        "L_harmonic_amplitude_seed": L_harmonic_seed,
        "H_harmonic_amplitude_seed": H_harmonic_seed,
        "L_harmonic_amplitude_mean": L_harmonic_mean,
        "L_harmonic_amplitude_se": L_harmonic_se,
        "L_harmonic_amplitude_ci_low": L_harmonic_low,
        "L_harmonic_amplitude_ci_high": L_harmonic_high,
        "H_harmonic_amplitude_mean": H_harmonic_mean,
        "H_harmonic_amplitude_se": H_harmonic_se,
        "H_harmonic_amplitude_ci_low": H_harmonic_low,
        "H_harmonic_amplitude_ci_high": H_harmonic_high,
        "sampling_metadata": amplitude_result["sampling_metadata"],
    }
    _write_tables(run_dir, result)
    _write_figures(run_dir, result)
    return result
