"""Small independent convergence scan for solver, sampling, FFT, and Welch choices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import integrate, signal

from . import core, storage
from .fourier import fourier_coefficients
from .response import get_spinup_state, phase_mean_cached
from .statistics import mean_se_ci


TIME_PARAMETERS = ("rtol", "atol", "T_spinup", "n_phase", "n_cycle")
SPECTRAL_PARAMETERS = ("fft_length", "welch_segment_length")


def _cases(cfg):
    cases = [
        {"case_id": "reference:time", "family": "time", "parameter": "reference",
         "value": "reference"},
        {"case_id": "reference:spectrum", "family": "spectrum",
         "parameter": "reference", "value": "reference"},
    ]
    for parameter in TIME_PARAMETERS + SPECTRAL_PARAMETERS:
        family = "time" if parameter in TIME_PARAMETERS else "spectrum"
        for index, value in enumerate(cfg["scan"][parameter]):
            cases.append({
                "case_id": f"{parameter}:{index}",
                "family": family,
                "parameter": parameter,
                "value": value,
            })
    return cases


def _case_config(case, cfg):
    reference = cfg["reference"]
    local = {
        "lorenz": dict(cfg["lorenz"]),
        "solver": dict(cfg["solver"]),
        "T_spinup": float(reference["T_spinup"]),
        "phase": float(cfg.get("phase", 0.0)),
    }
    local["solver"]["rtol"] = float(reference["rtol"])
    local["solver"]["atol"] = float(reference["atol"])
    local.update({
        "n_phase": int(reference["n_phase"]),
        "n_record_cycles": int(reference["n_cycle"]),
        "fft_length": int(reference["fft_length"]),
        "welch_segment_length": int(reference["welch_segment_length"]),
    })
    parameter = case["parameter"]
    if parameter == "rtol":
        local["solver"]["rtol"] = float(case["value"])
    elif parameter == "atol":
        local["solver"]["atol"] = float(case["value"])
    elif parameter == "T_spinup":
        local["T_spinup"] = float(case["value"])
    elif parameter == "n_phase":
        local["n_phase"] = int(case["value"])
    elif parameter == "n_cycle":
        local["n_record_cycles"] = int(case["value"])
    elif parameter == "fft_length":
        local["fft_length"] = int(case["value"])
    elif parameter == "welch_segment_length":
        local["welch_segment_length"] = int(case["value"])
    return local


def sampling_case_worker(item, cases, cfg):
    case_index, seed = item
    case = cases[int(case_index)]
    local = _case_config(case, cfg)
    local_runtime = storage.create_runtime()
    vectors = []
    if case["family"] == "time":
        forcing = np.asarray(cfg["check_forcing"], dtype=float)
        for omega in cfg["frequencies"]:
            for amplitude in cfg["amplitudes"]:
                direction = forcing / max(float(np.linalg.norm(forcing)), 1e-300)
                value = phase_mean_cached(
                    int(seed),
                    float(amplitude) * direction,
                    float(omega),
                    cfg.get("phase", 0.0),
                    int(cfg["n_skip"]),
                    int(local["n_record_cycles"]),
                    int(local["n_phase"]),
                    local,
                    local_runtime,
                )
                coefficients = fourier_coefficients(value, int(cfg["fourier_kmax"]))
                vectors.append(coefficients.ravel())
    else:
        state = get_spinup_state(int(seed), local, local_runtime)
        count = int(cfg["reference"]["n_record_samples"])
        sample_rate = float(cfg["reference"]["sample_rate"])
        times = np.arange(count, dtype=float) / sample_rate
        solution = integrate.solve_ivp(
            lambda time, state_: core.lorenz_rhs(time, state_, local),
            [0.0, float(times[-1])],
            state,
            t_eval=times,
            **local["solver"],
        )
        core.check_solution(solution, (3, count))
        frequencies, psd = signal.welch(
            solution.y,
            fs=sample_rate,
            window="hann",
            nperseg=int(local["welch_segment_length"]),
            noverlap=min(
                int(cfg["reference"]["welch_overlap_samples"]),
                int(local["welch_segment_length"]) - 1,
            ),
            nfft=int(local["fft_length"]),
            detrend="constant",
            axis=-1,
            scaling="density",
        )
        evaluation = np.asarray(cfg["spectral_evaluation_frequencies"], dtype=float)
        for output in range(3):
            vectors.append(np.interp(evaluation, frequencies, psd[output]))
    return (
        str(case["case_id"]),
        int(seed),
        np.concatenate(vectors),
        dict(local_runtime["cache"]),
    )


def _checkpoint_key(case_id, seed):
    return f"case:{case_id}:seed:{int(seed)}"


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return str(data["case_id"]), int(data["seed"]), data["vector"], {
            "hits": 0, "misses": 0
        }


def _write_outputs(run_dir, result, cfg):
    rows = []
    for index, case in enumerate(result["cases"]):
        rows.append([
            case["case_id"], case["family"], case["parameter"], case["value"],
            result["relative_error_mean"][index], result["relative_error_se"][index],
            result["relative_error_ci_low"][index],
            result["relative_error_ci_high"][index],
            bool(result["converged"][index]),
        ])
    storage.write_csv(
        run_dir / "tables" / "sampling_convergence.csv",
        ["case_id", "family", "parameter", "value", "relative_error_mean",
         "relative_error_se", "ci_low", "ci_high", "converged"],
        rows,
    )
    storage.write_json(run_dir / "data" / "recommended_parameters.json",
                       result["recommendations"])

    import matplotlib.pyplot as plt

    parameters = list(TIME_PARAMETERS + SPECTRAL_PARAMETERS)
    fig, axes = plt.subplots(4, 2, figsize=(10, 13))
    for axis, parameter in zip(axes.ravel(), parameters):
        indices = [i for i, case in enumerate(result["cases"])
                   if case["parameter"] == parameter]
        x = np.arange(len(indices))
        mean = result["relative_error_mean"][indices]
        low = result["relative_error_ci_low"][indices]
        high = result["relative_error_ci_high"][indices]
        axis.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), marker="o")
        axis.axhline(float(cfg["relative_tolerance"]), color="tab:red",
                     linestyle="--", label="configured tolerance")
        axis.set_xticks(x)
        axis.set_xticklabels([str(result["cases"][i]["value"]) for i in indices],
                             rotation=35, ha="right")
        axis.set_title(parameter)
        axis.set_ylabel("relative error")
        axis.set_yscale("log")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Sampling and solver convergence (seed confidence intervals)")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "sampling_convergence.pdf")
    fig.savefig(run_dir / "figures" / "sampling_convergence.png", dpi=160)
    plt.close(fig)


def run(cfg, run_dir, runtime, resume=False, client=None):
    run_dir = Path(run_dir)
    cases = _cases(cfg)
    completed = storage.completed_parameters(run_dir) if resume else set()
    values = {}
    pending = []
    for case_index, case in enumerate(cases):
        for seed in range(int(cfg["n_seed"])):
            key = _checkpoint_key(case["case_id"], seed)
            checkpoint = storage.checkpoint_path(run_dir, key)
            if key in completed and storage.valid_npz(
                    checkpoint, ("case_id", "seed", "vector")):
                loaded = _load_checkpoint(checkpoint)
                values[(case["case_id"], seed)] = loaded
            else:
                pending.append((case_index, seed))

    def save_case(item, value):
        _, seed = item
        case_id, _, vector, cache_stats = value
        key = _checkpoint_key(case_id, seed)
        storage.save_npz_atomic(
            storage.checkpoint_path(run_dir, key),
            case_id=np.asarray(case_id), seed=int(seed), vector=vector,
        )
        storage.merge_cache_stats(runtime, cache_stats)
        storage.mark_parameter(run_dir, key, runtime)
        values[(case_id, int(seed))] = value

    if pending:
        from .parallel import execute_tasks

        def mark_failed(item, exc):
            case_index, seed = item
            storage.mark_parameter_failed(
                run_dir, _checkpoint_key(cases[case_index]["case_id"], seed), exc)

        execute_tasks(
            pending,
            sampling_case_worker,
            client=client,
            worker_kwargs={"cases": cases, "cfg": cfg},
            on_result=save_case,
            on_error=mark_failed,
            key_string=lambda item: f"{item[0]}:{item[1]}",
        )
    errors = np.zeros((len(cases), int(cfg["n_seed"])), dtype=float)
    reference_ids = {"time": "reference:time", "spectrum": "reference:spectrum"}
    for case_index, case in enumerate(cases):
        reference_id = reference_ids[case["family"]]
        for seed in range(int(cfg["n_seed"])):
            vector = values[(case["case_id"], seed)][2]
            reference = values[(reference_id, seed)][2]
            errors[case_index, seed] = np.linalg.norm(vector - reference) / max(
                float(np.linalg.norm(reference)), 1e-300)
    mean, se, low, high = mean_se_ci(
        errors, cfg.get("confidence_level", 0.95), axis=1)
    converged = high <= float(cfg["relative_tolerance"])
    recommendations = {}
    for parameter in TIME_PARAMETERS + SPECTRAL_PARAMETERS:
        indices = [index for index, case in enumerate(cases)
                   if case["parameter"] == parameter]
        accepted = [index for index in indices if converged[index]]
        recommendations[parameter] = {
            "recommended": cases[accepted[0]]["value"] if accepted else None,
            "criterion": "upper confidence bound on relative error <= relative_tolerance",
            "relative_tolerance": float(cfg["relative_tolerance"]),
            "advisory_only": True,
        }
    result = {
        "task": "sampling-check",
        "cases": cases,
        "relative_error_seed": errors,
        "relative_error_mean": mean,
        "relative_error_se": se,
        "relative_error_ci_low": low,
        "relative_error_ci_high": high,
        "converged": converged,
        "recommendations": recommendations,
    }
    _write_outputs(run_dir, result, cfg)
    return result
