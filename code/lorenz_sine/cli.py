"""Independent command-line stages for the Lorenz sine research workflow."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))

from . import storage  # noqa: E402
from .config import config_hash, forced_sampling_metadata, load_config  # noqa: E402


def mpi_setup():
    local_workers = os.getenv("LORENZ_LOCAL_WORKERS")
    if local_workers:
        from dask.distributed import Client, LocalCluster

        storage.ensure_roots()
        worker_count = int(local_workers)
        if worker_count < 1:
            raise ValueError("LORENZ_LOCAL_WORKERS must be positive")
        cluster = LocalCluster(
            n_workers=worker_count,
            threads_per_worker=1,
            dashboard_address=None,
            local_directory=str(storage.DASK_CACHE_ROOT),
        )
        client = Client(cluster)
        print(f"execution_backend=local-dask workers={worker_count}", flush=True)
        return client, "local-dask", worker_count

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
    except Exception:
        comm = None
        size = int(os.getenv(
            "OMPI_COMM_WORLD_SIZE",
            os.getenv("PMI_SIZE", os.getenv("SLURM_NTASKS", "1")),
        ))
    if size > 2:
        from dask_mpi import initialize

        logging.getLogger("distributed").setLevel(logging.WARNING)
        storage.ensure_roots()
        initialize(
            dashboard=False,
            local_directory=str(storage.DASK_CACHE_ROOT),
            worker_options={"silence_logs": logging.WARNING},
            comm=comm,
        )
        from dask.distributed import Client

        client = Client()
        client.wait_for_workers(size - 2)
        workers = len(client.scheduler_info()["workers"])
        print(f"execution_backend=dask-mpi workers={workers}", flush=True)
        return client, "dask-mpi", workers
    print("execution_backend=serial workers=1", flush=True)
    return None, "serial", 1


def _input_path(run_id):
    return str(Path("results") / "runs" / run_id / "data" / "result.npz")


def _read_parent_config(run_dir):
    return storage.read_json(Path(run_dir) / "config.json")


def _require_compatible(current, parent, keys, parent_name):
    for key in keys:
        if current.get(key) != parent.get(key):
            raise ValueError(
                f"{parent_name} run is incompatible for {key}: "
                f"{parent.get(key)!r} != {current.get(key)!r}"
            )


def _resolve_spectrum_frequencies(cfg, spectrum_result):
    ranks = [int(value) for value in cfg.get("spectrum_peak_ranks", [1])]
    peaks = np.asarray(spectrum_result["peaks"])
    selected = []
    for rank in ranks:
        rows = peaks[peaks[:, 0].astype(int) == rank]
        if len(rows) != 1:
            raise ValueError(f"spectrum run has no unique peak rank {rank}")
        selected.append(float(rows[0, 3]))
    return selected


def _prepare_inputs(task, args, cfg):
    parents = {}
    paths = {}
    inputs = {}
    if task == "steady":
        spectrum = None
        if args.spectrum_run:
            run_dir, manifest, result = storage.require_run(args.spectrum_run, "spectrum")
            parents["spectrum"] = args.spectrum_run
            paths["spectrum"] = _input_path(args.spectrum_run)
            inputs["spectrum"] = result
            spectrum = result
            _require_compatible(cfg, _read_parent_config(run_dir),
                                ("lorenz",), "spectrum")
        if not cfg["frequencies"]:
            if spectrum is None:
                raise ValueError(
                    "steady config has no frequencies; provide --spectrum-run <run_id>"
                )
            cfg["frequencies"] = _resolve_spectrum_frequencies(cfg, spectrum)
            cfg["frequency_source"] = {
                "kind": "spectrum-run",
                "run_id": args.spectrum_run,
                "peak_ranks": cfg.get("spectrum_peak_ranks", [1]),
            }
        else:
            cfg["frequencies"] = [float(value) for value in cfg["frequencies"]]
            cfg["frequency_source"] = {"kind": "configuration"}
        cfg["sampling_metadata"] = [
            forced_sampling_metadata(omega, cfg) for omega in cfg["frequencies"]
        ]
    elif task == "amplitude-scan":
        if args.steady_run:
            run_dir, manifest, result = storage.require_run(args.steady_run, "steady")
            parents["steady"] = args.steady_run
            paths["steady"] = _input_path(args.steady_run)
            inputs["steady"] = result
            parent_cfg = _read_parent_config(run_dir)
            _require_compatible(cfg, parent_cfg, ("lorenz", "solver", "n_phase"), "steady")
            cfg["frequencies"] = [float(value) for value in result["frequencies"]]
            n_skip = {}
            n_skip_source = {}
            for omega in cfg["frequencies"]:
                recommendation = result["recommendations"][str(float(omega))]
                if not recommendation["converged"]:
                    if not cfg.get("allow_unconverged_steady", False):
                        raise ValueError(
                            f"steady run {args.steady_run} has no converged n_skip "
                            f"for omega={omega}"
                        )
                    fallback = cfg.get("fallback_n_skip")
                    if fallback is None:
                        fallback = max(int(value) for value in result["candidate_n_skip"])
                    n_skip[str(float(omega))] = int(fallback)
                    n_skip_source[str(float(omega))] = {
                        "kind": "configured_fallback_after_unconverged_steady",
                        "steady_run": args.steady_run,
                        "stable_flags": recommendation.get("stable_flags", []),
                        "max_relative_delta": recommendation.get("max_relative_delta", []),
                    }
                else:
                    n_skip[str(float(omega))] = int(recommendation["recommended_n_skip"])
                    n_skip_source[str(float(omega))] = {
                        "kind": "steady_recommendation",
                        "steady_run": args.steady_run,
                    }
            cfg["n_skip_by_frequency"] = n_skip
            cfg["n_skip_source"] = n_skip_source
        else:
            if not cfg.get("frequencies") or not cfg.get("n_skip_by_frequency"):
                raise ValueError(
                    "amplitude-scan requires --steady-run unless config provides "
                    "explicit frequencies and n_skip_by_frequency"
                )
            cfg["frequencies"] = [float(value) for value in cfg["frequencies"]]
            n_skip = {}
            for omega in cfg["frequencies"]:
                key = str(float(omega))
                if key not in cfg["n_skip_by_frequency"]:
                    raise ValueError(f"missing n_skip_by_frequency entry for omega={omega}")
                n_skip[key] = int(cfg["n_skip_by_frequency"][key])
            cfg["n_skip_by_frequency"] = n_skip
            cfg["n_skip_source"] = {
                str(float(omega)): {
                    "kind": "explicit_configuration",
                    "reason": cfg.get("n_skip_rationale", ""),
                }
                for omega in cfg["frequencies"]
            }
        cfg["sampling_metadata"] = [
            forced_sampling_metadata(omega, cfg) for omega in cfg["frequencies"]
        ]
    elif task == "response":
        run_dir, manifest, result = storage.require_run(args.amplitude_run, "amplitude-scan")
        parents["amplitude-scan"] = args.amplitude_run
        paths["amplitude-scan"] = _input_path(args.amplitude_run)
        inputs["amplitude-scan"] = result
    elif task == "higher-order":
        run_dir, manifest, result = storage.require_run(args.response_run, "response")
        parents["response"] = args.response_run
        paths["response"] = _input_path(args.response_run)
        inputs["response"] = result
        amplitude_id, _, _, amplitude = storage.find_ancestor_run(
            args.response_run, "amplitude-scan")
        parents["amplitude-scan"] = amplitude_id
        paths["amplitude-scan"] = _input_path(amplitude_id)
        inputs["amplitude-scan"] = amplitude
    elif task == "validate":
        _, higher_manifest, higher = storage.require_run(
            args.higher_order_run, "higher-order")
        _, response_manifest, response = storage.require_run(args.response_run, "response")
        recorded_response = higher_manifest.get("parent_run_ids", {}).get("response")
        if recorded_response != args.response_run:
            raise ValueError(
                f"higher-order run {args.higher_order_run} was built from response run "
                f"{recorded_response!r}, not {args.response_run!r}"
            )
        parents.update({
            "response": args.response_run,
            "higher-order": args.higher_order_run,
        })
        paths.update({
            "response": _input_path(args.response_run),
            "higher-order": _input_path(args.higher_order_run),
        })
        inputs.update({"response": response, "higher-order": higher})
    elif task == "frequency-scan":
        _, _, response = storage.require_run(args.response_run, "response")
        parents["response"] = args.response_run
        paths["response"] = _input_path(args.response_run)
        inputs["response"] = response
    elif task == "report":
        _, _, validation = storage.require_run(args.input_run, "validate")
        parents["validate"] = args.input_run
        paths["validate"] = _input_path(args.input_run)
        sources = {}
        for source_task in ("validate", "higher-order", "response",
                            "amplitude-scan", "steady", "spectrum"):
            try:
                source_id, _, manifest, result = storage.find_ancestor_run(
                    args.input_run, source_task)
                sources[source_task] = (source_id, result, manifest)
            except ValueError:
                if source_task != "spectrum":
                    raise
        if args.sampling_run:
            _, manifest, result = storage.require_run(
                args.sampling_run, "sampling-check")
            parents["sampling-check"] = args.sampling_run
            paths["sampling-check"] = _input_path(args.sampling_run)
            sources["sampling-check"] = (args.sampling_run, result, manifest)
        if args.frequency_run:
            _, manifest, result = storage.require_run(
                args.frequency_run, "frequency-scan")
            parents["frequency-scan"] = args.frequency_run
            paths["frequency-scan"] = _input_path(args.frequency_run)
            sources["frequency-scan"] = (args.frequency_run, result, manifest)
        inputs["sources"] = sources
    cfg["config_hash"] = config_hash(cfg)
    return parents, paths, inputs


def _run_module(task, cfg, inputs, run_dir, runtime, resume, client):
    if task == "spectrum":
        from . import natural_spectrum
        return natural_spectrum.run(cfg, run_dir, runtime, resume=resume, client=client)
    if task == "sampling-check":
        from . import sampling_check
        return sampling_check.run(cfg, run_dir, runtime, resume=resume, client=client)
    if task == "steady":
        from . import steady_check
        return steady_check.run(cfg, run_dir, runtime, resume=resume, client=client)
    if task == "amplitude-scan":
        from . import amplitude_scan
        return amplitude_scan.run(cfg, run_dir, runtime, resume=resume, client=client)
    if task == "response":
        from . import response
        return response.run(
            cfg, inputs["amplitude-scan"], run_dir, runtime, resume=resume)
    if task == "higher-order":
        from . import higher_order
        return higher_order.run(
            cfg, inputs["response"], inputs["amplitude-scan"], run_dir,
            runtime, resume=resume)
    if task == "validate":
        from . import validation
        return validation.run(
            cfg, inputs["response"], inputs["higher-order"], run_dir)
    if task == "frequency-scan":
        from . import frequency_scan
        return frequency_scan.run(cfg, inputs["response"], run_dir)
    if task == "report":
        from . import report
        return report.run(cfg, inputs["sources"], run_dir)
    raise ValueError(f"unsupported task {task!r}")


def run_stage(args):
    task = args.command
    client, backend, workers = mpi_setup()
    run_dir = None
    run_id = None
    try:
        cfg = load_config(args.config, task=task)
        parents, paths, inputs = _prepare_inputs(task, args, cfg)
        run_id, run_dir, runtime, resumed = storage.begin_run(
            task,
            cfg,
            sys.argv,
            parents,
            paths,
            backend,
            workers,
            resume=args.resume,
        )
        print(f"RUN_ID={run_id}", flush=True)
        status = storage.read_json(run_dir / "status.json")
        if resumed and status.get("status") == "completed" and storage.valid_npz(
                run_dir / "data" / "result.npz", ("result",)):
            print("resume_skipped=completed_run", flush=True)
            return storage.package_run(run_id)
        started = time.time()
        try:
            with storage.stage_timer(run_dir, runtime, task):
                result = _run_module(
                    task, cfg, inputs, run_dir, runtime, resumed, client)
            storage.save_result(run_dir, result)
            storage.complete_run(run_dir, runtime, started)
        except Exception as exc:
            storage.fail_run(run_dir, exc, stage=task)
            storage.package_run(run_id)
            raise
        return storage.package_run(run_id)
    finally:
        if client is not None:
            client.close()


def package(args):
    return storage.package_run(args.run_id)


def _add_common(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.set_defaults(func=run_stage)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lorenz_sine.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_common(subparsers.add_parser("spectrum"))
    _add_common(subparsers.add_parser("sampling-check"))

    steady = subparsers.add_parser("steady")
    _add_common(steady)
    steady.add_argument("--spectrum-run")

    amplitude = subparsers.add_parser("amplitude-scan")
    _add_common(amplitude)
    amplitude.add_argument("--steady-run")

    response = subparsers.add_parser("response")
    _add_common(response)
    response.add_argument("--amplitude-run", required=True)

    higher = subparsers.add_parser("higher-order")
    _add_common(higher)
    higher.add_argument("--response-run", required=True)

    frequency = subparsers.add_parser("frequency-scan")
    _add_common(frequency)
    frequency.add_argument("--response-run", required=True)

    validate = subparsers.add_parser("validate")
    _add_common(validate)
    validate.add_argument("--response-run", required=True)
    validate.add_argument("--higher-order-run", required=True)

    report = subparsers.add_parser("report")
    _add_common(report)
    report.add_argument("--input-run", required=True)
    report.add_argument("--sampling-run")
    report.add_argument("--frequency-run")

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--run-id", required=True)
    package_parser.set_defaults(func=package)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
