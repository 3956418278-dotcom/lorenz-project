"""Unified command-line entry point for Lorenz sine experiments."""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import boundary, frequency_scan, natural_spectrum, report, steady_check, storage
from .config import load_config


def mpi_setup():
    pmi_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", os.getenv("PMI_SIZE", os.getenv("SLURM_NTASKS", 1))))
    if pmi_size > 2:
        from dask_mpi import initialize
        initialize()
        from dask.distributed import Client
        return Client()
    return None


def infer_frequencies(cfg, spectrum):
    if cfg.get("frequencies"):
        return cfg["frequencies"]
    peaks = spectrum.get("peaks")
    if peaks is not None and len(peaks):
        cfg["frequencies"] = [float(peaks[0, 2])]
    else:
        cfg["frequencies"] = [1.0]
    return cfg["frequencies"]


def run(args):
    mpi_setup()
    cfg = load_config(args.config, smoke_test=args.smoke_test)
    run_id, run_dir = storage.create_run(args.mode, cfg, resume=args.resume)
    t_start = time.time()
    if args.resume and (run_dir / "status.json").exists():
        runtime = storage.create_runtime()
    else:
        runtime = storage.init_run_files(run_dir, run_id, args.mode, cfg, sys.argv)
    storage.update_status(run_dir, status="running", mode=args.mode)
    spectrum = steady = responses = bound = None
    try:
        if args.mode in ("spectrum", "all"):
            with storage.stage_timer(run_dir, runtime, "spectrum"):
                spectrum = natural_spectrum.run(cfg)

        omegas = infer_frequencies(cfg, spectrum or {})

        if args.mode in ("steady", "all"):
            with storage.stage_timer(run_dir, runtime, "steady"):
                steady = steady_check.run(omegas, cfg, runtime)
        else:
            steady = {}

        if args.mode in ("response", "scan", "all"):
            with storage.stage_timer(run_dir, runtime, "response"):
                responses = frequency_scan.run(omegas, steady or {}, cfg, run_dir,
                                               runtime, resume=bool(args.resume))
        else:
            responses = []

        if args.mode in ("boundary", "all"):
            with storage.stage_timer(run_dir, runtime, "boundary"):
                bound = boundary.run(responses or [], cfg)

        with storage.stage_timer(run_dir, runtime, "report"):
            report.make_report(run_dir, cfg, spectrum=spectrum, steady=steady,
                               responses=responses, boundary=bound)
        storage.write_result(run_dir, spectrum, steady, responses, bound, runtime)
        storage.write_summary(run_dir, spectrum=spectrum, steady=steady,
                              responses=responses, boundary=bound)
        storage.complete_run(run_dir, t_start)
    except Exception as exc:
        storage.fail_run(run_dir, exc)
        storage.package_run(run_id)
        raise
    pkg = storage.package_run(run_id)
    return pkg


def package(args):
    return storage.package_run(args.run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lorenz_sine.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--mode", choices=["spectrum", "steady", "response", "scan", "boundary", "all"], default="all")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--resume")
    run_p.add_argument("--smoke-test", action="store_true")
    run_p.set_defaults(func=run)

    pkg_p = sub.add_parser("package")
    pkg_p.add_argument("--run-id", required=True)
    pkg_p.set_defaults(func=package)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
