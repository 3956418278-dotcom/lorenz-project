"""Release execution adapter for local processes and Dask/MPI workers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExecutionRuntime:
    client: Any = None
    cluster: Any = None
    backend: str = "process-pool"
    workers: int = 1


_ACTIVE_RUNTIME: ExecutionRuntime | None = None


def _cache_directory() -> Path:
    configured = os.getenv("DASK_TEMPORARY_DIRECTORY")
    path = (
        Path(configured).expanduser()
        if configured
        else RELEASE_ROOT / "runtime" / "cache" / "dask-worker-space"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def initialize_execution() -> ExecutionRuntime:
    """Initialize local Dask or a scheduler/client/worker dask-mpi layout."""
    global _ACTIVE_RUNTIME
    if _ACTIVE_RUNTIME is not None:
        raise RuntimeError("execution runtime is already initialized")

    cache = _cache_directory()
    local_workers = os.getenv("LORENZ_LOCAL_WORKERS")
    if local_workers is not None:
        worker_count = int(local_workers)
        if worker_count < 1:
            raise ValueError("LORENZ_LOCAL_WORKERS must be positive")
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=worker_count,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
            local_directory=str(cache),
        )
        client = Client(cluster)
        runtime = ExecutionRuntime(
            client=client,
            cluster=cluster,
            backend="local-dask",
            workers=worker_count,
        )
        _ACTIVE_RUNTIME = runtime
        print(f"execution_backend=local-dask workers={worker_count}", flush=True)
        return runtime

    launcher_size = int(os.getenv(
        "OMPI_COMM_WORLD_SIZE",
        os.getenv("PMI_SIZE", os.getenv("SLURM_NTASKS", "1")),
    ))
    if launcher_size > 1:
        from mpi4py import MPI

        communicator = MPI.COMM_WORLD
        mpi_size = communicator.Get_size()
    else:
        communicator = None
        mpi_size = 1

    if mpi_size == 2:
        raise ValueError(
            "dask-mpi requires at least three ranks: scheduler, client, worker"
        )
    if mpi_size > 2:
        from dask_mpi import initialize

        logging.getLogger("distributed").setLevel(logging.WARNING)
        initialize(
            dashboard=False,
            local_directory=str(cache),
            worker_options={"silence_logs": logging.WARNING},
            comm=communicator,
        )
        from dask.distributed import Client

        client = Client()
        client.wait_for_workers(mpi_size - 2)
        worker_count = len(client.scheduler_info()["workers"])
        runtime = ExecutionRuntime(
            client=client,
            backend="dask-mpi",
            workers=worker_count,
        )
        _ACTIVE_RUNTIME = runtime
        print(f"execution_backend=dask-mpi workers={worker_count}", flush=True)
        return runtime

    runtime = ExecutionRuntime(backend="process-pool")
    _ACTIVE_RUNTIME = runtime
    print("execution_backend=process-pool", flush=True)
    return runtime


def map_tasks(
    function: Callable[[Any], Any], tasks: Iterable[Any], *, workers: int
) -> list[Any]:
    """Map current numerical task functions without changing their inputs."""
    task_list = list(tasks)
    runtime = _ACTIVE_RUNTIME
    if runtime is not None and runtime.client is not None:
        futures = runtime.client.map(function, task_list, pure=False)
        try:
            return runtime.client.gather(futures)
        finally:
            runtime.client.cancel(futures)
    if workers == 1:
        return list(map(function, task_list))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, task_list))


def close_execution(runtime: ExecutionRuntime) -> None:
    global _ACTIVE_RUNTIME
    try:
        if runtime.client is not None:
            runtime.client.close()
        if runtime.cluster is not None:
            runtime.cluster.close()
    finally:
        _ACTIVE_RUNTIME = None
