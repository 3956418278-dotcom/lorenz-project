"""Execution adapter for local processes and Dask/MPI workers."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DASK_MAX_IN_FLIGHT = 2048


@dataclass
class ExecutionRuntime:
    client: Any = None
    cluster: Any = None
    backend: str = "process-pool"
    workers: int = 1
    executor: Any = None


_ACTIVE_RUNTIME: ExecutionRuntime | None = None


def _cache_directory() -> Path:
    configured = os.getenv("DASK_TEMPORARY_DIRECTORY")
    path = (
        Path(configured).expanduser()
        if configured
        else REPO_ROOT / "runtime" / "cache" / "dask-worker-space"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def initialize_execution(*, workers: int = 1) -> ExecutionRuntime:
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
        # Initialization can be slow when starting many local processes;
        # this connection grace period is separate from task throughput.
        try:
            client = Client(cluster, timeout="120s")
            registered = len(client.scheduler_info()["workers"])
            print(f"local-dask registered_workers={registered}/{worker_count}", flush=True)
            if registered != worker_count:
                client.wait_for_workers(worker_count, timeout="120s")
        except BaseException:
            cluster.close()
            raise
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

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    runtime = ExecutionRuntime(backend="process-pool", workers=workers)
    _ACTIVE_RUNTIME = runtime
    print("execution_backend=process-pool", flush=True)
    return runtime


def execution_worker_count(workers: int) -> int:
    """The active distributed runtime owns its worker count."""
    runtime = _ACTIVE_RUNTIME
    return runtime.workers if runtime is not None and runtime.client is not None else workers


def iter_task_results(
    function: Callable[[Any], Any], tasks: Iterable[Any], *, workers: int,
    max_in_flight: int | None = None,
) -> Iterator[tuple[int, Any]]:
    """Yield indexed completions with bounded, rolling task submission.

    Persistence chunks do not impose scheduling barriers. An active local
    process runtime reuses its executor across spinup and integration calls.
    Task indices preserve scientific axes even when completion is unordered.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    runtime = _ACTIVE_RUNTIME
    capacity = execution_worker_count(workers)
    limit = min(DASK_MAX_IN_FLIGHT, max(1, 2 * capacity)) if max_in_flight is None else max_in_flight
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_in_flight must be a positive integer")
    task_iterator = enumerate(tasks)
    if runtime is not None and runtime.client is not None:
        from dask.distributed import as_completed
        initial_tasks = list(islice(task_iterator, limit))
        if not initial_tasks:
            return
        futures = runtime.client.map(function, [task for _, task in initial_tasks], pure=False)
        future_indices = dict(zip(futures, (index for index, _ in initial_tasks)))
        completion = as_completed(futures, with_results=True)
        try:
            while future_indices:
                completed = completion.next_batch(block=True)
                results = []
                for future, result in completed:
                    results.append((future_indices.pop(future), result))
                    future.release()
                replacement_tasks = list(islice(task_iterator, len(completed)))
                if replacement_tasks:
                    replacements = runtime.client.map(
                        function, [task for _, task in replacement_tasks], pure=False
                    )
                    future_indices.update(zip(replacements, (index for index, _ in replacement_tasks)))
                    completion.update(replacements)
                yield from results
        finally:
            if future_indices:
                runtime.client.cancel(list(future_indices))
        return
    if workers == 1:
        for index, task in task_iterator:
            yield index, function(task)
        return
    owns_executor = runtime is None
    if owns_executor:
        executor = ProcessPoolExecutor(max_workers=workers)
    else:
        if runtime.executor is None:
            runtime.executor = ProcessPoolExecutor(max_workers=workers)
            runtime.workers = workers
        elif runtime.workers != workers:
            raise ValueError("an active process runtime cannot change workers")
        executor = runtime.executor
    pending = {executor.submit(function, task): index for index, task in islice(task_iterator, limit)}
    try:
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            results = [(pending.pop(future), future.result()) for future in completed]
            for index, task in islice(task_iterator, len(completed)):
                pending[executor.submit(function, task)] = index
            yield from results
    finally:
        for future in pending:
            future.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)


def map_tasks(
    function: Callable[[Any], Any], tasks: Iterable[Any], *, workers: int
) -> list[Any]:
    """Return results in input order through the shared rolling executor."""
    results = dict(iter_task_results(function, tasks, workers=workers))
    return [results[index] for index in range(len(results))]


def close_execution(runtime: ExecutionRuntime) -> None:
    global _ACTIVE_RUNTIME
    try:
        if runtime.client is not None:
            runtime.client.close()
        if runtime.cluster is not None:
            runtime.cluster.close()
        if runtime.executor is not None:
            runtime.executor.shutdown(wait=True, cancel_futures=True)
    finally:
        _ACTIVE_RUNTIME = None
