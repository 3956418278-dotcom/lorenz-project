"""One-level task execution with partial-result callbacks and deterministic order."""

from __future__ import annotations


def execute_tasks(items, worker, client=None, worker_kwargs=None,
                  on_result=None, on_error=None, key_string=str):
    """Execute top-level worker(item, **kwargs), retaining successful results.

    The callback runs on the client after each successful task, which lets a
    stage persist per-seed checkpoints even if a different seed later fails.
    """
    items = list(items)
    kwargs = dict(worker_kwargs or {})
    results = {}
    errors = []
    if client is None:
        for item in items:
            try:
                result = worker(item, **kwargs)
                results[key_string(item)] = result
                if on_result is not None:
                    on_result(item, result)
            except Exception as exc:
                errors.append((item, exc))
                if on_error is not None:
                    on_error(item, exc)
    else:
        from distributed import as_completed

        futures = {
            client.submit(worker, item, pure=False, **kwargs): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results[key_string(item)] = result
                if on_result is not None:
                    on_result(item, result)
            except Exception as exc:
                errors.append((item, exc))
                if on_error is not None:
                    on_error(item, exc)
    if errors:
        details = "; ".join(
            f"{key_string(item)}: {type(exc).__name__}: {exc}"
            for item, exc in errors
        )
        raise RuntimeError(f"{len(errors)} parameter task(s) failed: {details}")
    return [results[key_string(item)] for item in items]
