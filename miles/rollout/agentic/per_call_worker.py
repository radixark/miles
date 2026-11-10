"""Sandbox start functions, and running a call on an agent worker one of them starts for it.

A start function (``StartFn``) starts one sandbox and returns ``(close_fn, base_url)``. Named by
``--custom-agent-function-worker``, it is called for each call as ``start(label, "")``, and ``base_url`` is an agent
worker (``miles.rollout.agentic.worker``) serving there. The call runs on it as on a fixed worker, and ``close_fn``,
which copies out what the call left in the sandbox and frees it, runs however the call ends.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from miles.rollout.agentic.errors import InfraAbort
from miles.rollout.agentic.remote import call_on_worker
from miles.rollout.agentic.rollout_limits import UNSHARED_ENV_VAR

logger = logging.getLogger(__name__)

# A provider's "start one sandbox" hook: (task_id, tasks_dir) -> (close_fn, base_url).
StartFn = Callable[[str, str], tuple[Callable[[], None], str]]

_HEALTH_DEADLINE_S = 120.0
_HEALTH_POLL_S = 0.2


async def create_once(start_fn: StartFn, task_id: str, tasks_dir: str, *, logger: logging.Logger) -> tuple[Any, str]:
    """One sandbox-create attempt, safe against cancellation mid-create."""
    result: list[tuple[Any, str]] = []
    done = threading.Event()
    claim = threading.Lock()
    began, abandoned = threading.Event(), threading.Event()

    def _start() -> tuple[Any, str] | None:
        with claim:
            if abandoned.is_set():
                return None
            began.set()
        try:
            result.append(start_fn(task_id, tasks_dir))
        finally:
            done.set()
        return result[0]

    try:
        return await asyncio.to_thread(_start)
    except asyncio.CancelledError:
        with claim:
            abandoned.set()
        if not began.is_set():
            raise

        def _reap() -> None:
            done.wait()
            for close_fn, _url in result:
                try:
                    close_fn()
                    logger.info(f"Closed sandbox orphaned by cancelled episode: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to close orphaned sandbox for {task_id}: {e}")

        # not a daemon: a subproc agent call waits for it before its process exits
        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=False).start()
        raise


async def call_on_started_worker(
    start_fn: StartFn,
    fn: Callable,
    kwargs: dict[str, Any],
    *,
    env: dict[str, str],
    on_child_reports: Callable[[list[Any]], None] | None = None,
) -> Any:
    """Run ``fn(**kwargs)`` on the agent worker ``start_fn`` starts for this call, and close its sandbox after."""
    label = str((kwargs.get("metadata") or {}).get("instance_id") or fn.__name__)
    close_fn = None
    try:
        close_fn, worker_url = await _start(start_fn, label)
        await _wait_healthy(worker_url, label)
        return await call_on_worker(
            worker_url, fn, kwargs, env={**env, UNSHARED_ENV_VAR: "per-call worker"}, on_child_reports=on_child_reports
        )
    finally:
        if close_fn is not None:
            await _close(close_fn, label)


async def _start(start_fn: StartFn, label: str) -> tuple[Callable[[], None], str]:
    try:
        return await create_once(start_fn, label, "", logger=logger)
    except Exception as exc:
        raise InfraAbort("SandboxWorkerUnavailable", f"starting the sandbox worker for {label}: {exc!r}") from exc


async def _wait_healthy(worker_url: str, label: str) -> None:
    deadline = time.monotonic() + _HEALTH_DEADLINE_S
    async with httpx.AsyncClient(base_url=worker_url, timeout=10) as client:
        while True:
            if await _is_healthy(client):
                return
            if time.monotonic() > deadline:
                raise InfraAbort(
                    "SandboxWorkerUnavailable",
                    f"the sandbox worker for {label} at {worker_url} not healthy after {_HEALTH_DEADLINE_S:.0f}s",
                )
            await asyncio.sleep(_HEALTH_POLL_S)


async def _is_healthy(client: httpx.AsyncClient) -> bool:
    try:
        return (await client.get("/health")).status_code == 200
    except httpx.TransportError:
        return False


async def _close(close_fn: Callable[[], None], label: str) -> None:
    def close() -> None:
        try:
            close_fn()
        except Exception as exc:
            logger.warning(f"closing the sandbox worker for {label} failed: {exc!r}")

    await asyncio.shield(asyncio.to_thread(close))
