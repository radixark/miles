"""Shared per-episode sandbox orchestration for the provider agent functions.

A provider leg (``openenv_daytona_agent_function``, ``openenv_e2b_agent_function``)
owns only what is genuinely provider-specific: how ONE sandbox with the env
server comes into being (its ``tb2_sandbox_*`` materialization module), which
errors count as retryable throttling, and its env-var knobs. Everything that
makes per-episode sandboxes safe under a fanned-out rollout is provider-blind
and lives here once:

  ``create_once``          cancel-safe create. ``asyncio.to_thread`` is not
                cancellable: when an episode's wall-clock cap cancels the
                coroutine mid-create, the worker thread keeps running and its
                (close_fn, url) result would be discarded — leaking a sandbox
                until the provider-side TTL backstop reclaims it. The result
                is recorded thread-side and, on cancellation, handed to a
                reaper that closes the orphan promptly once the create
                finishes.
  ``lazy_semaphore``       the create-throttle semaphore each leg passes in,
                built on first use rather than at import.
  ``start_task_sandbox``   process-wide create throttling (the caller's
                semaphore) plus jittered exponential backoff on the errors the
                caller classifies as throttling; anything else propagates
                immediately. The semaphore is held only for the create attempt
                and released during backoff so other episodes keep the
                pipeline full.
  ``episode_env``          async context manager mirroring one episode's
                lifetime: fresh sandbox -> connected env client -> close.
"""

import asyncio
import logging
import random
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

# A provider's "start one sandbox" hook: (task_id, tasks_dir) -> (close_fn, base_url).
StartFn = Callable[[str, str], tuple[Callable[[], None], str]]

# The canonical per-episode backend registry — the launcher and the sibling
# tools (scan_golden, eval_tbench2_via_api) all resolve through here, so the
# accepted names and the "agentenv is the e2b leg" aliasing cannot drift.
AGENT_FUNCTIONS = {
    "daytona": "openenv_daytona_agent_function.run",
    "e2b": "openenv_e2b_agent_function.run",
}
_ALIASES = {"agentenv": "e2b"}


def resolve_backend(name: str | None) -> str:
    """Normalize a backend name; reject unknown and missing ones alike.

    There is deliberately no default. Which provider runs decides whose quota
    an entire rollout spends and which credentials must be present, so a name
    left out is a question to answer rather than one to guess at.
    """
    backend = (name or "").strip().lower()
    allowed = ", ".join(sorted([*AGENT_FUNCTIONS, *_ALIASES]))
    if not backend:
        raise ValueError(f"no sandbox backend named; choose one of: {allowed}")
    backend = _ALIASES.get(backend, backend)
    if backend not in AGENT_FUNCTIONS:
        raise ValueError(f"unknown sandbox backend {name!r}; choose one of: {allowed}")
    return backend


def lazy_semaphore(limit: int) -> Callable[[], asyncio.Semaphore]:
    """Return a getter for a *limit*-slot semaphore created on first call.

    A leg reads its concurrency knob at import, but a semaphore constructed
    then would belong to whatever loop happens to be current — the rollout
    loop does not exist yet. Deferring construction to the first episode ties
    it to the loop that actually awaits on it.
    """
    holder: list[asyncio.Semaphore] = []

    def get() -> asyncio.Semaphore:
        if not holder:
            holder.append(asyncio.Semaphore(limit))
        return holder[0]

    return get


async def create_once(start_fn: StartFn, task_id: str, tasks_dir: str, *, logger: logging.Logger) -> tuple[Any, str]:
    """One sandbox-create attempt, safe against cancellation mid-create."""
    result: list[tuple[Any, str]] = []
    done = threading.Event()

    def _start() -> tuple[Any, str]:
        try:
            result.append(start_fn(task_id, tasks_dir))
        finally:
            done.set()
        return result[0]

    try:
        return await asyncio.to_thread(_start)
    except asyncio.CancelledError:

        def _reap() -> None:
            done.wait()
            for close_fn, _url in result:
                try:
                    close_fn()
                    logger.info(f"Closed sandbox orphaned by cancelled episode: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to close orphaned sandbox for {task_id}: {e}")

        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=True).start()
        raise


async def start_task_sandbox(
    task_id: str,
    tasks_dir: str,
    *,
    start_fn: StartFn,
    is_throttle: Callable[[BaseException], bool],
    sem: asyncio.Semaphore,
    max_retries: int,
    backoff_base_s: float,
    backoff_cap_s: float,
    logger: logging.Logger,
    provider: str,
) -> tuple[Any, str]:
    """Create one sandbox for *task_id* with the env server running.

    Returns (close_fn, base_url). Creation is throttled process-wide and
    retried with jittered exponential backoff on throttle errors.
    """
    attempt = 0
    while True:
        try:
            async with sem:
                return await create_once(start_fn, task_id, tasks_dir, logger=logger)
        except Exception as e:
            if not is_throttle(e) or attempt >= max_retries:
                raise
            attempt += 1
            delay = min(backoff_cap_s, backoff_base_s * (2 ** (attempt - 1))) * (0.5 + random.random())
            logger.warning(
                f"{provider} create throttled for {task_id} (attempt {attempt}/{max_retries}); retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


@asynccontextmanager
async def episode_env(
    env_cls: Any,
    metadata: dict[str, Any],
    *,
    start: Callable[[str], Awaitable[tuple[Any, str]]],
    logger: logging.Logger,
):
    """Yield a connected env client on a fresh sandbox; close it after."""
    # Imported lazily so provider CLIs (bake) don't drag in the agent loop's
    # dependencies (openai) just to reach this module's registry.
    import openenv_agent_function as oaf

    task_id = metadata.get("task_id") or metadata.get("task_name")
    if not task_id:
        raise ValueError("the sandbox is built for one task: metadata['task_id'] is required")
    close_fn, url = await start(str(task_id))
    try:
        async with env_cls(base_url=url, message_timeout_s=oaf._MESSAGE_TIMEOUT_S) as env:
            yield env
    finally:
        try:
            await asyncio.to_thread(close_fn)
        except Exception as e:
            logger.warning(f"Failed to close sandbox for {task_id}: {e}")
