"""Call a custom agent function on the rollout loop or in a process of its own.

Every sample's agent function runs concurrently in the rollout process. On one
event loop, a synchronous step in one episode (a large response to parse, a slow
file write) stalls every other episode, long enough under many episodes for their
sandbox SDK requests to time out. ``subproc`` gives each call a fresh process on
this node, so a slow step, a crash or leaked state stays inside its own episode.
"""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

AGENT_FUNCTION_MODES = ("subproc", "inline")

# A cancelled call gets this long to run its own cleanup (closing its sandbox) before its process is killed.
_CANCEL_GRACE_S = 120.0
# How long the process waits, before exiting, for cleanup an agent left in a non-daemon thread.
_THREAD_JOIN_TIMEOUT_S = 600.0


async def call_agent_function(fn: Callable[..., Awaitable[Any]], *, mode: str, **kwargs: Any) -> Any:
    """Await ``fn(**kwargs)`` on this loop (``inline``) or in a fresh process on this node (``subproc``)."""
    if mode == "inline":
        return await fn(**kwargs)
    if not ray.is_initialized():
        raise RuntimeError("--custom-agent-function-mode subproc needs a Ray runtime; use inline outside Ray")
    ref = _call_in_fresh_process.options(scheduling_strategy=_on_this_node()).remote(fn, kwargs)
    try:
        return await ref
    except asyncio.CancelledError:
        ray.cancel(ref)
        threading.Thread(target=_kill_after_grace, args=(ref,), daemon=True).start()
        raise


def _on_this_node() -> NodeAffinitySchedulingStrategy:
    # trial directories and key files are where the rollout process runs
    return NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)


def _kill_after_grace(ref: ray.ObjectRef) -> None:
    ready, _ = ray.wait([ref], timeout=_CANCEL_GRACE_S)
    if not ready:
        ray.cancel(ref, force=True)


# num_cpus is a scheduling token: an episode mostly waits on the model and its sandbox.
# max_retries=0: a rerun would replay the episode against the same session.
@ray.remote(num_cpus=0.01, max_calls=1, max_retries=0)
def _call_in_fresh_process(fn: Callable[..., Awaitable[Any]], kwargs: dict[str, Any]) -> Any:
    configure_logger_raw("agent_function")
    threads_before = set(threading.enumerate())
    try:
        return asyncio.run(_await_waking_loop(fn(**kwargs)))
    finally:
        _join_threads_started_since(threads_before)


async def _await_waking_loop(call: Awaitable[Any]) -> Any:
    # ray.cancel lands only while Python code runs; a loop idle in select would notice it late
    waker = asyncio.create_task(_wake_every_second())
    try:
        return await call
    finally:
        waker.cancel()


async def _wake_every_second() -> None:
    while True:
        await asyncio.sleep(1.0)


def _join_threads_started_since(threads_before: set[threading.Thread]) -> None:
    for thread in set(threading.enumerate()) - threads_before:
        if thread.daemon:
            continue
        thread.join(_THREAD_JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.warning(f"Thread {thread.name!r} outlived its agent call by {_THREAD_JOIN_TIMEOUT_S:.0f}s")
