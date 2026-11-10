"""Sandbox start functions: a ``StartFn`` starts one sandbox and returns ``(close_fn, base_url)``."""

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

# A provider's "start one sandbox" hook: (task_id, tasks_dir) -> (close_fn, base_url).
StartFn = Callable[[str, str], tuple[Callable[[], None], str]]


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

        # not a daemon: a subproc agent call waits for it before its process exits
        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=False).start()
        raise
