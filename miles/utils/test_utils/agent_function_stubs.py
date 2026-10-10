"""Agent functions for the ``--custom-agent-function-mode`` tests.

Module-level so that a fresh process imports them the way it imports a real agent.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

from miles.rollout.agentic.rollout_limits import current_scope, rollout_lock, rollout_semaphore


class NeedsKeywordError(Exception):
    """An error that pickles but cannot be rebuilt from its args, like many HTTP client errors."""

    def __init__(self, message: str, *, detail: str):
        super().__init__(message)
        self.detail = detail


async def report_pid(**kwargs) -> dict:
    return {"pid": os.getpid(), "ppid": os.getppid()}


async def print_and_return(**kwargs) -> str:
    print("printed by the agent")
    return "returned by the agent"


async def report_limits_scope(**kwargs) -> str:
    return current_scope()


async def fail(**kwargs) -> None:
    raise ValueError("agent failed on purpose")


async def fail_unrebuildably(**kwargs) -> None:
    raise NeedsKeywordError("agent failed on purpose", detail="lost on the way")


async def exit_without_returning(**kwargs) -> None:
    os._exit(3)


async def call_sys_exit(**kwargs) -> None:
    sys.exit(2)


async def wait_until_cancelled(*, pid_file: str, cleaned_up: str, **kwargs) -> None:
    Path(pid_file).write_text(str(os.getpid()))
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        Path(cleaned_up).touch()
        raise


async def ignore_cancellation(*, pid_file: str, **kwargs) -> None:
    Path(pid_file).write_text(str(os.getpid()))
    while True:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass


async def leave_cleanup_thread(*, marker: str, delay_s: float, **kwargs) -> int:
    def _cleanup() -> None:
        time.sleep(delay_s)
        Path(marker).touch()

    threading.Thread(target=_cleanup, daemon=False).start()
    return os.getpid()


async def leave_endless_thread(**kwargs) -> int:
    threading.Thread(target=time.sleep, args=(3600,), daemon=False).start()
    return os.getpid()


async def hold_semaphore(*, name: str, limit: int, hold_s: float, **kwargs) -> tuple[float, float]:
    async with rollout_semaphore(name, limit):
        start = time.monotonic()
        await asyncio.sleep(hold_s)
        return start, time.monotonic()


async def hold_lock(*, name: str, hold_s: float, **kwargs) -> tuple[float, float]:
    with rollout_lock(name):
        start = time.monotonic()
        time.sleep(hold_s)
        return start, time.monotonic()


async def die_holding_semaphore(*, name: str, **kwargs) -> None:
    async with rollout_semaphore(name, 1):
        os._exit(1)
