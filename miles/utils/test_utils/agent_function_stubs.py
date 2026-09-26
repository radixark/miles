"""Agent functions for the ``--custom-agent-function-mode`` tests.

Module-level so that a fresh process imports them the way it imports a real agent.
"""

import asyncio
import os
import threading
import time
from pathlib import Path


async def report_pid(**kwargs) -> dict:
    return {"pid": os.getpid()}


async def fail(**kwargs) -> None:
    raise ValueError("agent failed on purpose")


async def wait_until_cancelled(*, started: str, cleaned_up: str, **kwargs) -> None:
    Path(started).touch()
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        Path(cleaned_up).touch()
        raise


async def leave_cleanup_thread(*, marker: str, delay_s: float, **kwargs) -> None:
    def _cleanup() -> None:
        time.sleep(delay_s)
        Path(marker).touch()

    threading.Thread(target=_cleanup, daemon=False).start()
