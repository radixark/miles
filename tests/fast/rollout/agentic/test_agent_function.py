"""A ``subproc`` agent call runs apart from its caller yet behaves like the awaited coroutine."""

import asyncio
import os
import time
from pathlib import Path

import pytest

from miles.rollout.agentic.agent_function import call_agent_function
from miles.utils.test_utils import agent_function_stubs as stubs


def _wait_for(path: Path, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def test_inline_runs_on_the_calling_process():
    assert asyncio.run(call_agent_function(stubs.report_pid, mode="inline"))["pid"] == os.getpid()


def test_subproc_runs_in_another_process(ray_local_mode):
    assert asyncio.run(call_agent_function(stubs.report_pid, mode="subproc"))["pid"] != os.getpid()


def test_subproc_raises_the_agent_error(ray_local_mode):
    with pytest.raises(ValueError, match="agent failed on purpose"):
        asyncio.run(call_agent_function(stubs.fail, mode="subproc"))


def test_cancelling_the_caller_runs_the_agent_cleanup(ray_local_mode, tmp_path):
    """A cancelled episode still gets to close what it opened, such as its sandbox."""
    started, cleaned_up = tmp_path / "started", tmp_path / "cleaned_up"

    async def cancel_once_started() -> None:
        call = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled, mode="subproc", started=str(started), cleaned_up=str(cleaned_up)
            )
        )
        while not started.exists():
            await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(asyncio.wait_for(cancel_once_started(), timeout=120))
    assert _wait_for(cleaned_up, timeout_s=10), "the cancelled agent never ran its cleanup"


def test_a_cleanup_thread_finishes_before_the_call_returns(ray_local_mode, tmp_path):
    """The process must not exit while an agent's cleanup thread is still closing a sandbox."""
    marker = tmp_path / "cleanup_done"
    asyncio.run(call_agent_function(stubs.leave_cleanup_thread, mode="subproc", marker=str(marker), delay_s=1.0))
    assert marker.exists()
