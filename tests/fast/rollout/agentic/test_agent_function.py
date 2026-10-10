"""A ``subproc`` agent call runs in a child process yet behaves like the awaited coroutine."""

import asyncio
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from miles.rollout.agentic import agent_function
from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.rollout_limits import current_scope
from miles.utils.test_utils import agent_function_stubs as stubs


def _wait_for(path: Path, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if not sys.platform.startswith("linux"):
        return True
    # a killed process whose new parent has not reaped it yet still answers signal 0
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


async def _wait_until_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while _is_running(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return not _is_running(pid)


def test_inline_runs_on_the_calling_process():
    assert asyncio.run(call_agent_function(stubs.report_pid, mode="inline"))["pid"] == os.getpid()


def test_subproc_runs_in_a_child_process():
    reported = asyncio.run(call_agent_function(stubs.report_pid, mode="subproc"))
    assert reported["pid"] != os.getpid()
    assert reported["ppid"] == os.getpid()


def test_what_the_agent_prints_does_not_reach_the_result():
    assert asyncio.run(call_agent_function(stubs.print_and_return, mode="subproc")) == "returned by the agent"


def test_subproc_raises_the_agent_error_with_its_traceback():
    with pytest.raises(ValueError, match="agent failed on purpose") as raised:
        asyncio.run(call_agent_function(stubs.fail, mode="subproc"))
    assert "in fail" in str(raised.value.__cause__), "the agent process's traceback should be the cause"


def test_an_error_that_cannot_be_rebuilt_arrives_by_name():
    with pytest.raises(RuntimeError, match="NeedsKeywordError: agent failed on purpose"):
        asyncio.run(call_agent_function(stubs.fail_unrebuildably, mode="subproc"))


def test_an_agent_process_that_exits_without_returning_fails_the_call():
    with pytest.raises(RuntimeError, match="exited with code 3"):
        asyncio.run(call_agent_function(stubs.exit_without_returning, mode="subproc"))


def test_an_agent_that_exits_fails_only_its_own_call():
    """``sys.exit`` in an agent must not reach the rollout process as ``SystemExit``."""
    with pytest.raises(RuntimeError, match="SystemExit"):
        asyncio.run(call_agent_function(stubs.call_sys_exit, mode="subproc"))


def test_the_agent_process_shares_the_callers_rollout_limits():
    assert asyncio.run(call_agent_function(stubs.report_limits_scope, mode="subproc")) == current_scope()


def test_cancelling_the_caller_runs_the_agent_cleanup(tmp_path):
    """A cancelled episode still gets to close what it opened, such as its sandbox."""
    pid_file, cleaned_up = tmp_path / "agent_pid", tmp_path / "cleaned_up"

    async def cancel_and_watch() -> bool:
        call = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled, mode="subproc", pid_file=str(pid_file), cleaned_up=str(cleaned_up)
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return await _wait_until_gone(int(pid_file.read_text()), timeout_s=30)

    assert asyncio.run(asyncio.wait_for(cancel_and_watch(), timeout=120)), "the cancelled agent never exited"
    assert cleaned_up.exists(), "the cancelled agent never ran its cleanup"


def test_an_agent_that_ignores_cancellation_is_killed_after_the_grace_period(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_function, "_CANCEL_GRACE_S", 1.0)
    pid_file = tmp_path / "agent_pid"

    async def cancel_and_watch() -> bool:
        call = asyncio.create_task(
            call_agent_function(stubs.ignore_cancellation, mode="subproc", pid_file=str(pid_file))
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return await _wait_until_gone(int(pid_file.read_text()), timeout_s=30)

    assert asyncio.run(asyncio.wait_for(cancel_and_watch(), timeout=120)), "the agent process outlived the grace"


def test_a_cleanup_thread_finishes_after_the_call_returns(tmp_path):
    """The outcome must not wait for a thread the agent left closing a sandbox, nor may the process cut it short."""
    marker = tmp_path / "cleanup_done"

    async def call_and_watch() -> tuple[bool, bool]:
        pid = await call_agent_function(stubs.leave_cleanup_thread, mode="subproc", marker=str(marker), delay_s=2.0)
        returned_first = not marker.exists()
        return returned_first, await _wait_until_gone(pid, timeout_s=30)

    returned_first, exited = asyncio.run(call_and_watch())
    assert returned_first, "the call waited for the agent's cleanup thread"
    assert exited and marker.exists(), "the agent process exited before its cleanup thread finished"


def test_leftover_cleanup_is_cut_at_its_deadline(monkeypatch):
    monkeypatch.setattr(agent_function, "_LEFTOVER_CLEANUP_S", 1.0)

    async def call_and_watch() -> bool:
        pid = await call_agent_function(stubs.leave_endless_thread, mode="subproc")
        return await _wait_until_gone(pid, timeout_s=30)

    assert asyncio.run(call_and_watch()), "the agent process outlived its leftover-cleanup deadline"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG exists only on Linux")
def test_the_agent_process_dies_with_the_rollout_process(tmp_path):
    pid_file = tmp_path / "agent_pid"
    rollout = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import asyncio
                from miles.rollout.agentic.agent_function import call_agent_function
                from miles.utils.test_utils import agent_function_stubs as stubs
                asyncio.run(call_agent_function(stubs.ignore_cancellation, mode="subproc", pid_file={str(pid_file)!r}))
                """
            ),
        ]
    )
    try:
        assert _wait_for(pid_file), "the agent process never started"
    finally:
        rollout.kill()
        rollout.wait()
    assert asyncio.run(_wait_until_gone(int(pid_file.read_text()), timeout_s=10)), "the agent outlived its rollout"
