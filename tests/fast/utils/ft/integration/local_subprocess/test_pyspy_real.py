"""Real py-spy integration tests using plain subprocess (no Ray).

Spawns Python child processes that deliberately block at different points,
then runs StackTraceDiagnostic + StackTraceAggregator against the live
PIDs to verify the full diagnostic pipeline with real stack traces.

Requires py-spy to be installed and the current user to have ptrace
permissions (root, CAP_SYS_PTRACE, or kernel.yama.ptrace_scope=0).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable

import pytest

from miles.utils.ft.agents.diagnostics.stack_trace import PySpyThread, StackTraceDiagnostic
from miles.utils.ft.controller.diagnostics.stack_trace import StackTraceAggregator

_HAS_PYSPY = shutil.which("py-spy") is not None

pytestmark = [
    pytest.mark.skipif(not _HAS_PYSPY, reason="py-spy not installed"),
    pytest.mark.anyio,
]

WORKER_SCRIPT = textwrap.dedent(
    """\
    import sys, time, threading

    def step_communication():
        time.sleep(3600)

    def step_lock_wait():
        lock = threading.Lock()
        lock.acquire()
        lock.acquire()

    mode = sys.argv[1]
    {"sleep": step_communication, "lock": step_lock_wait}[mode]()
"""
)

_SETTLE_SECONDS = 0.5


def _parse_threads(details: str) -> list[PySpyThread]:
    return [PySpyThread.model_validate(t) for t in json.loads(details)]


@pytest.fixture
def blocked_process() -> Callable[[str], int]:
    """Factory fixture: start a subprocess blocked in the given mode, return its PID."""
    procs: list[subprocess.Popen[bytes]] = []

    def _start(mode: str) -> int:
        proc = subprocess.Popen([sys.executable, "-c", WORKER_SCRIPT, mode])
        procs.append(proc)
        time.sleep(_SETTLE_SECONDS)
        return proc.pid

    yield _start  # type: ignore[misc]

    for proc in procs:
        proc.kill()
        proc.wait()


class TestStackTraceDiagnosticReal:
    async def test_single_blocked_process_captures_trace(
        self,
        blocked_process: Callable[[str], int],
    ) -> None:
        pid = blocked_process("sleep")

        diagnostic = StackTraceDiagnostic(pids=[pid])
        result = await diagnostic.run(node_id="test-node", timeout_seconds=10)

        assert result.passed is True
        threads = _parse_threads(result.details)
        assert len(threads) > 0
        assert any(t.frames for t in threads)

        all_frame_names = [f.name for t in threads for f in t.frames]
        assert any(
            "step_communication" in name for name in all_frame_names
        ), f"Expected a frame containing 'step_communication', got: {all_frame_names}"

    async def test_nonexistent_pid_fails_gracefully(self) -> None:
        diagnostic = StackTraceDiagnostic(pids=[999999])
        result = await diagnostic.run(node_id="test-node", timeout_seconds=10)

        assert result.passed is False
        assert json.loads(result.details) == []


class TestStackTraceAggregatorReal:
    async def test_aggregator_identifies_minority_outlier(
        self,
        blocked_process: Callable[[str], int],
    ) -> None:
        """2x sleep + 1x lock: the lock process should be flagged as suspect."""
        pid_sleep_0 = blocked_process("sleep")
        pid_sleep_1 = blocked_process("sleep")
        pid_lock = blocked_process("lock")

        diag_sleep_0 = StackTraceDiagnostic(pids=[pid_sleep_0])
        diag_sleep_1 = StackTraceDiagnostic(pids=[pid_sleep_1])
        diag_lock = StackTraceDiagnostic(pids=[pid_lock])

        result_0 = await diag_sleep_0.run(node_id="node-0", timeout_seconds=10)
        result_1 = await diag_sleep_1.run(node_id="node-1", timeout_seconds=10)
        result_2 = await diag_lock.run(node_id="node-2", timeout_seconds=10)

        assert result_0.passed is True
        assert result_1.passed is True
        assert result_2.passed is True

        aggregator = StackTraceAggregator()
        traces = {
            "node-0": _parse_threads(result_0.details),
            "node-1": _parse_threads(result_1.details),
            "node-2": _parse_threads(result_2.details),
        }
        agg_result = aggregator.aggregate(traces=traces)

        assert agg_result.suspect_node_ids == ["node-2"], (
            f"Expected node-2 as suspect, got: {agg_result.suspect_node_ids}. "
            f"Fingerprint groups: {agg_result.fingerprint_groups}"
        )

    async def test_aggregator_returns_empty_when_all_same(
        self,
        blocked_process: Callable[[str], int],
    ) -> None:
        """3x sleep: no suspects."""
        pid_0 = blocked_process("sleep")
        pid_1 = blocked_process("sleep")
        pid_2 = blocked_process("sleep")

        diag_0 = StackTraceDiagnostic(pids=[pid_0])
        diag_1 = StackTraceDiagnostic(pids=[pid_1])
        diag_2 = StackTraceDiagnostic(pids=[pid_2])

        result_0 = await diag_0.run(node_id="node-0", timeout_seconds=10)
        result_1 = await diag_1.run(node_id="node-1", timeout_seconds=10)
        result_2 = await diag_2.run(node_id="node-2", timeout_seconds=10)

        assert result_0.passed is True
        assert result_1.passed is True
        assert result_2.passed is True

        aggregator = StackTraceAggregator()
        traces = {
            "node-0": _parse_threads(result_0.details),
            "node-1": _parse_threads(result_1.details),
            "node-2": _parse_threads(result_2.details),
        }
        agg_result = aggregator.aggregate(traces=traces)

        assert agg_result.suspect_node_ids == [], (
            f"Expected no suspects, got: {agg_result.suspect_node_ids}. "
            f"Fingerprint groups: {agg_result.fingerprint_groups}"
        )
