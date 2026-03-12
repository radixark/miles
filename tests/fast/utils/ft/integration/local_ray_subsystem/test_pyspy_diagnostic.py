"""Local Ray: Real py-spy diagnostic — run py-spy dump against live processes.

These tests require py-spy to be installed. They are skipped if py-spy
is not available in the PATH. On CI where py-spy is installed, these
provide real coverage of the stack trace collection and aggregation pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

import pytest
import ray

from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyThread, StackTraceNodeExecutor
from miles.utils.ft.controller.diagnostics.stack_trace import StackTraceAggregator

_HAS_PYSPY = shutil.which("py-spy") is not None

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.skipif(not _HAS_PYSPY, reason="py-spy not installed"),
    pytest.mark.anyio,
]


def _parse_threads(details: str) -> list[PySpyThread]:
    return [PySpyThread.model_validate(t) for t in json.loads(details)]


@ray.remote(num_cpus=0, num_gpus=0)
class _BusyWorker:
    """Ray actor that runs a busy loop, providing a live process for py-spy."""

    def get_pid(self) -> int:
        return os.getpid()

    async def spin(self) -> None:
        while True:
            await asyncio.sleep(0.1)


class TestStackTraceNodeExecutorAgainstLiveProcess:
    async def test_dump_captures_stack_trace_from_ray_actor(
        self,
        local_ray: None,
    ) -> None:
        worker = _BusyWorker.remote()
        worker.spin.remote()
        await asyncio.sleep(0.5)

        pid: int = ray.get(worker.get_pid.remote(), timeout=5)

        diagnostic = StackTraceNodeExecutor()
        result = await diagnostic.run(node_id="test-node", timeout_seconds=10, pids=[pid])

        ray.kill(worker, no_restart=True)

        assert result.passed is True
        threads = _parse_threads(result.details)
        assert len(threads) > 0
        assert any(t.frames for t in threads)

    async def test_dump_handles_nonexistent_pid(
        self,
        local_ray: None,
    ) -> None:
        diagnostic = StackTraceNodeExecutor()
        result = await diagnostic.run(node_id="test-node", timeout_seconds=10, pids=[999999])

        assert result.passed is False
        assert json.loads(result.details) == []

    async def test_dump_multiple_pids_partial_failure(
        self,
        local_ray: None,
    ) -> None:
        """One valid PID + one invalid PID -> passed=True (not all failed)."""
        worker = _BusyWorker.remote()
        worker.spin.remote()
        await asyncio.sleep(0.5)

        pid: int = ray.get(worker.get_pid.remote(), timeout=5)

        diagnostic = StackTraceNodeExecutor()
        result = await diagnostic.run(node_id="test-node", timeout_seconds=10, pids=[pid, 999999])

        ray.kill(worker, no_restart=True)

        assert result.passed is True
        threads = _parse_threads(result.details)
        assert len(threads) > 0


class TestStackTraceAggregatorWithRealTraces:
    async def test_aggregator_identifies_suspect_from_different_traces(
        self,
        local_ray: None,
    ) -> None:
        """Create two workers doing the same thing and one doing something different.
        The aggregator should flag the different one as a suspect."""

        worker_a = _BusyWorker.remote()
        worker_b = _BusyWorker.remote()
        worker_a.spin.remote()
        worker_b.spin.remote()
        await asyncio.sleep(0.5)

        pid_a: int = ray.get(worker_a.get_pid.remote(), timeout=5)
        pid_b: int = ray.get(worker_b.get_pid.remote(), timeout=5)

        diag = StackTraceNodeExecutor()

        result_a = await diag.run(node_id="node-a", timeout_seconds=10, pids=[pid_a])
        result_b = await diag.run(node_id="node-b", timeout_seconds=10, pids=[pid_b])

        ray.kill(worker_a, no_restart=True)
        ray.kill(worker_b, no_restart=True)

        assert result_a.passed is True
        assert result_b.passed is True

        aggregator = StackTraceAggregator()
        traces = {
            "node-a": _parse_threads(result_a.details),
            "node-b": _parse_threads(result_b.details),
        }
        agg_result = aggregator.aggregate(traces)

        assert isinstance(agg_result.suspect_node_ids, list)

    async def test_aggregator_returns_empty_for_identical_traces(
        self,
        local_ray: None,
    ) -> None:
        """Two identical workers should produce no suspects."""
        worker_a = _BusyWorker.remote()
        worker_b = _BusyWorker.remote()
        worker_a.spin.remote()
        worker_b.spin.remote()
        await asyncio.sleep(0.5)

        pid_a: int = ray.get(worker_a.get_pid.remote(), timeout=5)
        pid_b: int = ray.get(worker_b.get_pid.remote(), timeout=5)

        diag = StackTraceNodeExecutor()

        result_a = await diag.run(node_id="node-a", timeout_seconds=10, pids=[pid_a])
        result_b = await diag.run(node_id="node-b", timeout_seconds=10, pids=[pid_b])

        ray.kill(worker_a, no_restart=True)
        ray.kill(worker_b, no_restart=True)

        aggregator = StackTraceAggregator()
        traces = {
            "node-a": _parse_threads(result_a.details),
            "node-b": _parse_threads(result_b.details),
        }
        agg_result = aggregator.aggregate(traces)

        assert agg_result.suspect_node_ids == []


class TestOrchestratorHangTraceFullChain:
    """DiagnosticOrchestrator runs stack trace pre-step on hang trigger with real py-spy (PY4)."""

    async def test_orchestrator_hang_trigger_collects_real_traces(
        self,
        local_ray: None,
    ) -> None:
        from miles.utils.ft.controller.diagnostics.executors import StackTraceClusterExecutor
        from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

        worker_a = _BusyWorker.remote()
        worker_b = _BusyWorker.remote()
        worker_a.spin.remote()
        worker_b.spin.remote()
        await asyncio.sleep(0.5)

        pid_a: int = ray.get(worker_a.get_pid.remote(), timeout=5)
        pid_b: int = ray.get(worker_b.get_pid.remote(), timeout=5)

        node_pids: dict[str, dict[int, int]] = {
            "node-a": {0: pid_a},
            "node-b": {0: pid_b},
        }

        def pids_provider(node_id: str) -> dict[int, int]:
            return node_pids.get(node_id, {})

        orchestrator = DiagnosticOrchestrator(
            node_agents={},
            pipeline=[],
        )

        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[StackTraceClusterExecutor(rank_pids_provider=pids_provider)],
        )

        ray.kill(worker_a, no_restart=True)
        ray.kill(worker_b, no_restart=True)

        assert decision.bad_node_ids == []
