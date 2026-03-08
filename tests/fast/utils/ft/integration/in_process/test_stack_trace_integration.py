"""Integration tests for the stack trace diagnostic pipeline.

Tests the full flow: DiagnosticOrchestrator with StackTraceClusterExecutor as pre_executor
-> collect_stack_trace_suspects -> call_agent_diagnostic -> FakeNodeAgent -> StackTraceAggregator,
verifying that the RPC path works end-to-end with deterministic per-node results.
"""

from __future__ import annotations

from tests.fast.utils.ft.utils import (
    SAMPLE_PYSPY_JSON_DIFFERENT_STUCK,
    SAMPLE_PYSPY_JSON_STUCK,
    make_rank_pids_provider,
    make_trace_result,
)
from tests.fast.utils.ft.utils.diagnostic_fakes import FakeNodeAgent

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.diagnostics.executors import GpuClusterExecutor, StackTraceClusterExecutor
from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator


def _make_agent(
    node_id: str,
    *,
    gpu_passed: bool = True,
    trace_result: DiagnosticResult | None = None,
) -> FakeNodeAgent:
    results: dict[str, DiagnosticResult] = {
        "gpu": DiagnosticResult(
            diagnostic_type="gpu",
            node_id=node_id,
            passed=gpu_passed,
            details="pass" if gpu_passed else "fail",
        ),
    }
    if trace_result is not None:
        results["stack_trace"] = trace_result
    return FakeNodeAgent(node_id=node_id, diagnostic_results=results)


class TestHangWithStackTraceSuspect:
    """Full pipeline: StackTraceClusterExecutor identifies outlier -> evicted immediately."""

    async def test_hang_suspects_from_trace_evicted_directly(self) -> None:
        agents = {
            "node-0": _make_agent("node-0", gpu_passed=True, trace_result=make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
            "node-1": _make_agent("node-1", gpu_passed=True, trace_result=make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
            "node-2": _make_agent("node-2", gpu_passed=False, trace_result=make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_DIFFERENT_STUCK)),
        }
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100, 1: 101},
                "node-1": {2: 200, 3: 201},
                "node-2": {4: 300, 5: 301},
            }
        )

        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[StackTraceClusterExecutor(rank_pids_provider=pids_provider)],
        )

        assert len(decision.bad_node_ids) > 0
        assert decision.bad_node_ids == ["node-2"]

    async def test_hang_all_traces_same_falls_through_to_gpu(self) -> None:
        agents = {
            "node-0": _make_agent("node-0", gpu_passed=True, trace_result=make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
            "node-1": _make_agent("node-1", gpu_passed=True, trace_result=make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
            "node-2": _make_agent("node-2", gpu_passed=True, trace_result=make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
        }
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[StackTraceClusterExecutor(rank_pids_provider=pids_provider)],
        )

        assert decision.bad_node_ids == []


class TestCrashSkipsStackTrace:
    """Without StackTraceClusterExecutor in pre_executors, no stack trace analysis runs."""

    async def test_crash_trigger_no_stack_trace(self) -> None:
        agents = {
            "node-0": _make_agent("node-0", gpu_passed=True),
            "node-1": _make_agent("node-1", gpu_passed=False),
        }
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert len(decision.bad_node_ids) > 0
        assert decision.bad_node_ids == ["node-1"]


class TestHangWithCollectionFailure:
    """When stack trace collection fails for a node, that node is evicted."""

    async def test_failed_collection_node_evicted(self) -> None:
        agents = {
            "node-0": _make_agent("node-0", gpu_passed=True, trace_result=make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
            "node-1": _make_agent("node-1", gpu_passed=False, trace_result=make_trace_result("node-1", passed=False, details="py-spy failed")),
            "node-2": _make_agent("node-2", gpu_passed=True, trace_result=make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_STUCK)),
        }
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[StackTraceClusterExecutor(rank_pids_provider=pids_provider)],
        )

        assert len(decision.bad_node_ids) > 0
        assert "node-1" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids
