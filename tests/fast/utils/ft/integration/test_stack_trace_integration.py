"""Integration tests for the stack trace diagnostic pipeline.

Tests the full flow: Controller → DiagnosticScheduler → StackTraceDiagnostic
→ StackTraceAggregator, with FakeNodeAgent instances providing configurable
stack trace results.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.fast.utils.ft.helpers import (
    SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
    SAMPLE_PYSPY_OUTPUT_STUCK,
    make_fake_agents,
    make_rank_pids_provider,
    make_trace_result,
    mock_stack_trace_diagnostic,
)

from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.models import ActionType


class TestHangWithStackTraceSuspect:
    """Full pipeline: hang trigger → stack trace identifies suspect → pipeline runs on suspect only."""

    async def test_hang_suspects_from_trace_only_run_gpu_diagnostic(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
                "node-2": {"gpu": False},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100, 1: 101},
                "node-1": {2: 200, 3: 201},
                "node-2": {4: 300, 5: 301},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK),
            ]
        ):
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

            assert decision.action == ActionType.MARK_BAD_AND_RESTART
            assert decision.bad_node_ids == ["node-2"]

    async def test_hang_all_traces_same_runs_pipeline_on_all(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
                "node-2": {"gpu": True},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            ]
        ):
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

            assert decision.action == ActionType.NOTIFY_HUMAN


class TestCrashSkipsStackTrace:
    """Non-hang triggers should skip the stack trace pre-step entirely."""

    async def test_crash_trigger_no_stack_trace(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
            }
        )

        with patch("miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic") as mock_diag_cls:
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="crash",
            )

            mock_diag_cls.assert_not_called()
            assert decision.action == ActionType.MARK_BAD_AND_RESTART
            assert decision.bad_node_ids == ["node-1"]


class TestHangWithCollectionFailure:
    """When stack trace collection fails for a node, that node becomes suspect."""

    async def test_failed_collection_node_is_suspect(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
                "node-2": {"gpu": True},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                make_trace_result("node-1", passed=False, details="py-spy failed"),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            ]
        ):
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

            assert decision.action == ActionType.MARK_BAD_AND_RESTART
            assert "node-1" in decision.bad_node_ids
            assert "node-0" not in decision.bad_node_ids
