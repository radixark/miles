"""Integration tests for the stack trace diagnostic pipeline.

Tests the full flow: Controller → DiagnosticScheduler → StackTraceDiagnostic
→ StackTraceAggregator, with FakeNodeAgent instances providing configurable
stack trace results.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.models import ActionType, DiagnosticResult
from tests.fast.utils.ft.conftest import FakeNodeAgent, make_fake_agents
from tests.fast.utils.ft.test_stack_trace import (
    SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
    SAMPLE_PYSPY_OUTPUT_NORMAL,
    SAMPLE_PYSPY_OUTPUT_STUCK,
)


def _make_rank_pids_provider(
    mapping: dict[str, dict[int, int]],
) -> "Callable[[str], dict[int, int]]":
    from collections.abc import Callable

    def provider(node_id: str) -> dict[int, int]:
        return mapping.get(node_id, {})

    return provider


def _make_trace_result(
    node_id: str,
    passed: bool = True,
    details: str = "trace output",
) -> DiagnosticResult:
    return DiagnosticResult(
        diagnostic_type="stack_trace",
        node_id=node_id,
        passed=passed,
        details=details,
    )


class TestHangWithStackTraceSuspect:
    """Full pipeline: hang trigger → stack trace identifies suspect → pipeline runs on suspect only."""

    @pytest.mark.asyncio
    async def test_hang_suspects_from_trace_only_run_gpu_diagnostic(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
            "node-2": {"gpu": False},
        })
        pids_provider = _make_rank_pids_provider({
            "node-0": {0: 100, 1: 101},
            "node-1": {2: 200, 3: 201},
            "node-2": {4: 300, 5: 301},
        })

        with patch(
            "miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic"
        ) as mock_diag_cls:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(side_effect=[
                _make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                _make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                _make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK),
            ])
            mock_diag_cls.return_value = mock_instance

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

    @pytest.mark.asyncio
    async def test_hang_all_traces_same_runs_pipeline_on_all(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
            "node-2": {"gpu": True},
        })
        pids_provider = _make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
            "node-2": {2: 300},
        })

        with patch(
            "miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic"
        ) as mock_diag_cls:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(side_effect=[
                _make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                _make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                _make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            ])
            mock_diag_cls.return_value = mock_instance

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

    @pytest.mark.asyncio
    async def test_crash_trigger_no_stack_trace(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
        })
        pids_provider = _make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
        })

        with patch(
            "miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic"
        ) as mock_diag_cls:
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

    @pytest.mark.asyncio
    async def test_failed_collection_node_is_suspect(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
            "node-2": {"gpu": True},
        })
        pids_provider = _make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
            "node-2": {2: 300},
        })

        with patch(
            "miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic"
        ) as mock_diag_cls:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(side_effect=[
                _make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
                _make_trace_result("node-1", passed=False, details="py-spy failed"),
                _make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            ])
            mock_diag_cls.return_value = mock_instance

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
