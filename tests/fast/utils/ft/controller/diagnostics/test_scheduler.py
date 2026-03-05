"""Tests for DiagnosticScheduler and BaseDiagnostic."""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.diagnostics.inter_machine_orchestrator import (
    PairResult,
    cross_compare,
)
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.models import ActionType, DiagnosticResult
from tests.fast.utils.ft.helpers import (
    FakeNodeAgent,
    SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
    SAMPLE_PYSPY_OUTPUT_STUCK,
    StubDiagnostic,
    make_fake_agents,
    make_rank_pids_provider,
    make_trace_result,
    mock_stack_trace_diagnostic,
)


# ---------------------------------------------------------------------------
# BaseDiagnostic tests
# ---------------------------------------------------------------------------


class _ConcreteDiagnostic(BaseDiagnostic):
    diagnostic_type = "test_concrete"

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=True,
            details="ok",
        )


class TestBaseDiagnostic:
    @pytest.mark.anyio
    async def test_concrete_subclass_can_run(self) -> None:
        diag = _ConcreteDiagnostic()
        result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert result.node_id == "node-0"
        assert result.diagnostic_type == "test_concrete"

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDiagnostic()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DiagnosticScheduler tests
# ---------------------------------------------------------------------------


class TestDiagnosticSchedulerEmptyPipeline:
    @pytest.mark.anyio
    async def test_empty_pipeline_returns_notify_human(self) -> None:
        agents = make_fake_agents({"node-0": {}})
        scheduler = DiagnosticScheduler(agents=agents, pipeline=[])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_no_pipeline_arg_returns_notify_human(self) -> None:
        scheduler = DiagnosticScheduler(agents={})
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="hang")

        assert decision.action == ActionType.NOTIFY_HUMAN


class TestDiagnosticSchedulerSingleStep:
    @pytest.mark.anyio
    async def test_all_pass_returns_notify_human(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
        })
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_one_node_fails_returns_mark_bad(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
        })
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-1" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_all_nodes_fail_returns_mark_bad(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": False},
            "node-1": {"gpu": False},
        })
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert sorted(decision.bad_node_ids) == ["node-0", "node-1"]


class TestDiagnosticSchedulerMultiStep:
    @pytest.mark.anyio
    async def test_first_step_catches_bad_node(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": False, "intra": True},
            "node-1": {"gpu": True, "intra": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["gpu", "intra"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert decision.bad_node_ids == ["node-0"]
        assert "gpu" in decision.reason

    @pytest.mark.anyio
    async def test_first_passes_second_catches(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True, "intra": True},
            "node-1": {"gpu": True, "intra": False},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["gpu", "intra"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert decision.bad_node_ids == ["node-1"]
        assert "intra" in decision.reason

    @pytest.mark.anyio
    async def test_all_steps_pass_returns_notify(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True, "intra": True},
            "node-1": {"gpu": True, "intra": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["gpu", "intra"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN


class TestDiagnosticSchedulerErrorHandling:
    @pytest.mark.anyio
    async def test_agent_exception_treated_as_failure(self) -> None:
        class _RaisingAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120,
            ) -> DiagnosticResult:
                raise RuntimeError("agent crashed")

        good_agents = make_fake_agents({"node-0": {"gpu": True}})
        agents = {
            **good_agents,
            "node-1": _RaisingAgent(),
        }
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-1" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_trigger_reason_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        agents: dict[str, FakeNodeAgent] = {}
        scheduler = DiagnosticScheduler(agents=agents, pipeline=[])

        with caplog.at_level(logging.INFO):
            await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
                suspect_node_ids=["node-0"],
            )

        assert "hang" in caplog.text

    @pytest.mark.anyio
    async def test_no_agents_returns_notify(self) -> None:
        scheduler = DiagnosticScheduler(agents={}, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_suspect_node_ids_limits_scope(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": False},
            "node-1": {"gpu": False},
            "node-2": {"gpu": True},
        })
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(
            trigger_reason="crash",
            suspect_node_ids=["node-0", "node-2"],
        )

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids
        assert "node-1" not in decision.bad_node_ids
        assert "node-2" not in decision.bad_node_ids


class TestNodeAgentDynamicDiagnostics:
    """Test FtNodeAgent.set_diagnostic / remove_diagnostic."""

    @pytest.mark.anyio
    async def test_set_diagnostic_overrides_existing(self) -> None:
        original = StubDiagnostic(passed=True, details="original")
        agent = FtNodeAgent(node_id="node-0", diagnostics=[original])

        try:
            result1 = await agent.run_diagnostic("stub")
            assert result1.passed is True
            assert result1.details == "original"

            override = StubDiagnostic(passed=False, details="override")
            agent.set_diagnostic(override)

            result2 = await agent.run_diagnostic("stub")
            assert result2.passed is False
            assert result2.details == "override"
        finally:
            await agent.stop()

    @pytest.mark.anyio
    async def test_remove_diagnostic(self) -> None:
        from miles.utils.ft.models import UnknownDiagnosticError

        stub = StubDiagnostic(passed=True)
        agent = FtNodeAgent(node_id="node-0", diagnostics=[stub])

        try:
            result1 = await agent.run_diagnostic("stub")
            assert result1.passed is True

            agent.remove_diagnostic("stub")

            with pytest.raises(UnknownDiagnosticError, match="unknown diagnostic type"):
                await agent.run_diagnostic("stub")
        finally:
            await agent.stop()

    @pytest.mark.anyio
    async def test_remove_nonexistent_is_noop(self) -> None:
        agent = FtNodeAgent(node_id="node-0")
        try:
            agent.remove_diagnostic("nonexistent")
        finally:
            await agent.stop()


class TestDiagnosticSchedulerInterMachine:
    """Tests for the inter-machine diagnostic step with cross-comparison."""

    @pytest.mark.anyio
    async def test_inter_machine_all_pass(self) -> None:
        agents = make_fake_agents({
            "node-0": {"inter_machine": True},
            "node-1": {"inter_machine": True},
            "node-2": {"inter_machine": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")
        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_inter_machine_one_bad_node(self) -> None:
        agents = make_fake_agents({
            "node-0": {"inter_machine": False},
            "node-1": {"inter_machine": True},
            "node-2": {"inter_machine": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_inter_machine_cannot_localize_all_fail(self) -> None:
        agents = make_fake_agents({
            "node-0": {"inter_machine": False},
            "node-1": {"inter_machine": False},
            "node-2": {"inter_machine": False},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")
        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_inter_machine_two_nodes_pair_fails(self) -> None:
        agents = make_fake_agents({
            "node-0": {"inter_machine": False},
            "node-1": {"inter_machine": False},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")
        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_inter_machine_single_node_skipped(self) -> None:
        agents = make_fake_agents({"node-0": {"inter_machine": True}})
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")
        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_inter_machine_pairing_is_round_robin(self) -> None:
        node_ids = ["node-0", "node-1", "node-2", "node-3"]
        expected_pairs = [
            ("node-0", "node-1"),
            ("node-1", "node-2"),
            ("node-2", "node-3"),
            ("node-3", "node-0"),
        ]
        pairs = [
            (node_ids[i], node_ids[(i + 1) % len(node_ids)])
            for i in range(len(node_ids))
        ]
        assert pairs == expected_pairs

    def test_inter_machine_port_assignment(self) -> None:
        from miles.utils.ft.controller.diagnostics.inter_machine_orchestrator import (
            _BASE_PORT,
        )

        assert _BASE_PORT == 29500

    @pytest.mark.anyio
    async def test_inter_machine_agent_exception(self) -> None:
        class _RaisingInterMachineAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120,
                **kwargs: object,
            ) -> DiagnosticResult:
                if diagnostic_type == "inter_machine":
                    raise asyncio.TimeoutError("timed out")
                return DiagnosticResult(
                    diagnostic_type=diagnostic_type,
                    node_id="node-1",
                    passed=True,
                    details="pass",
                )

        good_agents = make_fake_agents({
            "node-0": {"inter_machine": True},
            "node-2": {"inter_machine": True},
        })
        agents = {**good_agents, "node-1": _RaisingInterMachineAgent()}  # type: ignore[dict-item]
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-1" in decision.bad_node_ids


class TestCrossCompare:
    """Direct unit tests for cross_compare."""

    def test_all_pass(self) -> None:
        result = cross_compare(
            node_ids=["A", "B", "C"],
            pair_results=[
                PairResult(master_id="A", worker_id="B", passed=True),
                PairResult(master_id="B", worker_id="C", passed=True),
                PairResult(master_id="C", worker_id="A", passed=True),
            ],
        )
        assert result == []

    def test_single_bad_node(self) -> None:
        result = cross_compare(
            node_ids=["A", "B", "C"],
            pair_results=[
                PairResult(master_id="A", worker_id="B", passed=False),
                PairResult(master_id="B", worker_id="C", passed=True),
                PairResult(master_id="C", worker_id="A", passed=False),
            ],
        )
        assert result == ["A"]

    def test_all_equal_failure_count_cannot_localize(self) -> None:
        result = cross_compare(
            node_ids=["A", "B", "C"],
            pair_results=[
                PairResult(master_id="A", worker_id="B", passed=False),
                PairResult(master_id="B", worker_id="C", passed=False),
                PairResult(master_id="C", worker_id="A", passed=False),
            ],
        )
        assert result == []

    def test_multiple_bad_nodes_with_highest_count(self) -> None:
        result = cross_compare(
            node_ids=["A", "B", "C", "D"],
            pair_results=[
                PairResult(master_id="A", worker_id="B", passed=False),
                PairResult(master_id="B", worker_id="C", passed=False),
                PairResult(master_id="C", worker_id="D", passed=True),
                PairResult(master_id="D", worker_id="A", passed=False),
            ],
        )
        assert result == ["A", "B"]

    def test_empty_pair_results(self) -> None:
        result = cross_compare(
            node_ids=["A", "B"],
            pair_results=[],
        )
        assert result == []


class TestDiagnosticSchedulerLiveAgents:
    """Test scheduler with real FtNodeAgent instances (not FakeNodeAgent)."""

    @pytest.mark.anyio
    async def test_scheduler_with_real_node_agents(self) -> None:
        stub = StubDiagnostic(passed=True, details="all good")
        agent0 = FtNodeAgent(node_id="node-0", diagnostics=[stub])
        agent1 = FtNodeAgent(node_id="node-1", diagnostics=[stub])

        agents = {"node-0": agent0, "node-1": agent1}
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["stub"])

        try:
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="crash",
            )
            assert decision.action == ActionType.NOTIFY_HUMAN
        finally:
            await agent0.stop()
            await agent1.stop()

    @pytest.mark.anyio
    async def test_scheduler_with_mismatched_diagnostic_type(self) -> None:
        good = StubDiagnostic(passed=True)
        bad = StubDiagnostic(passed=False, details="gpu broken", diagnostic_type="failing")
        agent0 = FtNodeAgent(node_id="node-0", diagnostics=[good])
        agent1 = FtNodeAgent(node_id="node-1", diagnostics=[bad])

        agents = {"node-0": agent0, "node-1": agent1}
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["stub"],
        )

        try:
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="crash",
            )
            # node-1 has diagnostic_type="failing", not "stub",
            # so it gets "unknown diagnostic type" → treated as fail
            assert decision.action == ActionType.MARK_BAD_AND_RESTART
            assert "node-1" in decision.bad_node_ids
        finally:
            await agent0.stop()
            await agent1.stop()


# ---------------------------------------------------------------------------
# Stack trace pre-step tests
# ---------------------------------------------------------------------------


class TestStackTracePreStep:
    @pytest.mark.anyio
    async def test_hang_trigger_runs_stack_trace(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
        })
        pids_provider = make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
        })

        with mock_stack_trace_diagnostic([
            make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
        ]) as mock_diag_cls:
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

            assert mock_diag_cls.call_count == 2
            assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_crash_trigger_skips_stack_trace(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
        })
        pids_provider = make_rank_pids_provider({
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
            assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_no_rank_pids_provider_skips_stack_trace(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
        })

        with patch(
            "miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic"
        ) as mock_diag_cls:
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

            mock_diag_cls.assert_not_called()
            assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_stack_trace_suspect_limits_pipeline_scope(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": False},
            "node-1": {"gpu": False},
            "node-2": {"gpu": False},
        })
        pids_provider = make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
            "node-2": {2: 300},
        })

        with mock_stack_trace_diagnostic([
            make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK),
        ]):
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

    @pytest.mark.anyio
    async def test_stack_trace_no_suspect_runs_on_all(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
        })
        pids_provider = make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
        })

        with mock_stack_trace_diagnostic([
            make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
        ]):
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

            assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.anyio
    async def test_collection_failure_makes_node_suspect(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
        })
        pids_provider = make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
        })

        with mock_stack_trace_diagnostic([
            make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-1", passed=False, details="failed to collect"),
        ]):
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

    @pytest.mark.anyio
    async def test_stack_trace_exception_makes_node_suspect(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
            "node-2": {"gpu": True},
        })
        pids_provider = make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
            "node-2": {2: 300},
        })

        with mock_stack_trace_diagnostic([
            make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            RuntimeError("py-spy crashed"),
            make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
        ]):
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

    @pytest.mark.anyio
    async def test_rank_pids_provider_exception_isolates_to_one_node(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": False},
            "node-1": {"gpu": True},
            "node-2": {"gpu": True},
        })

        def raising_provider(node_id: str) -> dict[int, int]:
            if node_id == "node-0":
                raise RuntimeError("cannot query pids for node-0")
            return {0: 100}

        with mock_stack_trace_diagnostic([
            make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
        ]):
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=raising_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
            )

        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-0" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_hang_merges_trace_suspects_with_existing(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
            "node-2": {"gpu": False},
        })
        pids_provider = make_rank_pids_provider({
            "node-0": {0: 100},
            "node-1": {1: 200},
            "node-2": {2: 300},
        })

        with mock_stack_trace_diagnostic([
            make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_OUTPUT_STUCK),
            make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK),
        ]):
            scheduler = DiagnosticScheduler(
                agents=agents,
                pipeline=["gpu"],
                rank_pids_provider=pids_provider,
            )
            decision = await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
                suspect_node_ids=["node-1"],
            )

            assert decision.action == ActionType.MARK_BAD_AND_RESTART
            assert "node-1" in decision.bad_node_ids
            assert "node-2" in decision.bad_node_ids
            assert "node-0" not in decision.bad_node_ids
