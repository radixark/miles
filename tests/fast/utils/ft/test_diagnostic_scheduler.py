"""Tests for DiagnosticScheduler and BaseDiagnostic."""
from __future__ import annotations

import asyncio
import logging

import pytest

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.models import ActionType, DiagnosticResult
from tests.fast.utils.ft.conftest import (
    FailingDiagnostic,
    FakeNodeAgent,
    StubDiagnostic,
    make_fake_agents,
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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_empty_pipeline_returns_notify_human(self) -> None:
        agents = make_fake_agents({"node-0": {}})
        scheduler = DiagnosticScheduler(agents=agents, pipeline=[])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.asyncio
    async def test_no_pipeline_arg_returns_notify_human(self) -> None:
        scheduler = DiagnosticScheduler(agents={})
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="hang")

        assert decision.action == ActionType.NOTIFY_HUMAN


class TestDiagnosticSchedulerSingleStep:
    @pytest.mark.asyncio
    async def test_all_pass_returns_notify_human(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
        })
        scheduler = DiagnosticScheduler(agents=agents, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_trigger_reason_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        agents: dict[str, FakeNodeAgent] = {}
        scheduler = DiagnosticScheduler(agents=agents, pipeline=[])

        with caplog.at_level(logging.INFO):
            await scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
                suspect_node_ids=["node-0"],
            )

        assert "hang" in caplog.text

    @pytest.mark.asyncio
    async def test_no_agents_returns_notify(self) -> None:
        scheduler = DiagnosticScheduler(agents={}, pipeline=["gpu"])
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_set_diagnostic_overrides_existing(self) -> None:
        from miles.utils.ft.agents.node_agent import FtNodeAgent

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

    @pytest.mark.asyncio
    async def test_remove_diagnostic(self) -> None:
        from miles.utils.ft.agents.node_agent import FtNodeAgent

        stub = StubDiagnostic(passed=True)
        agent = FtNodeAgent(node_id="node-0", diagnostics=[stub])

        try:
            result1 = await agent.run_diagnostic("stub")
            assert result1.passed is True

            agent.remove_diagnostic("stub")

            result2 = await agent.run_diagnostic("stub")
            assert result2.passed is False
            assert "unknown diagnostic type" in result2.details
        finally:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_is_noop(self) -> None:
        from miles.utils.ft.agents.node_agent import FtNodeAgent

        agent = FtNodeAgent(node_id="node-0")
        try:
            agent.remove_diagnostic("nonexistent")
        finally:
            await agent.stop()


class TestDiagnosticSchedulerInterMachine:
    """Tests for the inter-machine diagnostic step with cross-comparison."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_inter_machine_one_bad_node(self) -> None:
        # 3 nodes: A, B, C. Pairs: (A,B), (B,C), (C,A)
        # A is bad → pairs (A,B) and (C,A) fail, (B,C) passes
        # A has failure_count=2, B has 1, C has 1 → A is bad
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

    @pytest.mark.asyncio
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

        # All nodes involved in failures → can't localize → NOTIFY_HUMAN
        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.asyncio
    async def test_inter_machine_two_nodes_pair_fails(self) -> None:
        # 2 nodes: pair (A,B) fails → both marked bad
        agents = make_fake_agents({
            "node-0": {"inter_machine": False},
            "node-1": {"inter_machine": False},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        # 2 nodes, both fail → all nodes in failures → can't localize
        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.asyncio
    async def test_inter_machine_single_node_skipped(self) -> None:
        agents = make_fake_agents({
            "node-0": {"inter_machine": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        # Single node → skip inter-machine → all pass
        assert decision.action == ActionType.NOTIFY_HUMAN

    @pytest.mark.asyncio
    async def test_inter_machine_pairing_is_round_robin(self) -> None:
        from miles.utils.ft.controller.diagnostics.scheduler import (
            DiagnosticScheduler as DS,
        )

        # Verify pair structure with 4 nodes
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

    @pytest.mark.asyncio
    async def test_inter_machine_port_assignment(self) -> None:
        from miles.utils.ft.controller.diagnostics.scheduler import (
            _INTER_MACHINE_BASE_PORT,
        )

        # 3 nodes → 3 pairs → ports 29500, 29501, 29502
        assert _INTER_MACHINE_BASE_PORT == 29500

    @pytest.mark.asyncio
    async def test_inter_machine_agent_timeout(self) -> None:
        class _TimeoutAgent:
            def set_diagnostic(self, diagnostic: object) -> None:
                pass

            def remove_diagnostic(self, diagnostic_type: str) -> None:
                pass

            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120,
            ) -> DiagnosticResult:
                raise asyncio.TimeoutError("timed out")

        good_agents = make_fake_agents({"node-0": {"inter_machine": True}})
        agents: dict[str, object] = {
            **good_agents,
            "node-1": _TimeoutAgent(),
            "node-2": make_fake_agents({"node-2": {"inter_machine": True}})["node-2"],
        }
        scheduler = DiagnosticScheduler(
            agents=agents, pipeline=["inter_machine"],
        )
        decision = await scheduler.run_diagnostic_pipeline(trigger_reason="crash")

        # node-1 times out in pairs (node-0, node-1) and (node-1, node-2)
        # node-1 has failure_count=2, others have 1 each → node-1 is bad
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert "node-1" in decision.bad_node_ids


class TestDiagnosticSchedulerLiveAgents:
    """Test scheduler with real FtNodeAgent instances (not FakeNodeAgent)."""

    @pytest.mark.asyncio
    async def test_scheduler_with_real_node_agents(self) -> None:
        from miles.utils.ft.agents.node_agent import FtNodeAgent

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

    @pytest.mark.asyncio
    async def test_scheduler_with_mismatched_diagnostic_type(self) -> None:
        from miles.utils.ft.agents.node_agent import FtNodeAgent

        good = StubDiagnostic(passed=True)
        bad = FailingDiagnostic(details="gpu broken")
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
            # node-1 has FailingDiagnostic (type="failing"), not "stub",
            # so it gets "unknown diagnostic type" → failure
            assert decision.action == ActionType.MARK_BAD_AND_RESTART
            assert "node-1" in decision.bad_node_ids
        finally:
            await agent0.stop()
            await agent1.stop()
