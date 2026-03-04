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
