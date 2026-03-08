from __future__ import annotations

import asyncio

import pytest
from tests.fast.utils.ft.utils import (
    FakeNodeAgent,
    HangingNodeAgent,
    StubDiagnostic,
    make_fake_agents,
)

from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.controller.diagnostics.executors import GpuClusterExecutor, InterMachineClusterExecutor, SingleNodeClusterExecutor
from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator
from miles.utils.ft.models.diagnostics import DiagnosticResult

# ---------------------------------------------------------------------------
# BaseNodeExecutor tests
# ---------------------------------------------------------------------------


class _ConcreteDiagnostic(BaseNodeExecutor):
    diagnostic_type = "test_concrete"

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=True,
            details="ok",
        )


class TestBaseNodeExecutor:
    @pytest.mark.anyio
    async def test_concrete_subclass_can_run(self) -> None:
        diag = _ConcreteDiagnostic()
        result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert result.node_id == "node-0"
        assert result.diagnostic_type == "test_concrete"

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseNodeExecutor()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DiagnosticOrchestrator tests
# ---------------------------------------------------------------------------


class TestDiagnosticOrchestratorEmptyPipeline:
    @pytest.mark.anyio
    async def test_empty_pipeline_returns_notify_human(self) -> None:
        agents = make_fake_agents({"node-0": {}})
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []

    @pytest.mark.anyio
    async def test_no_pipeline_arg_returns_notify_human(self) -> None:
        orchestrator = DiagnosticOrchestrator(agents={})
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []


class TestDiagnosticOrchestratorSingleStep:
    @pytest.mark.anyio
    async def test_all_pass_returns_notify_human(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []

    @pytest.mark.anyio
    async def test_one_node_fails_returns_mark_bad(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-1" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_all_nodes_fail_returns_mark_bad(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": False},
                "node-1": {"gpu": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert sorted(decision.bad_node_ids) == ["node-0", "node-1"]


class TestDiagnosticOrchestratorMultiStep:
    @pytest.mark.anyio
    async def test_first_step_catches_bad_node(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": False, "intra": True},
                "node-1": {"gpu": True, "intra": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor(), SingleNodeClusterExecutor("intra")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == ["node-0"]

    @pytest.mark.anyio
    async def test_first_passes_second_catches(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True, "intra": True},
                "node-1": {"gpu": True, "intra": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor(), SingleNodeClusterExecutor("intra")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == ["node-1"]

    @pytest.mark.anyio
    async def test_all_steps_pass_returns_notify(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True, "intra": True},
                "node-1": {"gpu": True, "intra": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor(), SingleNodeClusterExecutor("intra")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []


class TestDiagnosticOrchestratorErrorHandling:
    @pytest.mark.anyio
    async def test_agent_exception_treated_as_failure(self) -> None:
        class _RaisingAgent:
            async def run_diagnostic(
                self,
                diagnostic_type: str,
                timeout_seconds: int = 120,
            ) -> DiagnosticResult:
                raise RuntimeError("agent crashed")

        good_agents = make_fake_agents({"node-0": {"gpu": True}})
        agents = {
            **good_agents,
            "node-1": _RaisingAgent(),
        }
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-1" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_no_agents_returns_notify(self) -> None:
        orchestrator = DiagnosticOrchestrator(agents={}, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []


class TestDiagnosticOrchestratorInterMachine:
    """Tests for the inter-machine diagnostic step with cross-comparison."""

    @pytest.mark.anyio
    async def test_inter_machine_all_pass(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"inter_machine": True},
                "node-1": {"inter_machine": True},
                "node-2": {"inter_machine": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[InterMachineClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []

    @pytest.mark.anyio
    async def test_inter_machine_one_bad_node(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"inter_machine": False},
                "node-1": {"inter_machine": True},
                "node-2": {"inter_machine": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[InterMachineClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-0" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_inter_machine_cannot_localize_all_fail_is_inconclusive(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"inter_machine": False},
                "node-1": {"inter_machine": False},
                "node-2": {"inter_machine": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[InterMachineClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []
        assert decision.conclusive is False

    @pytest.mark.anyio
    async def test_inter_machine_two_nodes_pair_fails_is_inconclusive(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"inter_machine": False},
                "node-1": {"inter_machine": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[InterMachineClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []
        assert decision.conclusive is False

    @pytest.mark.anyio
    async def test_inter_machine_single_node_skipped(self) -> None:
        agents = make_fake_agents({"node-0": {"inter_machine": True}})
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[InterMachineClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []

    def test_inter_machine_pairing_is_round_robin(self) -> None:
        node_ids = ["node-0", "node-1", "node-2", "node-3"]
        expected_pairs = [
            ("node-0", "node-1"),
            ("node-1", "node-2"),
            ("node-2", "node-3"),
            ("node-3", "node-0"),
        ]
        pairs = [(node_ids[i], node_ids[(i + 1) % len(node_ids)]) for i in range(len(node_ids))]
        assert pairs == expected_pairs

    def test_inter_machine_port_assignment(self) -> None:
        from miles.utils.ft.agents.diagnostics.executors.inter_machine import DEFAULT_NCCL_MASTER_PORT

        assert DEFAULT_NCCL_MASTER_PORT == 29500

    @pytest.mark.anyio
    async def test_inter_machine_agent_exception(self) -> None:
        class _RaisingInterMachineAgent:
            async def run_diagnostic(
                self,
                diagnostic_type: str,
                timeout_seconds: int = 120,
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

        good_agents = make_fake_agents(
            {
                "node-0": {"inter_machine": True},
                "node-2": {"inter_machine": True},
            }
        )
        agents = {**good_agents, "node-1": _RaisingInterMachineAgent()}  # type: ignore[dict-item]
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[InterMachineClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-1" in decision.bad_node_ids


class TestDiagnosticOrchestratorLiveAgents:
    """Test orchestrator with real FtNodeAgent instances (not FakeNodeAgent)."""

    @pytest.mark.anyio
    async def test_orchestrator_with_real_node_agents(self) -> None:
        stub = StubDiagnostic(passed=True, details="all good")
        agent0 = FtNodeAgent(node_id="node-0", diagnostics=[stub])
        agent1 = FtNodeAgent(node_id="node-1", diagnostics=[stub])

        agents = {"node-0": agent0, "node-1": agent1}
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[SingleNodeClusterExecutor("stub")])

        try:
            decision = await orchestrator.run_diagnostic_pipeline()
            assert decision.bad_node_ids == []
        finally:
            await agent0.stop()
            await agent1.stop()

    @pytest.mark.anyio
    async def test_orchestrator_with_mismatched_diagnostic_type(self) -> None:
        good = StubDiagnostic(passed=True)
        bad = StubDiagnostic(passed=False, details="gpu broken", diagnostic_type="failing")
        agent0 = FtNodeAgent(node_id="node-0", diagnostics=[good])
        agent1 = FtNodeAgent(node_id="node-1", diagnostics=[bad])

        agents = {"node-0": agent0, "node-1": agent1}
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[SingleNodeClusterExecutor("stub")],
        )

        try:
            decision = await orchestrator.run_diagnostic_pipeline()
            # node-1 has diagnostic_type="failing", not "stub",
            # so it gets "unknown diagnostic type" → treated as fail
            assert "node-1" in decision.bad_node_ids
        finally:
            await agent0.stop()
            await agent1.stop()


# ---------------------------------------------------------------------------
# Pre-executor tests (verifying the pre_executors parameter)
# ---------------------------------------------------------------------------


class TestPreExecutors:
    @pytest.mark.anyio
    async def test_pre_executor_runs_before_pipeline(self) -> None:
        """A pre_executor that returns bad_node_ids short-circuits the pipeline."""

        class _EvictAllExecutor:
            async def execute(
                self,
                agents: dict[str, FakeNodeAgent],
                timeout_seconds: int,
            ) -> list[str]:
                return list(agents.keys())

        agents = make_fake_agents({"node-0": {"gpu": True}, "node-1": {"gpu": True}})
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[_EvictAllExecutor()],
        )

        assert sorted(decision.bad_node_ids) == ["node-0", "node-1"]

    @pytest.mark.anyio
    async def test_pre_executor_passes_through_to_pipeline(self) -> None:
        """A pre_executor that returns no bad nodes allows the pipeline to continue."""

        class _PassThroughExecutor:
            async def execute(
                self,
                agents: dict[str, FakeNodeAgent],
                timeout_seconds: int,
            ) -> list[str]:
                return []

        agents = make_fake_agents({"node-0": {"gpu": False}})
        orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[_PassThroughExecutor()],
        )

        assert decision.bad_node_ids == ["node-0"]


# ---------------------------------------------------------------------------
# Agent RPC hang tests
# ---------------------------------------------------------------------------


class TestAgentRpcHang:
    """Verify _call_agent_diagnostic times out when agent RPC never returns."""

    @pytest.mark.anyio
    async def test_hanging_agent_returns_fail_result(self) -> None:
        good_agents = make_fake_agents({"node-0": {"gpu": True}})
        agents = {**good_agents, "node-hang": HangingNodeAgent(node_id="node-hang")}
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-hang" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_hanging_agent_does_not_block_healthy_agents(self) -> None:
        """All agents run in parallel; a hanging agent must not prevent others from completing."""
        good_agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
            }
        )
        agents = {**good_agents, "node-hang": HangingNodeAgent(node_id="node-hang")}
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-hang" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids
        assert "node-1" not in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_all_agents_hanging_returns_mark_bad(self) -> None:
        agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
            "node-1": HangingNodeAgent(node_id="node-1"),
        }
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert sorted(decision.bad_node_ids) == ["node-0", "node-1"]


# ---------------------------------------------------------------------------
# Pipeline-level timeout tests
# ---------------------------------------------------------------------------


class TestPipelineTimeout:
    @pytest.mark.anyio
    async def test_pipeline_timeout_returns_inconclusive(self) -> None:
        """When the entire pipeline exceeds pipeline_timeout_seconds, return inconclusive."""
        agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
        }
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=9999,
            pipeline_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []
        assert "timed out" in decision.reason
        assert decision.conclusive is False

    @pytest.mark.anyio
    async def test_pipeline_timeout_fires_before_per_call_timeout(self) -> None:
        """Pipeline timeout should fire even when per-call timeout is very large."""
        agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
            "node-1": HangingNodeAgent(node_id="node-1"),
        }
        orchestrator = DiagnosticOrchestrator(
            agents=agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=99999,
            pipeline_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []
        assert "timed out" in decision.reason
        assert decision.conclusive is False
