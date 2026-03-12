from __future__ import annotations

import asyncio

import pytest
from tests.fast.utils.ft.utils import FakeNodeAgent, HangingNodeAgent, StubDiagnostic, make_fake_agents

from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.diagnostics.executors import (
    GpuClusterExecutor,
    PairwiseClusterExecutor,
    PerNodeClusterExecutor,
)
from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

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


class TestDiagnosticOrchestratorValidation:
    def test_negative_pipeline_timeout_raises(self) -> None:
        """Previously negative pipeline_timeout_seconds caused
        asyncio.wait_for to raise ValueError at runtime. Now
        rejected at construction time."""
        with pytest.raises(ValueError, match="pipeline_timeout_seconds must be >= 0"):
            DiagnosticOrchestrator(pipeline=[], pipeline_timeout_seconds=-1)

    def test_zero_pipeline_timeout_allowed(self) -> None:
        orchestrator = DiagnosticOrchestrator(pipeline=[], pipeline_timeout_seconds=0)
        assert orchestrator._pipeline_timeout_seconds == 0


class TestDiagnosticOrchestratorEmptyPipeline:
    @pytest.mark.anyio
    async def test_empty_pipeline_returns_notify_human(self) -> None:
        node_agents = make_fake_agents({"node-0": {}})
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []

    @pytest.mark.anyio
    async def test_no_pipeline_arg_returns_notify_human(self) -> None:
        orchestrator = DiagnosticOrchestrator(node_agents={}, pipeline=[])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []


class TestDiagnosticOrchestratorSingleStep:
    @pytest.mark.anyio
    async def test_all_pass_returns_notify_human(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []

    @pytest.mark.anyio
    async def test_one_node_fails_returns_mark_bad(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-1" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_all_nodes_fail_returns_mark_bad(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"gpu": False},
                "node-1": {"gpu": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert sorted(decision.bad_node_ids) == ["node-0", "node-1"]


class TestDiagnosticOrchestratorMultiStep:
    @pytest.mark.anyio
    async def test_first_step_catches_bad_node(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"gpu": False, "intra": True},
                "node-1": {"gpu": True, "intra": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[GpuClusterExecutor(), PerNodeClusterExecutor("intra")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == ["node-0"]

    @pytest.mark.anyio
    async def test_first_passes_second_catches(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"gpu": True, "intra": True},
                "node-1": {"gpu": True, "intra": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[GpuClusterExecutor(), PerNodeClusterExecutor("intra")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == ["node-1"]

    @pytest.mark.anyio
    async def test_all_steps_pass_returns_notify(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"gpu": True, "intra": True},
                "node-1": {"gpu": True, "intra": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[GpuClusterExecutor(), PerNodeClusterExecutor("intra")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []


class TestDiagnosticOrchestratorExecutorCrash:
    @pytest.mark.anyio
    async def test_crashed_executor_reflected_in_reason(self) -> None:
        """H-6: when an executor raises, the reason must mention the failure
        instead of claiming 'all diagnostics passed'."""

        class _CrashingExecutor:
            async def execute(self, node_agents: dict, timeout_seconds: int) -> list[str]:
                raise RuntimeError("executor boom")

        node_agents = make_fake_agents({"node-0": {"gpu": True}})
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[_CrashingExecutor(), GpuClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []
        assert "failed" in decision.reason
        assert "_CrashingExecutor" in decision.reason
        assert "all diagnostics passed" not in decision.reason


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
        node_agents = {
            **good_agents,
            "node-1": _RaisingAgent(),
        }
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-1" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_no_agents_returns_notify(self) -> None:
        orchestrator = DiagnosticOrchestrator(node_agents={}, pipeline=[GpuClusterExecutor()])
        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []


class TestDiagnosticOrchestratorInterMachine:
    """Tests for the inter-machine diagnostic step with cross-comparison."""

    @pytest.mark.anyio
    async def test_inter_machine_all_pass(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"nccl_pairwise": True},
                "node-1": {"nccl_pairwise": True},
                "node-2": {"nccl_pairwise": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []

    @pytest.mark.anyio
    async def test_inter_machine_one_bad_node(self) -> None:
        node_agents = make_fake_agents(
            {
                "node-0": {"nccl_pairwise": False},
                "node-1": {"nccl_pairwise": True},
                "node-2": {"nccl_pairwise": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-0" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_inter_machine_cannot_localize_all_fail_odd_nodes(self) -> None:
        """With 3 nodes all-fail under two-round pairing, node-1 appears in
        both rounds and accumulates the highest failure count."""
        node_agents = make_fake_agents(
            {
                "node-0": {"nccl_pairwise": False},
                "node-1": {"nccl_pairwise": False},
                "node-2": {"nccl_pairwise": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == ["node-1"]

    @pytest.mark.anyio
    async def test_inter_machine_two_nodes_pair_fails_treated_as_executor_failure(self) -> None:
        """Two nodes both fail — pairwise cannot localize. Previously returned [],
        misinterpreted as 'no fault'. Now raises PairwiseInconclusiveError,
        caught by the orchestrator as executor failure."""
        node_agents = make_fake_agents(
            {
                "node-0": {"nccl_pairwise": False},
                "node-1": {"nccl_pairwise": False},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []
        assert "failed" in decision.reason
        assert "all diagnostics passed" not in decision.reason

    @pytest.mark.anyio
    async def test_inter_machine_all_fail_continues_to_next_executor(self) -> None:
        """When inter-machine inconclusive (PairwiseInconclusiveError), orchestrator
        catches it and continues to the next executor (gpu)."""
        node_agents = make_fake_agents(
            {
                "node-0": {"nccl_pairwise": False, "gpu": False},
                "node-1": {"nccl_pairwise": False, "gpu": True},
            }
        )
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise"), GpuClusterExecutor()],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert "node-0" in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_inter_machine_single_node_skipped(self) -> None:
        node_agents = make_fake_agents({"node-0": {"nccl_pairwise": True}})
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise")],
        )
        decision = await orchestrator.run_diagnostic_pipeline()
        assert decision.bad_node_ids == []

    def test_inter_machine_pairing_is_two_round(self) -> None:
        from miles.utils.ft.controller.diagnostics.executors.pairwise import _generate_round_pairs

        node_ids = ["node-0", "node-1", "node-2", "node-3"]
        r1, r2 = _generate_round_pairs(node_ids)
        assert r1 == [("node-0", "node-1"), ("node-2", "node-3")]
        assert r2 == [("node-1", "node-2"), ("node-3", "node-0")]

    # P2 item 19: degenerate pairing cases
    def test_pairing_exactly_two_nodes(self) -> None:
        from miles.utils.ft.controller.diagnostics.executors.pairwise import _generate_round_pairs

        r1, r2 = _generate_round_pairs(["node-0", "node-1"])
        assert r1 == [("node-0", "node-1")]
        assert r2 == [("node-1", "node-0")]

    def test_pairing_exactly_one_node(self) -> None:
        from miles.utils.ft.controller.diagnostics.executors.pairwise import _generate_round_pairs

        r1, r2 = _generate_round_pairs(["node-0"])
        assert r1 == []
        assert r2 == []

    def test_pairing_three_nodes_odd(self) -> None:
        from miles.utils.ft.controller.diagnostics.executors.pairwise import _generate_round_pairs

        r1, r2 = _generate_round_pairs(["node-0", "node-1", "node-2"])
        assert r1 == [("node-0", "node-1")]
        assert r2 == [("node-1", "node-2")]

    def test_inter_machine_port_assignment(self) -> None:
        from miles.utils.ft.agents.diagnostics.executors.nccl import DEFAULT_NCCL_MASTER_PORT

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
                if diagnostic_type == "nccl_pairwise":
                    raise asyncio.TimeoutError("timed out")
                return DiagnosticResult(
                    diagnostic_type=diagnostic_type,
                    node_id="node-1",
                    passed=True,
                    details="pass",
                )

        good_agents = make_fake_agents(
            {
                "node-0": {"nccl_pairwise": True},
                "node-2": {"nccl_pairwise": True},
            }
        )
        node_agents = {**good_agents, "node-1": _RaisingInterMachineAgent()}  # type: ignore[dict-item]
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PairwiseClusterExecutor(diagnostic_type="nccl_pairwise")],
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

        node_agents = {"node-0": agent0, "node-1": agent1}
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[PerNodeClusterExecutor("stub")])

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

        node_agents = {"node-0": agent0, "node-1": agent1}
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[PerNodeClusterExecutor("stub")],
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
                node_agents: dict[str, FakeNodeAgent],
                timeout_seconds: int,
            ) -> list[str]:
                return list(node_agents.keys())

        node_agents = make_fake_agents({"node-0": {"gpu": True}, "node-1": {"gpu": True}})
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[GpuClusterExecutor()])
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
                node_agents: dict[str, FakeNodeAgent],
                timeout_seconds: int,
            ) -> list[str]:
                return []

        node_agents = make_fake_agents({"node-0": {"gpu": False}})
        orchestrator = DiagnosticOrchestrator(node_agents=node_agents, pipeline=[GpuClusterExecutor()])
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
        node_agents = {**good_agents, "node-hang": HangingNodeAgent(node_id="node-hang")}
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
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
        node_agents = {**good_agents, "node-hang": HangingNodeAgent(node_id="node-hang")}
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert "node-hang" in decision.bad_node_ids
        assert "node-0" not in decision.bad_node_ids
        assert "node-1" not in decision.bad_node_ids

    @pytest.mark.anyio
    async def test_all_agents_hanging_returns_mark_bad(self) -> None:
        node_agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
            "node-1": HangingNodeAgent(node_id="node-1"),
        }
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
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
    async def test_pipeline_timeout_returns_empty(self) -> None:
        """When the entire pipeline exceeds pipeline_timeout_seconds, return empty result."""
        node_agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
        }
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=9999,
            pipeline_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []
        assert "timed out" in decision.reason

    # P2 item 23: pre-executor timeout
    @pytest.mark.anyio
    async def test_pipeline_timeout_during_pre_executors(self) -> None:
        """Pipeline timeout fires during pre_executors, before main pipeline."""
        node_agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
        }
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[],
            default_timeout_seconds=9999,
            pipeline_timeout_seconds=0,
        )
        hanging_pre_executor = GpuClusterExecutor()

        decision = await orchestrator.run_diagnostic_pipeline(
            pre_executors=[hanging_pre_executor],
        )

        assert decision.bad_node_ids == []
        assert "timed out" in decision.reason

    @pytest.mark.anyio
    async def test_pipeline_timeout_fires_before_per_call_timeout(self) -> None:
        """Pipeline timeout should fire even when per-call timeout is very large."""
        node_agents: dict = {
            "node-0": HangingNodeAgent(node_id="node-0"),
            "node-1": HangingNodeAgent(node_id="node-1"),
        }
        orchestrator = DiagnosticOrchestrator(
            node_agents=node_agents,
            pipeline=[GpuClusterExecutor()],
            default_timeout_seconds=99999,
            pipeline_timeout_seconds=0,
        )

        decision = await orchestrator.run_diagnostic_pipeline()

        assert decision.bad_node_ids == []
        assert "timed out" in decision.reason
