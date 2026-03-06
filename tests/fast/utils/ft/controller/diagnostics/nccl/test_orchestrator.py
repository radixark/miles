"""Tests for InterMachineOrchestrator and _cross_compare."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.diagnostics.nccl.orchestrator import (
    InterMachineOrchestrator,
    _PairResult,
    _cross_compare,
)
from miles.utils.ft.models.diagnostics import DiagnosticResult
from tests.fast.utils.ft.conftest import FakeNodeAgent, HangingNodeAgent


def _make_inter_machine_agent(node_id: str, *, passed: bool = True) -> FakeNodeAgent:
    return FakeNodeAgent(
        diagnostic_results={
            "inter_machine": DiagnosticResult(
                diagnostic_type="inter_machine",
                node_id=node_id,
                passed=passed,
                details="pass" if passed else "fail",
            ),
        },
        node_id=node_id,
    )


# ===================================================================
# _cross_compare
# ===================================================================


class TestCrossCompare:
    def test_all_pass_returns_empty(self) -> None:
        results = [
            _PairResult(master_id="A", worker_id="B", passed=True),
            _PairResult(master_id="B", worker_id="A", passed=True),
        ]

        assert _cross_compare(node_ids=["A", "B"], pair_results=results) == []

    def test_isolates_bad_node(self) -> None:
        """B fails in both pairs it participates in, A and C only fail when paired with B."""
        results = [
            _PairResult(master_id="A", worker_id="B", passed=False),
            _PairResult(master_id="B", worker_id="C", passed=False),
            _PairResult(master_id="C", worker_id="A", passed=True),
        ]

        bad = _cross_compare(node_ids=["A", "B", "C"], pair_results=results)

        assert bad == ["B"]

    def test_all_fail_returns_empty(self) -> None:
        """All nodes fail equally — cannot localize."""
        results = [
            _PairResult(master_id="A", worker_id="B", passed=False),
            _PairResult(master_id="B", worker_id="A", passed=False),
        ]

        assert _cross_compare(node_ids=["A", "B"], pair_results=results) == []

    def test_single_node_no_pairs(self) -> None:
        """Single node with no pair results."""
        assert _cross_compare(node_ids=["A"], pair_results=[]) == []

    def test_multiple_bad_nodes_with_highest_count(self) -> None:
        """Two nodes both have highest failure count and get flagged."""
        results = [
            _PairResult(master_id="A", worker_id="B", passed=False),
            _PairResult(master_id="B", worker_id="C", passed=False),
            _PairResult(master_id="C", worker_id="D", passed=True),
            _PairResult(master_id="D", worker_id="A", passed=False),
        ]

        bad = _cross_compare(node_ids=["A", "B", "C", "D"], pair_results=results)

        assert bad == ["A", "B"]

    def test_empty_pair_results(self) -> None:
        result = _cross_compare(
            node_ids=["A", "B"],
            pair_results=[],
        )
        assert result == []


# ===================================================================
# _resolve_address
# ===================================================================


class TestResolveAddress:
    def test_with_custom_addresses(self) -> None:
        orch = InterMachineOrchestrator(
            node_agents={},
            node_addresses={"n1": "10.0.0.1", "n2": "10.0.0.2"},
        )

        assert orch._resolve_address("n1") == "10.0.0.1"

    def test_fallback_to_node_id(self) -> None:
        orch = InterMachineOrchestrator(node_agents={}, node_addresses=None)

        assert orch._resolve_address("n1") == "n1"

    def test_missing_node_in_addresses_falls_back(self) -> None:
        orch = InterMachineOrchestrator(
            node_agents={},
            node_addresses={"n1": "10.0.0.1"},
        )

        assert orch._resolve_address("n99") == "n99"


# ===================================================================
# _run_single_pair — missing agent
# ===================================================================


class TestRunSinglePairMissingAgent:
    @pytest.mark.anyio
    async def test_missing_master_agent_returns_failed(self) -> None:
        agents = {"worker": _make_inter_machine_agent("worker")}
        orch = InterMachineOrchestrator(node_agents=agents)

        result = await orch._run_single_pair(
            master_id="master", worker_id="worker",
            master_addr="10.0.0.1", port=29500, timeout_seconds=30,
        )

        assert result.passed is False
        assert result.master_id == "master"

    @pytest.mark.anyio
    async def test_missing_worker_agent_returns_failed(self) -> None:
        agents = {"master": _make_inter_machine_agent("master")}
        orch = InterMachineOrchestrator(node_agents=agents)

        result = await orch._run_single_pair(
            master_id="master", worker_id="worker",
            master_addr="10.0.0.1", port=29500, timeout_seconds=30,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_agents_missing_returns_failed(self) -> None:
        orch = InterMachineOrchestrator(node_agents={})

        result = await orch._run_single_pair(
            master_id="A", worker_id="B",
            master_addr="10.0.0.1", port=29500, timeout_seconds=30,
        )

        assert result.passed is False


# ===================================================================
# run — fewer than 2 nodes
# ===================================================================


class TestRunEdgeCases:
    @pytest.mark.anyio
    async def test_single_node_returns_empty(self) -> None:
        agents = {"A": _make_inter_machine_agent("A")}
        orch = InterMachineOrchestrator(node_agents=agents)

        bad = await orch.run(node_ids=["A"], timeout_seconds=30)

        assert bad == []

    @pytest.mark.anyio
    async def test_empty_nodes_returns_empty(self) -> None:
        orch = InterMachineOrchestrator(node_agents={})

        bad = await orch.run(node_ids=[], timeout_seconds=30)

        assert bad == []

    @pytest.mark.anyio
    async def test_two_nodes_all_pass(self) -> None:
        agents = {
            "A": _make_inter_machine_agent("A"),
            "B": _make_inter_machine_agent("B"),
        }
        orch = InterMachineOrchestrator(node_agents=agents)

        bad = await orch.run(node_ids=["A", "B"], timeout_seconds=30)

        assert bad == []


# ===================================================================
# _run_single_pair — agent RPC hang
# ===================================================================


class TestRunSinglePairAgentHang:
    @pytest.mark.anyio
    async def test_hanging_master_agent_times_out(self) -> None:
        """When master agent RPC hangs, the pair should fail after timeout."""
        agents: dict = {
            "master": HangingNodeAgent(node_id="master"),
            "worker": _make_inter_machine_agent("worker"),
        }
        orch = InterMachineOrchestrator(node_agents=agents)

        result = await orch._run_single_pair(
            master_id="master", worker_id="worker",
            master_addr="10.0.0.1", port=29500, timeout_seconds=0,
        )

        assert result.passed is False
        assert result.master_id == "master"
        assert result.worker_id == "worker"

    @pytest.mark.anyio
    async def test_hanging_worker_agent_times_out(self) -> None:
        agents: dict = {
            "master": _make_inter_machine_agent("master"),
            "worker": HangingNodeAgent(node_id="worker"),
        }
        orch = InterMachineOrchestrator(node_agents=agents)

        result = await orch._run_single_pair(
            master_id="master", worker_id="worker",
            master_addr="10.0.0.1", port=29500, timeout_seconds=0,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_hanging_agents_time_out(self) -> None:
        agents: dict = {
            "A": HangingNodeAgent(node_id="A"),
            "B": HangingNodeAgent(node_id="B"),
        }
        orch = InterMachineOrchestrator(node_agents=agents)

        result = await orch._run_single_pair(
            master_id="A", worker_id="B",
            master_addr="10.0.0.1", port=29500, timeout_seconds=0,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_run_with_hanging_agent_isolates_bad_node(self) -> None:
        """Full run() with one hanging node should localize it as the bad node."""
        agents: dict = {
            "A": _make_inter_machine_agent("A"),
            "B": HangingNodeAgent(node_id="B"),
            "C": _make_inter_machine_agent("C"),
        }
        orch = InterMachineOrchestrator(node_agents=agents)

        bad = await orch.run(node_ids=["A", "B", "C"], timeout_seconds=0)

        assert "B" in bad
