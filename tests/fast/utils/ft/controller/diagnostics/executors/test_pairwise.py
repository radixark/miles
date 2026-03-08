"""Tests for PairwiseClusterExecutor and _cross_compare."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import FakeNodeAgent, HangingNodeAgent

from miles.utils.ft.controller.diagnostics.executors.pairwise import (
    PairwiseClusterExecutor,
    _cross_compare,
    _PairResult,
)
from miles.utils.ft.models.diagnostics import DiagnosticResult

_DIAG_TYPE = "nccl_pairwise"


def _make_pairwise_agent(node_id: str, *, passed: bool = True) -> FakeNodeAgent:
    return FakeNodeAgent(
        diagnostic_results={
            _DIAG_TYPE: DiagnosticResult(
                diagnostic_type=_DIAG_TYPE,
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
        """All nodes fail equally — cannot localize, returns empty."""
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
# _run_single_pair — missing agent
# ===================================================================


class TestRunSinglePairMissingAgent:
    @pytest.mark.anyio
    async def test_missing_master_agent_returns_failed(self) -> None:
        agents = {"worker": _make_pairwise_agent("worker")}
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_single_pair(
            agents=agents,
            master_id="master",
            worker_id="worker",
            master_addr="10.0.0.1",
            port=29500,
            timeout_seconds=30,
        )

        assert result.passed is False
        assert result.master_id == "master"

    @pytest.mark.anyio
    async def test_missing_worker_agent_returns_failed(self) -> None:
        agents = {"master": _make_pairwise_agent("master")}
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_single_pair(
            agents=agents,
            master_id="master",
            worker_id="worker",
            master_addr="10.0.0.1",
            port=29500,
            timeout_seconds=30,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_agents_missing_returns_failed(self) -> None:
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_single_pair(
            agents={},
            master_id="A",
            worker_id="B",
            master_addr="10.0.0.1",
            port=29500,
            timeout_seconds=30,
        )

        assert result.passed is False


# ===================================================================
# execute — fewer than 2 nodes
# ===================================================================


class TestExecuteEdgeCases:
    @pytest.mark.anyio
    async def test_single_node_returns_empty(self) -> None:
        agents = {"A": _make_pairwise_agent("A")}
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        bad = await executor.execute(agents=agents, timeout_seconds=30)

        assert bad == []

    @pytest.mark.anyio
    async def test_empty_nodes_returns_empty(self) -> None:
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        bad = await executor.execute(agents={}, timeout_seconds=30)

        assert bad == []

    @pytest.mark.anyio
    async def test_two_nodes_all_pass(self) -> None:
        agents = {
            "A": _make_pairwise_agent("A"),
            "B": _make_pairwise_agent("B"),
        }
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        bad = await executor.execute(agents=agents, timeout_seconds=30)

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
            "worker": _make_pairwise_agent("worker"),
        }
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_single_pair(
            agents=agents,
            master_id="master",
            worker_id="worker",
            master_addr="10.0.0.1",
            port=29500,
            timeout_seconds=0,
        )

        assert result.passed is False
        assert result.master_id == "master"
        assert result.worker_id == "worker"

    @pytest.mark.anyio
    async def test_hanging_worker_agent_times_out(self) -> None:
        agents: dict = {
            "master": _make_pairwise_agent("master"),
            "worker": HangingNodeAgent(node_id="worker"),
        }
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_single_pair(
            agents=agents,
            master_id="master",
            worker_id="worker",
            master_addr="10.0.0.1",
            port=29500,
            timeout_seconds=0,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_hanging_agents_time_out(self) -> None:
        agents: dict = {
            "A": HangingNodeAgent(node_id="A"),
            "B": HangingNodeAgent(node_id="B"),
        }
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_single_pair(
            agents=agents,
            master_id="A",
            worker_id="B",
            master_addr="10.0.0.1",
            port=29500,
            timeout_seconds=0,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_execute_with_hanging_agent_isolates_bad_node(self) -> None:
        """Full execute() with one hanging node should localize it as the bad node."""
        agents: dict = {
            "A": _make_pairwise_agent("A"),
            "B": HangingNodeAgent(node_id="B"),
            "C": _make_pairwise_agent("C"),
        }
        executor = PairwiseClusterExecutor(diagnostic_type=_DIAG_TYPE)

        bad = await executor.execute(agents=agents, timeout_seconds=0)

        assert "B" in bad
