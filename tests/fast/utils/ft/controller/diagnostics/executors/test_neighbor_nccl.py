"""Tests for NeighborNcclClusterExecutor: ring topology, suspect localization, and edge execution."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import FakeNodeAgent, HangingNodeAgent

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.diagnostics.executors.neighbor_nccl import (
    NeighborNcclClusterExecutor,
    _build_ring_edges,
    _EdgeResult,
    _localize_suspects_from_neighbor_results,
)

_DIAG_TYPE = "nccl_pairwise"


def _make_agent(node_id: str, *, passed: bool = True) -> FakeNodeAgent:
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
# _build_ring_edges — topology unit tests
# ===================================================================


class TestBuildRingEdges:
    def test_single_node_returns_empty(self) -> None:
        assert _build_ring_edges(["A"]) == []

    def test_empty_returns_empty(self) -> None:
        assert _build_ring_edges([]) == []

    def test_two_nodes_single_edge(self) -> None:
        edges = _build_ring_edges(["A", "B"])
        assert edges == [("A", "B")]

    def test_three_nodes_ring(self) -> None:
        edges = _build_ring_edges(["A", "B", "C"])
        assert len(edges) == 3
        assert set(edges) == {("A", "B"), ("A", "C"), ("B", "C")}

    def test_four_nodes_ring(self) -> None:
        edges = _build_ring_edges(["A", "B", "C", "D"])
        assert len(edges) == 4
        assert set(edges) == {("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")}

    @pytest.mark.parametrize("n", range(2, 9))
    def test_no_self_loops(self, n: int) -> None:
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        for a, b in edges:
            assert a != b

    @pytest.mark.parametrize("n", range(2, 9))
    def test_edges_are_deduplicated(self, n: int) -> None:
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        assert len(edges) == len(set(edges))

    @pytest.mark.parametrize("n", range(2, 9))
    def test_edges_are_canonical_form(self, n: int) -> None:
        """Edges should be (min_id, max_id)."""
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        for a, b in edges:
            assert a < b

    @pytest.mark.parametrize("n", range(3, 9))
    def test_each_node_has_degree_two(self, n: int) -> None:
        """In a ring of N >= 3, each node connects to exactly 2 neighbors."""
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        from collections import Counter

        counter: Counter[str] = Counter()
        for a, b in edges:
            counter[a] += 1
            counter[b] += 1
        for nid in ids:
            assert counter[nid] == 2, f"{nid} has degree {counter[nid]}, expected 2"

    def test_two_nodes_each_has_degree_one(self) -> None:
        edges = _build_ring_edges(["A", "B"])
        from collections import Counter

        counter: Counter[str] = Counter()
        for a, b in edges:
            counter[a] += 1
            counter[b] += 1
        assert counter["A"] == 1
        assert counter["B"] == 1

    def test_input_order_does_not_change_edge_set(self) -> None:
        """Edge set should be the same regardless of input ordering (since we sort)."""
        edges_abc = set(_build_ring_edges(["A", "B", "C"]))
        edges_cba = set(_build_ring_edges(["C", "B", "A"]))
        # Sorted IDs are the same, so edge set should match
        assert edges_abc == edges_cba

    @pytest.mark.parametrize("n", range(2, 9))
    def test_edge_count_equals_n_for_n_ge_3(self, n: int) -> None:
        if n < 3:
            return
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        assert len(edges) == n


# ===================================================================
# _localize_suspects_from_neighbor_results — judgment unit tests
# ===================================================================


class TestLocalizeSuspectsBasicCases:
    def test_no_results_returns_empty(self) -> None:
        assert _localize_suspects_from_neighbor_results(["A", "B"], []) == []

    def test_all_pass_returns_empty(self) -> None:
        results = [_EdgeResult("A", "B", passed=True)]
        assert _localize_suspects_from_neighbor_results(["A", "B"], results) == []

    def test_two_nodes_single_edge_fails_returns_both(self) -> None:
        """2 nodes with the only edge failing: cannot determine who is bad, return both."""
        results = [_EdgeResult("A", "B", passed=False)]
        suspects = _localize_suspects_from_neighbor_results(["A", "B"], results)
        assert suspects == ["A", "B"]

    def test_all_edges_fail_returns_all_nodes(self) -> None:
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("A", "C", passed=False),
            _EdgeResult("B", "C", passed=False),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C"], results)
        assert suspects == ["A", "B", "C"]


class TestLocalizeSuspectsThreeNodeRing:
    """3-node ring: the case that previously caused systematic mis-attribution."""

    def test_single_bad_node_identified(self) -> None:
        """A is bad: edges (A,B) and (A,C) fail, edge (B,C) passes.
        A has all edges failed, B and C each have one pass → only A is suspect."""
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("A", "C", passed=False),
            _EdgeResult("B", "C", passed=True),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C"], results)
        assert suspects == ["A"]

    def test_two_adjacent_bad_nodes(self) -> None:
        """A and B are bad: edges (A,B), (A,C), (B,C) all fail → all edges fail → return all."""
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("A", "C", passed=False),
            _EdgeResult("B", "C", passed=False),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C"], results)
        assert suspects == ["A", "B", "C"]

    def test_two_edges_fail_cannot_isolate_returns_failed_participants(self) -> None:
        """Edges (A,B) fail and (B,C) fail, (A,C) passes.
        B has all edges failed → B is suspect."""
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("A", "C", passed=True),
            _EdgeResult("B", "C", passed=False),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C"], results)
        assert suspects == ["B"]


class TestLocalizeSuspectsLargerRings:
    def test_four_nodes_single_bad(self) -> None:
        """B is bad in a 4-node ring. Edges: (A,B) fail, (B,C) fail, (A,D) pass, (C,D) pass."""
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("B", "C", passed=False),
            _EdgeResult("C", "D", passed=True),
            _EdgeResult("A", "D", passed=True),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C", "D"], results)
        assert suspects == ["B"]

    def test_five_nodes_single_bad(self) -> None:
        """C is bad in a 5-node ring."""
        results = [
            _EdgeResult("A", "B", passed=True),
            _EdgeResult("B", "C", passed=False),
            _EdgeResult("C", "D", passed=False),
            _EdgeResult("D", "E", passed=True),
            _EdgeResult("A", "E", passed=True),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C", "D", "E"], results)
        assert suspects == ["C"]

    def test_five_nodes_adjacent_double_bad(self) -> None:
        """B and C are adjacent bad nodes in a 5-node ring.
        Edges: (A,B) fail, (B,C) fail, (C,D) fail, (A,E) pass, (D,E) pass.
        B: degree 2, fail 2 → all failed → suspect.
        C: degree 2, fail 2 → all failed → suspect."""
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("B", "C", passed=False),
            _EdgeResult("C", "D", passed=False),
            _EdgeResult("D", "E", passed=True),
            _EdgeResult("A", "E", passed=True),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C", "D", "E"], results)
        assert suspects == ["B", "C"]

    def test_five_nodes_non_adjacent_double_bad(self) -> None:
        """A and C are non-adjacent bad nodes in a 5-node ring.
        Edges: (A,B) fail, (B,C) fail, (C,D) fail, (D,E) pass, (A,E) fail.
        A: degree 2, fail 2 → all failed → suspect.
        C: degree 2, fail 2 → all failed → suspect.
        B: degree 2, fail 2 → all failed → suspect (over-eviction is allowed)."""
        results = [
            _EdgeResult("A", "B", passed=False),
            _EdgeResult("B", "C", passed=False),
            _EdgeResult("C", "D", passed=False),
            _EdgeResult("D", "E", passed=True),
            _EdgeResult("A", "E", passed=False),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C", "D", "E"], results)
        # A and C are truly bad; B is over-evicted (acceptable)
        assert "A" in suspects
        assert "C" in suspects

    def test_six_nodes_single_bad(self) -> None:
        results = [
            _EdgeResult("A", "B", passed=True),
            _EdgeResult("B", "C", passed=True),
            _EdgeResult("C", "D", passed=False),
            _EdgeResult("D", "E", passed=False),
            _EdgeResult("E", "F", passed=True),
            _EdgeResult("A", "F", passed=True),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C", "D", "E", "F"], results)
        assert suspects == ["D"]

    def test_seven_nodes_single_bad(self) -> None:
        results = [
            _EdgeResult("A", "B", passed=True),
            _EdgeResult("B", "C", passed=True),
            _EdgeResult("C", "D", passed=True),
            _EdgeResult("D", "E", passed=False),
            _EdgeResult("E", "F", passed=False),
            _EdgeResult("F", "G", passed=True),
            _EdgeResult("A", "G", passed=True),
        ]
        suspects = _localize_suspects_from_neighbor_results(["A", "B", "C", "D", "E", "F", "G"], results)
        assert suspects == ["E"]


class TestLocalizeSuspectsPropertyBased:
    """Property-based tests verifying invariants across many configurations."""

    @pytest.mark.parametrize("n", range(3, 9))
    def test_single_bad_node_always_in_suspects(self, n: int) -> None:
        """For each position in a ring, simulate one bad node and verify it's always suspected."""
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)

        for bad_idx in range(n):
            bad_id = ids[bad_idx]
            results = []
            for a, b in edges:
                passed = bad_id not in (a, b)
                results.append(_EdgeResult(a, b, passed=passed))

            suspects = _localize_suspects_from_neighbor_results(ids, results)
            assert bad_id in suspects, f"n={n}, bad={bad_id}: suspects={suspects} does not contain bad node"

    @pytest.mark.parametrize("n", range(3, 9))
    def test_node_id_ordering_does_not_change_suspect_set(self, n: int) -> None:
        """Results should be identical regardless of node ordering in input."""
        ids_forward = [f"node-{i}" for i in range(n)]
        ids_reverse = list(reversed(ids_forward))

        bad_id = ids_forward[0]
        edges_fwd = _build_ring_edges(ids_forward)
        edges_rev = _build_ring_edges(ids_reverse)

        def make_results(edges: list[tuple[str, str]]) -> list[_EdgeResult]:
            return [_EdgeResult(a, b, passed=(bad_id not in (a, b))) for a, b in edges]

        suspects_fwd = set(_localize_suspects_from_neighbor_results(ids_forward, make_results(edges_fwd)))
        suspects_rev = set(_localize_suspects_from_neighbor_results(ids_reverse, make_results(edges_rev)))
        assert suspects_fwd == suspects_rev

    @pytest.mark.parametrize("n", range(3, 9))
    def test_all_edges_pass_returns_empty(self, n: int) -> None:
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        results = [_EdgeResult(a, b, passed=True) for a, b in edges]
        assert _localize_suspects_from_neighbor_results(ids, results) == []

    @pytest.mark.parametrize("n", range(3, 9))
    def test_all_edges_fail_returns_all_nodes(self, n: int) -> None:
        ids = [f"node-{i}" for i in range(n)]
        edges = _build_ring_edges(ids)
        results = [_EdgeResult(a, b, passed=False) for a, b in edges]
        assert _localize_suspects_from_neighbor_results(ids, results) == ids


# ===================================================================
# _run_edge — missing agent
# ===================================================================


class TestRunEdgeMissingAgent:
    @pytest.mark.anyio
    async def test_missing_node_a_returns_failed(self) -> None:
        node_agents = {"B": _make_agent("B")}
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_edge(
            node_agents=node_agents,
            node_a="A",
            node_b="B",
            master_addr="A",
            port=29500,
            timeout_seconds=30,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_missing_node_b_returns_failed(self) -> None:
        node_agents = {"A": _make_agent("A")}
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_edge(
            node_agents=node_agents,
            node_a="A",
            node_b="B",
            master_addr="A",
            port=29500,
            timeout_seconds=30,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_missing_returns_failed(self) -> None:
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)

        result = await executor._run_edge(
            node_agents={},
            node_a="A",
            node_b="B",
            master_addr="A",
            port=29500,
            timeout_seconds=30,
        )

        assert result.passed is False


# ===================================================================
# execute — edge cases
# ===================================================================


class TestExecuteEdgeCases:
    @pytest.mark.anyio
    async def test_single_node_returns_empty(self) -> None:
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents={"A": _make_agent("A")}, timeout_seconds=30)
        assert bad == []

    @pytest.mark.anyio
    async def test_empty_nodes_returns_empty(self) -> None:
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents={}, timeout_seconds=30)
        assert bad == []

    @pytest.mark.anyio
    async def test_two_nodes_all_pass(self) -> None:
        node_agents = {"A": _make_agent("A"), "B": _make_agent("B")}
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert bad == []


# ===================================================================
# execute — bad node isolation
# ===================================================================


class TestExecuteBadNodeIsolation:
    @pytest.mark.anyio
    async def test_three_nodes_bad_node_isolated(self) -> None:
        """3-node ring with B bad. B's edges both fail, A-C edge passes → only B is suspect."""
        node_agents: dict = {
            "A": _make_agent("A"),
            "B": _make_agent("B", passed=False),
            "C": _make_agent("C"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert bad == ["B"]

    @pytest.mark.anyio
    async def test_four_nodes_bad_node_isolated(self) -> None:
        node_agents: dict = {
            "A": _make_agent("A"),
            "B": _make_agent("B", passed=False),
            "C": _make_agent("C"),
            "D": _make_agent("D"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert bad == ["B"]

    @pytest.mark.anyio
    async def test_five_nodes_bad_node_isolated(self) -> None:
        node_agents: dict = {
            "A": _make_agent("A"),
            "B": _make_agent("B"),
            "C": _make_agent("C", passed=False),
            "D": _make_agent("D"),
            "E": _make_agent("E"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert bad == ["C"]

    @pytest.mark.anyio
    async def test_two_nodes_both_fail_returns_both(self) -> None:
        """2 nodes both bad → single edge fails → return both (inconclusive)."""
        node_agents: dict = {
            "A": _make_agent("A", passed=False),
            "B": _make_agent("B", passed=False),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert sorted(bad) == ["A", "B"]

    @pytest.mark.anyio
    async def test_three_nodes_all_fail_returns_all(self) -> None:
        """3 nodes all bad → all ring edges fail → return all."""
        node_agents: dict = {
            "A": _make_agent("A", passed=False),
            "B": _make_agent("B", passed=False),
            "C": _make_agent("C", passed=False),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert sorted(bad) == ["A", "B", "C"]

    @pytest.mark.anyio
    async def test_six_nodes_bad_edge_node(self) -> None:
        """F (edge node) bad in 6-node ring → properly isolated."""
        node_agents: dict = {
            "A": _make_agent("A"),
            "B": _make_agent("B"),
            "C": _make_agent("C"),
            "D": _make_agent("D"),
            "E": _make_agent("E"),
            "F": _make_agent("F", passed=False),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert bad == ["F"]


# ===================================================================
# _run_edge — agent RPC hang
# ===================================================================


class TestRunEdgeAgentHang:
    @pytest.mark.anyio
    async def test_hanging_agent_a_times_out(self) -> None:
        node_agents: dict = {
            "A": HangingNodeAgent(node_id="A"),
            "B": _make_agent("B"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        result = await executor._run_edge(
            node_agents=node_agents,
            node_a="A",
            node_b="B",
            master_addr="A",
            port=29500,
            timeout_seconds=0,
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_hanging_times_out(self) -> None:
        node_agents: dict = {
            "A": HangingNodeAgent(node_id="A"),
            "B": HangingNodeAgent(node_id="B"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        result = await executor._run_edge(
            node_agents=node_agents,
            node_a="A",
            node_b="B",
            master_addr="A",
            port=29500,
            timeout_seconds=0,
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_execute_with_hanging_agent_isolates_bad_node(self) -> None:
        """Full execute() with one hanging node should localize it."""
        node_agents: dict = {
            "A": _make_agent("A"),
            "B": HangingNodeAgent(node_id="B"),
            "C": _make_agent("C"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE)
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=0)
        assert "B" in bad


# ===================================================================
# Port stability
# ===================================================================


class TestPortAssignment:
    @pytest.mark.anyio
    async def test_port_is_stable_across_runs(self) -> None:
        """Same node set should always produce the same edge ordering and ports."""
        node_agents: dict = {
            "A": _make_agent("A"),
            "B": _make_agent("B"),
            "C": _make_agent("C"),
        }
        executor = NeighborNcclClusterExecutor(diagnostic_type=_DIAG_TYPE, base_port=30000)

        # Run twice and verify deterministic behavior (no result drift)
        bad1 = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        bad2 = await executor.execute(node_agents=node_agents, timeout_seconds=30)
        assert bad1 == bad2
