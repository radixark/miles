"""Neighbor-based NCCL diagnostic executor.

Builds a ring topology over sorted node IDs, then runs NCCL diagnostics
on each neighbor edge exactly once.  Suspects are identified by local
failure-ratio analysis rather than global max-count scoring, which
avoids the systematic mis-attribution that occurs with odd-sized clusters
in pairwise round-robin schemes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from miles.utils.ft.adapters.types import ClusterExecutorProtocol, NodeAgentProtocol
from miles.utils.ft.agents.diagnostics.executors.nccl import DEFAULT_NCCL_MASTER_PORT
from miles.utils.ft.controller.diagnostics.utils import RPC_TIMEOUT_BUFFER_SECONDS

logger = logging.getLogger(__name__)


class _EdgeResult(NamedTuple):
    node_a: str
    node_b: str
    passed: bool


def _build_ring_edges(sorted_ids: list[str]) -> list[tuple[str, str]]:
    """Build undirected ring-topology edges from sorted node IDs.

    Each node connects to its right neighbor ``(i, i+1) % n`` in a ring.
    Edges are deduplicated and represented as ``(min_id, max_id)`` tuples.
    For 2 nodes, produces a single edge.  For N >= 3, produces N edges
    forming a complete ring.
    """
    n = len(sorted_ids)
    if n < 2:
        return []

    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for i in range(n):
        a, b = sorted_ids[i], sorted_ids[(i + 1) % n]
        edge = (min(a, b), max(a, b))
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)

    return edges


def _localize_suspects_from_neighbor_results(
    sorted_ids: list[str],
    edge_results: list[_EdgeResult],
) -> list[str]:
    """Identify suspect nodes from neighbor edge diagnostics.

    Rules (coarse-grained suspect isolation):
    - If a node has ALL its neighbor edges failed, and at least one of its
      neighbors has a passing edge elsewhere, this node is a suspect.
    - If a connected component of failures cannot uniquely isolate a node
      (e.g. all edges failed), return the entire set as suspects.
    - For 2 nodes with the single edge failed, return both (inconclusive).
    - Never miss a truly bad node (over-eviction is acceptable).
    """
    if not edge_results:
        return []

    nodes_with_pass: set[str] = set()
    nodes_with_fail: set[str] = set()

    for er in edge_results:
        if er.passed:
            nodes_with_pass.update((er.node_a, er.node_b))
        else:
            nodes_with_fail.update((er.node_a, er.node_b))

    if not nodes_with_fail:
        return []
    if not nodes_with_pass:
        return sorted_ids[:]

    suspects = nodes_with_fail - nodes_with_pass

    if not suspects:
        suspects = nodes_with_fail

    return sorted(suspects)


class NeighborNcclClusterExecutor(ClusterExecutorProtocol):
    """Ring-topology neighbor NCCL diagnostic executor.

    Replaces the round-robin pairwise approach with a ring neighbor
    topology that provides uniform participation for both odd and even
    cluster sizes.
    """

    def __init__(
        self,
        diagnostic_type: str,
        base_port: int = DEFAULT_NCCL_MASTER_PORT,
    ) -> None:
        self._diagnostic_type = diagnostic_type
        self._base_port = base_port

    async def execute(
        self,
        node_agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        sorted_ids = sorted(node_agents.keys())

        if len(sorted_ids) < 2:
            logger.info("diagnostics: neighbor nccl skipped, fewer than 2 nodes")
            return []

        edges = _build_ring_edges(sorted_ids)
        logger.info("diagnostics: neighbor nccl start edges=%d, nodes=%d", len(edges), len(sorted_ids))

        tasks = []
        for edge_index, (node_a, node_b) in enumerate(edges):
            port = self._base_port + edge_index
            tasks.append(
                self._run_edge(
                    node_agents=node_agents,
                    node_a=node_a,
                    node_b=node_b,
                    master_addr=node_a,
                    port=port,
                    timeout_seconds=timeout_seconds,
                )
            )

        results = await asyncio.gather(*tasks)

        return _localize_suspects_from_neighbor_results(
            sorted_ids=sorted_ids,
            edge_results=list(results),
        )

    async def _run_edge(
        self,
        node_agents: dict[str, NodeAgentProtocol],
        node_a: str,
        node_b: str,
        master_addr: str,
        port: int,
        timeout_seconds: int,
    ) -> _EdgeResult:
        agent_a = node_agents.get(node_a)
        agent_b = node_agents.get(node_b)

        if agent_a is None or agent_b is None:
            logger.warning(
                "diagnostics: neighbor nccl edge skip, missing agent a=%s(%s), b=%s(%s)",
                node_a,
                "ok" if agent_a else "missing",
                node_b,
                "ok" if agent_b else "missing",
            )
            return _EdgeResult(node_a=node_a, node_b=node_b, passed=False)

        try:
            result_a, result_b = await asyncio.wait_for(
                asyncio.gather(
                    agent_a.run_diagnostic(
                        diagnostic_type=self._diagnostic_type,
                        timeout_seconds=timeout_seconds,
                        master_addr=master_addr,
                        master_port=port,
                    ),
                    agent_b.run_diagnostic(
                        diagnostic_type=self._diagnostic_type,
                        timeout_seconds=timeout_seconds,
                        master_addr=master_addr,
                        master_port=port,
                    ),
                ),
                timeout=timeout_seconds + RPC_TIMEOUT_BUFFER_SECONDS,
            )
            passed = result_a.passed and result_b.passed
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostics: neighbor nccl edge timeout a=%s, b=%s, timeout=%d",
                node_a,
                node_b,
                timeout_seconds,
            )
            passed = False
        except Exception:
            logger.warning(
                "diagnostics: neighbor nccl edge failed a=%s, b=%s",
                node_a,
                node_b,
                exc_info=True,
            )
            passed = False

        logger.info(
            "diagnostics: neighbor nccl edge result a=%s, b=%s, passed=%s",
            node_a,
            node_b,
            passed,
        )
        return _EdgeResult(node_a=node_a, node_b=node_b, passed=passed)
