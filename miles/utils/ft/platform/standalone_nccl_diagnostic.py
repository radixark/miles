"""Standalone NCCL diagnostic runners for ad-hoc and test use.

Provides high-level async functions that handle Ray actor lifecycle
internally, so callers only need a live Ray cluster with GPU nodes.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.ft.agents.diagnostics.nccl.inter_machine import InterMachineCommDiagnostic
from miles.utils.ft.agents.diagnostics.nccl.intra_machine import IntraMachineCommDiagnostic
from miles.utils.ft.controller.diagnostics.nccl.orchestrator import InterMachineOrchestrator
from miles.utils.ft.models.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)

_MIN_INTER_MACHINE_NODES = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_intra_machine_diagnostics(
    node_ids: list[str] | None = None,
    timeout_seconds: int = 120,
) -> list[DiagnosticResult]:
    """Run intra-machine NCCL all_reduce_perf on all (or specified) GPU nodes.

    Returns one DiagnosticResult per node.
    """
    nodes = _discover_gpu_nodes(node_ids=node_ids)

    async with _managed_agents(nodes) as agents:
        futures = {
            nid: agent.run_diagnostic.remote(
                diagnostic_type="intra_machine",
                timeout_seconds=timeout_seconds,
            )
            for nid, agent in agents.items()
        }

        results: list[DiagnosticResult] = []
        for node_id, ref in futures.items():
            result: DiagnosticResult = ray.get(ref, timeout=timeout_seconds + 60)
            logger.info(
                "intra_machine node=%s passed=%s details=%s",
                node_id, result.passed, result.details,
            )
            results.append(result)

        return results


async def run_inter_machine_diagnostics(
    node_ids: list[str] | None = None,
    timeout_seconds: int = 180,
) -> list[str]:
    """Run inter-machine NCCL all_gather_perf across all (or specified) GPU nodes.

    Returns list of bad node IDs (empty if all healthy).
    """
    nodes = _discover_gpu_nodes(node_ids=node_ids)

    if len(nodes) < _MIN_INTER_MACHINE_NODES:
        logger.info(
            "inter_machine_skip — fewer than %d GPU nodes (%d found)",
            _MIN_INTER_MACHINE_NODES, len(nodes),
        )
        return []

    async with _managed_agents(nodes) as agents:
        node_addresses = _build_node_addresses(nodes)
        orchestrator = InterMachineOrchestrator(
            node_agents=agents,
            node_addresses=node_addresses,
        )

        return await orchestrator.run(
            node_ids=sorted(agents.keys()),
            timeout_seconds=timeout_seconds,
        )


# ---------------------------------------------------------------------------
# Node discovery
# ---------------------------------------------------------------------------

def _discover_gpu_nodes(
    node_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    nodes = [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
    ]

    if node_ids is not None:
        allowed = set(node_ids)
        nodes = [n for n in nodes if n["NodeID"] in allowed]

    return nodes


def _build_node_addresses(nodes: list[dict[str, Any]]) -> dict[str, str]:
    return {
        node["NodeID"]: addr
        for node in nodes
        if (addr := node.get("NodeManagerAddress", ""))
    }


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=0)
class _StandaloneDiagnosticAgent:
    """Lightweight Ray actor pinned to a node for running NCCL diagnostics."""

    def __init__(self, node_id: str, num_gpus: int) -> None:
        self._node_id = node_id
        self._num_gpus = num_gpus
        self._intra = IntraMachineCommDiagnostic(num_gpus=num_gpus)
        self._inter = InterMachineCommDiagnostic(num_gpus=num_gpus)

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        if diagnostic_type == "intra_machine":
            return await self._intra.run(
                node_id=self._node_id,
                timeout_seconds=timeout_seconds,
            )

        if diagnostic_type == "inter_machine":
            return await self._inter.run(
                node_id=self._node_id,
                timeout_seconds=timeout_seconds,
                master_addr=str(kwargs.get("master_addr", "")),
                master_port=int(kwargs.get("master_port", 29500)),
            )

        raise ValueError(f"Unknown diagnostic type: {diagnostic_type}")


def _deploy_agents(
    nodes: list[dict[str, Any]],
) -> dict[str, ray.actor.ActorHandle]:
    agents: dict[str, ray.actor.ActorHandle] = {}

    for node in nodes:
        node_id = node["NodeID"]
        num_gpus = int(node["Resources"]["GPU"])
        actor = _StandaloneDiagnosticAgent.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
        ).remote(node_id=node_id, num_gpus=num_gpus)
        agents[node_id] = actor

    return agents


def _kill_agents(agents: dict[str, ray.actor.ActorHandle]) -> None:
    for actor in agents.values():
        ray.kill(actor)


@contextlib.asynccontextmanager
async def _managed_agents(
    nodes: list[dict[str, Any]],
) -> AsyncIterator[dict[str, ray.actor.ActorHandle]]:
    agents = _deploy_agents(nodes)
    try:
        yield agents
    finally:
        _kill_agents(agents)
