"""Standalone diagnostic runners for ad-hoc and test use.

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

from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.inter_machine import InterMachineNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.intra_machine import IntraMachineNodeExecutor
from miles.utils.ft.agents.diagnostics.runner import NodeExecutorRunner
from miles.utils.ft.controller.diagnostics.executors import InterMachineClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.gpu import find_gpu_hash_outlier_nodes
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.platform.ray_wrappers.node_discovery import build_node_address_map, get_alive_gpu_nodes
from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_MIN_INTER_MACHINE_NODES = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_intra_machine_diagnostics(
    node_ids: list[str] | None = None,
    timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
) -> list[DiagnosticResult]:
    """Run intra-machine NCCL all_reduce_perf on all (or specified) GPU nodes.

    Returns one DiagnosticResult per node.
    """
    nodes = get_alive_gpu_nodes(node_ids=node_ids)

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
                node_id,
                result.passed,
                result.details,
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
    nodes = get_alive_gpu_nodes(node_ids=node_ids)

    if len(nodes) < _MIN_INTER_MACHINE_NODES:
        logger.info(
            "inter_machine_skip — fewer than %d GPU nodes (%d found)",
            _MIN_INTER_MACHINE_NODES,
            len(nodes),
        )
        return []

    async with _managed_agents(nodes) as agents:
        node_addresses = build_node_address_map(nodes)
        executor = InterMachineClusterExecutor(node_addresses=node_addresses)
        return await executor.execute(
            agents=agents,
            timeout_seconds=timeout_seconds,
        )


async def run_gpu_diagnostics(
    node_ids: list[str] | None = None,
    timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
) -> tuple[list[DiagnosticResult], list[str]]:
    """Run GPU diagnostic on all GPU nodes + cross-node hash comparison.

    Returns (per_node_results, outlier_node_ids).
    """
    nodes = get_alive_gpu_nodes(node_ids=node_ids)

    async with _managed_agents(nodes) as agents:
        futures = {
            nid: agent.run_diagnostic.remote(
                diagnostic_type="gpu",
                timeout_seconds=timeout_seconds,
            )
            for nid, agent in agents.items()
        }

        results: dict[str, DiagnosticResult] = {}
        for node_id, ref in futures.items():
            result: DiagnosticResult = ray.get(ref, timeout=timeout_seconds + 60)
            logger.info(
                "gpu node=%s passed=%s details=%s",
                node_id,
                result.passed,
                result.details,
            )
            results[node_id] = result

        outlier_node_ids = find_gpu_hash_outlier_nodes(results)

        return list(results.values()), outlier_node_ids


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=0)
class _StandaloneDiagnosticAgent:
    """Lightweight Ray actor pinned to a node for running NCCL diagnostics."""

    def __init__(self, node_id: str, num_gpus: int) -> None:
        self._runner = NodeExecutorRunner(
            node_id=node_id,
            diagnostics=[
                GpuNodeExecutor(),
                IntraMachineNodeExecutor(num_gpus=num_gpus),
                InterMachineNodeExecutor(num_gpus=num_gpus),
            ],
        )

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult:
        return await self._runner.run_diagnostic(
            diagnostic_type=diagnostic_type,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )


def _deploy_agents(
    nodes: list[dict[str, Any]],
) -> dict[str, ray.actor.ActorHandle]:
    agents: dict[str, ray.actor.ActorHandle] = {}

    for node in nodes:
        node_id = node["NodeID"]
        num_gpus = int(node["Resources"]["GPU"])
        actor = _StandaloneDiagnosticAgent.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
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
