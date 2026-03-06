"""Standalone NCCL diagnostic smoke test.

Runs intra-machine and inter-machine NCCL bandwidth tests on every GPU
node in the Ray cluster.  Does NOT require training, FT controller, or
any other E2E infrastructure — only a live Ray cluster with GPU nodes
that have nccl-tests binaries installed.

Expected result: all pass (healthy cluster).  Also serves as a cluster
network health canary.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.ft.controller.diagnostics.inter_machine_comm import (
    InterMachineCommDiagnostic,
)
from miles.utils.ft.controller.diagnostics.inter_machine_orchestrator import (
    InterMachineOrchestrator,
)
from miles.utils.ft.controller.diagnostics.intra_machine_comm import (
    IntraMachineCommDiagnostic,
)
from miles.utils.ft.models._diagnostics import DiagnosticResult
from tests.e2e.ft.conftest import gpu_nodes

logger = logging.getLogger(__name__)

_MIN_NODES = 2


# ---------------------------------------------------------------------------
# Ray actor: lightweight diagnostic runner pinned to a specific node
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=0)
class _DiagnosticOnlyAgent:
    """Minimal agent for running NCCL diagnostics on a specific node.

    Handles the master_addr/master_port kwargs that InterMachineOrchestrator
    passes to run_diagnostic() by constructing a fresh InterMachineCommDiagnostic
    per call (the diagnostic reads these values from __init__, not from run()).
    """

    def __init__(self, node_id: str, num_gpus: int) -> None:
        self._node_id = node_id
        self._num_gpus = num_gpus
        self._intra = IntraMachineCommDiagnostic(num_gpus=num_gpus)

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
            diag = InterMachineCommDiagnostic(
                num_gpus=self._num_gpus,
                master_addr=str(kwargs.get("master_addr", "")),
                master_port=int(kwargs.get("master_port", 29500)),
            )
            return await diag.run(
                node_id=self._node_id,
                timeout_seconds=timeout_seconds,
            )

        raise ValueError(f"Unknown diagnostic type: {diagnostic_type}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diagnostic_agents(
    ray_cluster: None,
) -> Generator[dict[str, ray.actor.ActorHandle], None, None]:
    """Deploy diagnostic-only agents on all GPU nodes."""
    nodes = gpu_nodes()
    if len(nodes) < _MIN_NODES:
        pytest.skip(
            f"Need >= {_MIN_NODES} GPU nodes for NCCL diagnostics, got {len(nodes)}"
        )

    agents: dict[str, ray.actor.ActorHandle] = {}
    for node in nodes:
        node_id = node["NodeID"]
        num_gpus = int(node["Resources"]["GPU"])
        actor = _DiagnosticOnlyAgent.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
        ).remote(node_id=node_id, num_gpus=num_gpus)
        agents[node_id] = actor

    logger.info("diagnostic_agents_deployed nodes=%d", len(agents))
    yield agents

    for actor in agents.values():
        ray.kill(actor)


@pytest.fixture(scope="module")
def node_addresses(ray_cluster: None) -> dict[str, str]:
    """Build node_id → IP address mapping from Ray cluster metadata."""
    mapping: dict[str, str] = {}
    for node in gpu_nodes():
        addr = node.get("NodeManagerAddress", "")
        if addr:
            mapping[node["NodeID"]] = addr
    return mapping


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_intra_machine_all_pass(
    diagnostic_agents: dict[str, ray.actor.ActorHandle],
) -> None:
    """Every GPU node should pass intra-machine all_reduce_perf."""
    futures = {
        node_id: agent.run_diagnostic.remote(
            diagnostic_type="intra_machine",
            timeout_seconds=120,
        )
        for node_id, agent in diagnostic_agents.items()
    }

    failed_nodes: list[str] = []
    for node_id, ref in futures.items():
        result: DiagnosticResult = ray.get(ref, timeout=180)
        logger.info(
            "intra_machine node=%s passed=%s details=%s",
            node_id, result.passed, result.details,
        )
        if not result.passed:
            failed_nodes.append(node_id)

    assert not failed_nodes, (
        f"Intra-machine NCCL diagnostic failed on nodes: {failed_nodes}"
    )


async def test_inter_machine_all_pass(
    diagnostic_agents: dict[str, ray.actor.ActorHandle],
    node_addresses: dict[str, str],
) -> None:
    """All node pairs should pass inter-machine all_gather_perf."""
    orchestrator = InterMachineOrchestrator(
        agents=diagnostic_agents,
        node_addresses=node_addresses,
    )

    node_ids = sorted(diagnostic_agents.keys())
    bad_nodes = await orchestrator.run(
        node_ids=node_ids,
        timeout_seconds=180,
    )

    assert bad_nodes == [], (
        f"Inter-machine NCCL diagnostic identified bad nodes: {bad_nodes}"
    )
