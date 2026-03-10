"""Shared Ray node discovery utilities.

Centralises ray.nodes() traversal logic used by both RayMainJob
(node-id resolution) and standalone NCCL diagnostics (GPU node discovery).
"""

from __future__ import annotations

import logging
from typing import Any

import ray

logger = logging.getLogger(__name__)

CPU_ONLY_RESOURCE = "cpu_only"


def assert_cpu_only_nodes_exist() -> None:
    """Assert that at least one alive Ray node has the ``cpu_only`` custom resource.

    Raises ``RuntimeError`` immediately if no such node exists, so callers
    fail fast instead of hanging on an unsatisfiable scheduling request.
    """
    for node in ray.nodes():
        if node.get("Alive") and node.get("Resources", {}).get(CPU_ONLY_RESOURCE, 0) > 0:
            return

    raise RuntimeError(
        f"No alive Ray node has the '{CPU_ONLY_RESOURCE}' custom resource. "
        "Ensure at least one non-GPU node (e.g. the head node) is started with "
        f"--resources='{{\"cpu_only\": 1}}' in its rayStartParams."
    )


def get_alive_gpu_nodes(
    node_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return alive Ray nodes that have GPUs.

    If *node_ids* is given, only return nodes whose NodeID is in the list.
    """
    nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]

    if node_ids is not None:
        allowed = set(node_ids)
        nodes = [n for n in nodes if n["NodeID"] in allowed]

    return nodes


def build_node_address_map(nodes: list[dict[str, Any]]) -> dict[str, str]:
    """Build a mapping from NodeID to NodeManagerAddress."""
    return {node["NodeID"]: addr for node in nodes if (addr := node.get("NodeManagerAddress", ""))}
