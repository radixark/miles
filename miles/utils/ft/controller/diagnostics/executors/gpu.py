from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import ClusterExecutorProtocol, NodeAgentProtocol
from miles.utils.ft.utils.diagnostic_types import DiagnosticResult
from miles.utils.ft.controller.diagnostics.utils import gather_diagnostic_results, partition_results

logger = logging.getLogger(__name__)

_GPU_DIAGNOSTIC_TYPE = "gpu"


class GpuClusterExecutor(ClusterExecutorProtocol):
    """GPU diagnostic with cross-node compute hash comparison.

    First partitions by local pass/fail (nvml + compute errors).
    Then compares compute hashes across locally-passed nodes to detect
    SDC outliers via majority vote (bitwise alignment test).
    """

    async def execute(
        self,
        node_agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        results = await gather_diagnostic_results(
            diagnostic_type=_GPU_DIAGNOSTIC_TYPE,
            node_agents=node_agents,
            timeout_seconds=timeout_seconds,
        )

        bad_node_ids = partition_results(
            results=results,
            diagnostic_type=_GPU_DIAGNOSTIC_TYPE,
        )

        bad_set = set(bad_node_ids)
        passed_results = {nid: r for nid, r in results.items() if nid not in bad_set}
        hash_outliers = _find_gpu_hash_outlier_nodes(passed_results)
        if hash_outliers:
            bad_node_ids.extend(hash_outliers)

        return bad_node_ids


def _find_gpu_hash_outlier_nodes(
    results: dict[str, DiagnosticResult],
) -> list[str]:
    """Compare compute hashes across nodes, return nodes with minority hashes.

    For each GPU index, the most common hash is considered correct
    (majority vote). Nodes whose hash differs are outliers.

    If no clear majority exists (majority_count <= total/2), we assume
    non-determinism rather than SDC and skip that GPU index.
    """
    node_gpu_hashes: dict[str, dict[str, str]] = {}
    for node_id, result in results.items():
        if result.metadata and "compute_hashes" in result.metadata:
            node_gpu_hashes[node_id] = result.metadata["compute_hashes"]

    if len(node_gpu_hashes) < 2:
        return []

    all_gpu_indices: set[str] = set()
    for hashes in node_gpu_hashes.values():
        all_gpu_indices.update(hashes.keys())

    outlier_nodes: set[str] = set()

    for gpu_idx in sorted(all_gpu_indices):
        hash_to_nodes: dict[str, list[str]] = {}
        for node_id, hashes in node_gpu_hashes.items():
            h = hashes.get(gpu_idx, "")
            if h:
                hash_to_nodes.setdefault(h, []).append(node_id)

        if len(hash_to_nodes) <= 1:
            continue

        total = sum(len(nodes) for nodes in hash_to_nodes.values())
        majority_hash = max(hash_to_nodes, key=lambda h: len(hash_to_nodes[h]))
        majority_count = len(hash_to_nodes[majority_hash])

        if majority_count <= total / 2:
            logger.warning(
                "gpu_hash_no_majority gpu_idx=%s hash_distribution=%s — "
                "possible non-determinism, skipping this GPU index",
                gpu_idx,
                {h[:12]: len(n) for h, n in hash_to_nodes.items()},
            )
            continue

        for h, nodes in hash_to_nodes.items():
            if h != majority_hash:
                logger.info(
                    "gpu_hash_outlier gpu_idx=%s nodes=%s " "outlier_hash=%s majority_hash=%s (%d/%d nodes agree)",
                    gpu_idx,
                    nodes,
                    h[:12],
                    majority_hash[:12],
                    majority_count,
                    total,
                )
                outlier_nodes.update(nodes)

    return sorted(outlier_nodes)
