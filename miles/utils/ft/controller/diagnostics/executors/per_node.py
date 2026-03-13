from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import ClusterExecutorProtocol, NodeAgentProtocol
from miles.utils.ft.controller.diagnostics.utils import gather_diagnostic_results, partition_results

logger = logging.getLogger(__name__)


class PerNodeClusterExecutor(ClusterExecutorProtocol):
    """Run one diagnostic type on every node independently, partition by pass/fail."""

    def __init__(self, diagnostic_type: str) -> None:
        self._diagnostic_type = diagnostic_type

    async def execute(
        self,
        node_agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        logger.info(
            "diagnostics: per_node execute type=%s, node_count=%d, timeout=%d",
            self._diagnostic_type,
            len(node_agents),
            timeout_seconds,
        )
        results = await gather_diagnostic_results(
            diagnostic_type=self._diagnostic_type,
            node_agents=node_agents,
            timeout_seconds=timeout_seconds,
        )
        bad_node_ids = partition_results(
            results=results,
            diagnostic_type=self._diagnostic_type,
        )
        logger.info(
            "diagnostics: per_node done type=%s, bad_nodes=%s",
            self._diagnostic_type,
            bad_node_ids,
        )
        return bad_node_ids
