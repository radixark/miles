"""Node agent registry — owns the shared collection of registered node agents."""

from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import NodeAgentProtocol

logger = logging.getLogger(__name__)


class NodeAgentRegistry:
    """Single-threaded asyncio-only registry for node agents.

    Wraps the mutable dict that was previously shared as a raw reference
    between FtController (writer), TickLoop and DiagnosticOrchestrator (readers).
    Not thread-safe — all access must happen on the same event loop.
    """

    def __init__(self) -> None:
        self._agents: dict[str, NodeAgentProtocol] = {}
        self._metadata: dict[str, dict[str, str]] = {}

    def register(
        self,
        node_id: str,
        agent: NodeAgentProtocol,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._agents[node_id] = agent
        if metadata:
            self._metadata[node_id] = metadata
        logger.info(
            "node_agents: registered node_id=%s, metadata_keys=%s, total_agents=%d",
            node_id,
            sorted(metadata) if metadata else "(none)",
            len(self._agents),
        )

    def get(self, node_id: str) -> NodeAgentProtocol | None:
        return self._agents.get(node_id)

    def get_all(self) -> dict[str, NodeAgentProtocol]:
        return dict(self._agents)

    def registered_node_ids(self) -> set[str]:
        return set(self._agents.keys())

    def unregister(self, node_id: str) -> None:
        was_present = node_id in self._agents
        self._agents.pop(node_id, None)
        self._metadata.pop(node_id, None)
        if was_present:
            logger.info("node_agents: unregistered node_id=%s, remaining=%d", node_id, len(self._agents))
        else:
            logger.debug("node_agents: unregister no-op, node_id=%s not found", node_id)

    def clear(self) -> None:
        count = len(self._agents)
        self._agents.clear()
        self._metadata.clear()
        logger.info("node_agents: cleared all agents, removed=%d", count)

    def get_metadata(self, node_id: str) -> dict[str, str] | None:
        return self._metadata.get(node_id)

    @property
    def all_metadata(self) -> dict[str, dict[str, str]]:
        return dict(self._metadata)
