"""Node agent registry — owns the shared collection of registered node agents."""

from __future__ import annotations

from miles.utils.ft.adapters.types import NodeAgentProtocol


class NodeAgentRegistry:
    """Thread-safe-ish registry for node agents.

    Wraps the mutable dict that was previously shared as a raw reference
    between FtController (writer), TickLoop and DiagnosticOrchestrator (readers).
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

    def get(self, node_id: str) -> NodeAgentProtocol | None:
        return self._agents.get(node_id)

    def get_all(self) -> dict[str, NodeAgentProtocol]:
        return dict(self._agents)

    def registered_node_ids(self) -> set[str]:
        return set(self._agents.keys())

    def get_metadata(self, node_id: str) -> dict[str, str] | None:
        return self._metadata.get(node_id)

    @property
    def all_metadata(self) -> dict[str, dict[str, str]]:
        return self._metadata
