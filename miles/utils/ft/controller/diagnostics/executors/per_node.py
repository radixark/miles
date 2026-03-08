from __future__ import annotations

from miles.utils.ft.controller.diagnostics.utils import (
    gather_diagnostic_results,
    partition_results,
)
from miles.utils.ft.protocols.agents import NodeAgentProtocol


class PerNodeClusterExecutor:
    """Run one diagnostic type on every node independently, partition by pass/fail."""

    def __init__(self, diagnostic_type: str) -> None:
        self._diagnostic_type = diagnostic_type

    async def execute(
        self,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        results = await gather_diagnostic_results(
            diagnostic_type=self._diagnostic_type,
            agents=agents,
            timeout_seconds=timeout_seconds,
        )
        return partition_results(
            results=results,
            diagnostic_type=self._diagnostic_type,
        )
