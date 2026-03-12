from __future__ import annotations

import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import ClusterExecutorProtocol, NodeAgentProtocol
from miles.utils.ft.controller.diagnostics.stack_trace import collect_stack_trace_suspects

logger = logging.getLogger(__name__)


class StackTraceClusterExecutor(ClusterExecutorProtocol):
    """Identifies outlier nodes via stack trace aggregation and evicts them.

    Used as the first executor in the pipeline when the trigger is HANG.
    If outliers are found they are returned as bad_node_ids immediately;
    otherwise the pipeline continues to hardware diagnostics.
    """

    def __init__(self, rank_pids_provider: Callable[[str], dict[int, int]]) -> None:
        self._rank_pids_provider = rank_pids_provider

    async def execute(
        self,
        node_agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        suspects = await collect_stack_trace_suspects(
            node_agents=node_agents,
            rank_pids_provider=self._rank_pids_provider,
            default_timeout_seconds=timeout_seconds,
        )
        return suspects if suspects else []
