from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import NodeAgentProtocol
from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyThread
from miles.utils.ft.controller.diagnostics.stack_trace.aggregator import StackTraceAggregator
from miles.utils.ft.controller.diagnostics.utils import call_agent_diagnostic

logger = logging.getLogger(__name__)


async def collect_stack_trace_suspects(
    node_agents: dict[str, NodeAgentProtocol],
    rank_pids_provider: Callable[[str], dict[int, int]],
    default_timeout_seconds: int,
) -> list[str]:
    """Collect stack traces from all nodes and identify suspects via aggregation."""
    traces: dict[str, list[PySpyThread]] = {}
    suspect_from_failures: list[str] = []

    async def _collect_node(node_id: str) -> None:
        try:
            rank_pids = rank_pids_provider(node_id)
        except Exception:
            suspect_from_failures.append(node_id)
            logger.warning(
                "rank_pids_provider_failed node=%s",
                node_id,
                exc_info=True,
            )
            return

        if not rank_pids:
            return

        result = await call_agent_diagnostic(
            agent=node_agents[node_id],
            node_id=node_id,
            diagnostic_type="stack_trace",
            timeout_seconds=default_timeout_seconds,
            pids=list(rank_pids.values()),
        )

        if result.passed:
            try:
                threads = [PySpyThread.model_validate(t) for t in json.loads(result.details)]
            except (json.JSONDecodeError, Exception) as exc:
                suspect_from_failures.append(node_id)
                logger.warning(
                    "stack_trace_parse_failed node=%s: %s",
                    node_id,
                    exc,
                    exc_info=True,
                )
                return
            traces[node_id] = threads
        else:
            suspect_from_failures.append(node_id)
            logger.info(
                "stack_trace_collection_failed node=%s details=%s",
                node_id,
                result.details,
            )

    await asyncio.gather(*(_collect_node(nid) for nid in node_agents))

    aggregation_result = StackTraceAggregator().aggregate(traces=traces)
    all_suspects = sorted(set(suspect_from_failures) | set(aggregation_result.suspect_node_ids))

    logger.info(
        "collect_stack_trace_suspects_done traces_collected=%d suspect_from_failures=%s suspect_from_aggregation=%s",
        len(traces),
        suspect_from_failures,
        aggregation_result.suspect_node_ids,
    )
    return all_suspects
