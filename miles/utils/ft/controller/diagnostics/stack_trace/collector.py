from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import NodeAgentProtocol
from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyThread
from miles.utils.ft.controller.diagnostics.stack_trace.aggregator import StackTraceAggregator, StackTraceTieError
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
                "diagnostics: rank_pids_provider failed node=%s",
                node_id,
                exc_info=True,
            )
            return

        if not rank_pids:
            suspect_from_failures.append(node_id)
            logger.warning("diagnostics: stack trace no PIDs for node=%s", node_id)
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
                    "diagnostics: stack trace parse failed node=%s, error=%s",
                    node_id,
                    exc,
                    exc_info=True,
                )
                return
            if not threads:
                suspect_from_failures.append(node_id)
                logger.warning("diagnostics: stack trace empty threads node=%s", node_id)
                return
            traces[node_id] = threads
        else:
            suspect_from_failures.append(node_id)
            logger.info(
                "diagnostics: stack trace collection failed node=%s, details=%s",
                node_id,
                result.details,
            )

    await asyncio.gather(*(_collect_node(nid) for nid in node_agents))

    aggregation_suspects: list[str] = []
    try:
        aggregation_result = StackTraceAggregator().aggregate(traces=traces)
        aggregation_suspects = aggregation_result.suspect_node_ids
    except StackTraceTieError:
        logger.warning("diagnostics: stack trace aggregation tie traces_collected=%d", len(traces), exc_info=True)
        if suspect_from_failures:
            logger.info("diagnostics: aggregation tie suppressed, returning %d failure suspects", len(suspect_from_failures))
        else:
            raise

    all_suspects = sorted(set(suspect_from_failures) | set(aggregation_suspects))

    logger.info(
        "diagnostics: stack trace collection done traces=%d, failure_suspects=%s, aggregation_suspects=%s",
        len(traces),
        suspect_from_failures,
        aggregation_suspects,
    )
    return all_suspects
