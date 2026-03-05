from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from miles.utils.ft.controller.diagnostics.stack_trace import (
    StackTraceAggregator,
    StackTraceDiagnostic,
)
from miles.utils.ft.models import ActionType, Decision, DiagnosticResult, TriggerType

logger = logging.getLogger(__name__)


class DiagnosticScheduler:
    """Layered progressive diagnostic pipeline.

    Runs registered diagnostic steps in order on all agents (nodes)
    in parallel. Failed nodes are excluded from subsequent steps.
    """

    def __init__(
        self,
        agents: dict[str, Any],
        pipeline: list[str] | None = None,
        default_timeout_seconds: int = 120,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> None:
        self._agents = agents
        self._pipeline = pipeline or []
        self._default_timeout_seconds = default_timeout_seconds
        self._rank_pids_provider = rank_pids_provider

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: str,
        suspect_node_ids: list[str] | None = None,
    ) -> Decision:
        logger.info(
            "diagnostic_pipeline_start trigger=%s suspect_nodes=%s pipeline=%s",
            trigger_reason, suspect_node_ids, self._pipeline,
        )

        if trigger_reason == TriggerType.HANG and self._rank_pids_provider is not None:
            suspect_from_trace = await self._run_stack_trace_pre_step()
            if suspect_from_trace:
                if suspect_node_ids is not None:
                    suspect_node_ids = sorted(set(suspect_node_ids) | set(suspect_from_trace))
                else:
                    suspect_node_ids = suspect_from_trace

        if not self._pipeline:
            logger.info("diagnostic_pipeline_empty — all pass by default")
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="all diagnostics passed (empty pipeline)",
            )

        if suspect_node_ids is not None:
            remaining_agents = {
                nid: agent for nid, agent in self._agents.items()
                if nid in suspect_node_ids
            }
        else:
            remaining_agents = dict(self._agents)

        for diagnostic_type in self._pipeline:
            if not remaining_agents:
                break

            bad_node_ids, remaining_agents = await self._run_step(
                diagnostic_type=diagnostic_type,
                agents=remaining_agents,
                timeout_seconds=self._default_timeout_seconds,
            )

            if bad_node_ids:
                logger.info(
                    "diagnostic_step_found_bad step=%s bad_nodes=%s",
                    diagnostic_type, bad_node_ids,
                )
                return Decision(
                    action=ActionType.MARK_BAD_AND_RESTART,
                    bad_node_ids=sorted(bad_node_ids),
                    reason=f"diagnostic '{diagnostic_type}' failed on nodes: {bad_node_ids}",
                )

        logger.info("diagnostic_pipeline_all_passed trigger=%s", trigger_reason)
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="all diagnostics passed — no bad nodes found",
        )

    async def _run_stack_trace_pre_step(self) -> list[str]:
        assert self._rank_pids_provider is not None

        traces: dict[str, str] = {}
        suspect_from_failures: list[str] = []

        async def _collect_node(node_id: str) -> None:
            rank_pids = self._rank_pids_provider(node_id)
            if not rank_pids:
                return

            pids = list(rank_pids.values())
            diag = StackTraceDiagnostic(pids=pids)
            try:
                result = await diag.run(
                    node_id=node_id,
                    timeout_seconds=self._default_timeout_seconds,
                )
                if result.passed:
                    traces[node_id] = result.details
                else:
                    suspect_from_failures.append(node_id)
                    logger.info(
                        "stack_trace_collection_failed node=%s details=%s",
                        node_id, result.details,
                    )
            except Exception:
                suspect_from_failures.append(node_id)
                logger.warning(
                    "stack_trace_pre_step_exception node=%s",
                    node_id,
                    exc_info=True,
                )

        await asyncio.gather(*(_collect_node(nid) for nid in self._agents))

        suspect_from_aggregation = StackTraceAggregator().aggregate(traces=traces)

        all_suspects = sorted(set(suspect_from_failures) | set(suspect_from_aggregation))
        logger.info(
            "stack_trace_pre_step_done traces_collected=%d suspect_from_failures=%s suspect_from_aggregation=%s",
            len(traces), suspect_from_failures, suspect_from_aggregation,
        )
        return all_suspects

    async def _run_step(
        self,
        diagnostic_type: str,
        agents: dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[list[str], dict[str, Any]]:
        """Run one diagnostic step on all agents.

        Returns (bad_node_ids, remaining_agents_without_bad_nodes).
        """
        node_ids = list(agents.keys())
        logger.info(
            "diagnostic_step_start type=%s nodes=%s",
            diagnostic_type, node_ids,
        )

        raw_results = await asyncio.gather(*(
            self._call_agent_diagnostic(
                agent=agents[node_id],
                node_id=node_id,
                diagnostic_type=diagnostic_type,
                timeout_seconds=timeout_seconds,
            )
            for node_id in node_ids
        ))
        results = dict(zip(node_ids, raw_results))

        bad_node_ids: list[str] = []
        remaining: dict[str, Any] = {}
        for node_id, result in results.items():
            if result.passed:
                remaining[node_id] = agents[node_id]
            else:
                bad_node_ids.append(node_id)
                logger.info(
                    "diagnostic_node_failed type=%s node=%s details=%s",
                    diagnostic_type, node_id, result.details,
                )

        return bad_node_ids, remaining

    async def _call_agent_diagnostic(
        self,
        agent: Any,
        node_id: str,
        diagnostic_type: str,
        timeout_seconds: int,
    ) -> DiagnosticResult:
        try:
            return await agent.run_diagnostic(
                diagnostic_type, timeout_seconds=timeout_seconds,
            )
        except Exception:
            logger.warning(
                "diagnostic_agent_call_failed node=%s type=%s",
                node_id, diagnostic_type,
                exc_info=True,
            )
            return DiagnosticResult(
                diagnostic_type=diagnostic_type,
                node_id=node_id,
                passed=False,
                details="agent call raised exception",
            )
