from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from miles.utils.ft.controller.diagnostics.inter_machine_comm import (
    InterMachineCommDiagnostic,
)
from miles.utils.ft.controller.diagnostics.stack_trace import (
    StackTraceAggregator,
    StackTraceDiagnostic,
)
from miles.utils.ft.models import (
    ActionType,
    Decision,
    DiagnosticResult,
    NodeAgentProtocol,
    TriggerType,
)

logger = logging.getLogger(__name__)

_INTER_MACHINE_BASE_PORT = 29500


class PairResult(NamedTuple):
    master_id: str
    worker_id: str
    passed: bool


class DiagnosticScheduler:
    """Layered progressive diagnostic pipeline.

    Runs registered diagnostic steps in order on all agents (nodes)
    in parallel. Failed nodes are excluded from subsequent steps.

    The ``"inter_machine"`` step is handled specially: nodes are paired
    in a round-robin ring and tested simultaneously with NCCL
    all_gather_perf.  Cross-comparison isolates the bad node.
    """

    def __init__(
        self,
        agents: dict[str, NodeAgentProtocol],
        pipeline: list[str] | None = None,
        default_timeout_seconds: int = 120,
        node_addresses: dict[str, str] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> None:
        self._agents = agents
        self._pipeline = pipeline or []
        self._default_timeout_seconds = default_timeout_seconds
        self._node_addresses = node_addresses
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
            remaining_agents: dict[str, NodeAgentProtocol] = {
                nid: agent for nid, agent in self._agents.items()
                if nid in suspect_node_ids
            }
        else:
            remaining_agents: dict[str, NodeAgentProtocol] = dict(self._agents)

        for diagnostic_type in self._pipeline:
            if not remaining_agents:
                break

            if diagnostic_type == InterMachineCommDiagnostic.diagnostic_type:
                bad_node_ids = await self._run_inter_machine_step(
                    agents=remaining_agents,
                    timeout_seconds=self._default_timeout_seconds,
                )
            else:
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

    # ------------------------------------------------------------------
    # Single-node step (gpu, intra_machine, etc.)
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        diagnostic_type: str,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> tuple[list[str], dict[str, NodeAgentProtocol]]:
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
        remaining: dict[str, NodeAgentProtocol] = {}
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

    # ------------------------------------------------------------------
    # Inter-machine step (multi-node coordination)
    # ------------------------------------------------------------------

    async def _run_inter_machine_step(
        self,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        """Run inter-machine communication diagnostics with cross-comparison.

        Pairs nodes in a round-robin ring, runs all_gather_perf on each
        pair simultaneously, then uses failure counts to isolate the bad
        node.

        Each pair gets its own diagnostic instance called directly (not
        injected into agents) to avoid race conditions when a node
        participates in multiple concurrent pairs.

        Returns list of bad node IDs (empty if all pass or cannot localize).
        """
        node_ids = sorted(agents.keys())

        if len(node_ids) < 2:
            logger.info("inter_machine_skip — fewer than 2 nodes")
            return []

        pairs = [
            (node_ids[i], node_ids[(i + 1) % len(node_ids)])
            for i in range(len(node_ids))
        ]
        logger.info("inter_machine_step_start pairs=%s", pairs)

        tasks = []
        for pair_index, (master_id, worker_id) in enumerate(pairs):
            port = _INTER_MACHINE_BASE_PORT + pair_index
            master_addr = self._get_node_address(master_id)
            tasks.append(self._run_single_pair(
                master_id=master_id,
                worker_id=worker_id,
                master_addr=master_addr,
                port=port,
                timeout_seconds=timeout_seconds,
            ))

        results = await asyncio.gather(*tasks)

        return self._cross_compare(node_ids=node_ids, pair_results=results)

    async def _run_single_pair(
        self,
        master_id: str,
        worker_id: str,
        master_addr: str,
        port: int,
        timeout_seconds: int,
    ) -> PairResult:
        """Run one pair test.

        Creates a single diagnostic instance per pair and calls run()
        directly for each side (no agent injection needed).
        """
        diag = InterMachineCommDiagnostic(
            master_addr=master_addr, master_port=port,
        )

        try:
            master_result, worker_result = await asyncio.gather(
                diag.run(node_id=master_id, timeout_seconds=timeout_seconds),
                diag.run(node_id=worker_id, timeout_seconds=timeout_seconds),
            )
            passed = master_result.passed and worker_result.passed
        except Exception:
            logger.warning(
                "inter_machine_pair_failed master=%s worker=%s",
                master_id, worker_id,
                exc_info=True,
            )
            passed = False

        logger.info(
            "inter_machine_pair_result master=%s worker=%s passed=%s",
            master_id, worker_id, passed,
        )
        return PairResult(master_id=master_id, worker_id=worker_id, passed=passed)

    @staticmethod
    def _cross_compare(
        node_ids: list[str],
        pair_results: list[PairResult],
    ) -> list[str]:
        """Cross-compare pair results to isolate bad nodes.

        Algorithm:
        1. Count failures per node across all pairs.
        2. If no failures → return empty (all healthy).
        3. Find nodes with the highest failure count.
        4. If ALL nodes share the same (non-zero) failure count → cannot
           localize → return empty (NOTIFY_HUMAN).
        5. Otherwise → return nodes with the highest failure count.
        """
        failure_count: dict[str, int] = {nid: 0 for nid in node_ids}
        for result in pair_results:
            if not result.passed:
                failure_count[result.master_id] += 1
                failure_count[result.worker_id] += 1

        counts = list(failure_count.values())
        max_count = max(counts)
        if max_count == 0:
            return []

        if min(counts) == max_count:
            logger.warning("inter_machine_all_failed — cannot localize bad node")
            return []

        bad_nodes = sorted(
            nid for nid, count in failure_count.items() if count == max_count
        )
        return bad_nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_node_address(self, node_id: str) -> str:
        if self._node_addresses and node_id in self._node_addresses:
            return self._node_addresses[node_id]
        return node_id

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
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type, node_id=node_id,
                details="agent call raised exception",
            )
