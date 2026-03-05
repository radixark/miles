from __future__ import annotations

import asyncio
import logging
from typing import Any

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.diagnostics.inter_machine_comm import (
    InterMachineCommDiagnostic,
)
from miles.utils.ft.models import ActionType, Decision, DiagnosticResult

logger = logging.getLogger(__name__)

_INTER_MACHINE_TYPE = "inter_machine"
_INTER_MACHINE_BASE_PORT = 29500


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
        agents: dict[str, Any],
        pipeline: list[str] | None = None,
        default_timeout_seconds: int = 120,
        node_addresses: dict[str, str] | None = None,
    ) -> None:
        self._agents = agents
        self._pipeline = pipeline or []
        self._default_timeout_seconds = default_timeout_seconds
        self._node_addresses = node_addresses

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: str,
        suspect_node_ids: list[str] | None = None,
    ) -> Decision:
        logger.info(
            "diagnostic_pipeline_start trigger=%s suspect_nodes=%s pipeline=%s",
            trigger_reason, suspect_node_ids, self._pipeline,
        )

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

            if diagnostic_type == _INTER_MACHINE_TYPE:
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

    # ------------------------------------------------------------------
    # Single-node step (gpu, intra_machine, etc.)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Inter-machine step (multi-node coordination)
    # ------------------------------------------------------------------

    async def _run_inter_machine_step(
        self,
        agents: dict[str, Any],
        timeout_seconds: int,
    ) -> list[str]:
        """Run inter-machine communication diagnostics with cross-comparison.

        Pairs nodes in a round-robin ring, runs all_gather_perf on each
        pair simultaneously, then uses failure counts to isolate the bad
        node.

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
        logger.info(
            "inter_machine_step_start pairs=%s",
            [(m, w) for m, w in pairs],
        )

        tasks = []
        for pair_index, (master_id, worker_id) in enumerate(pairs):
            port = _INTER_MACHINE_BASE_PORT + pair_index
            master_addr = self._get_node_address(master_id)
            tasks.append(self._run_single_pair(
                master_id=master_id,
                worker_id=worker_id,
                master_addr=master_addr,
                port=port,
                agents=agents,
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
        agents: dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[str, str, bool]:
        """Run one pair test. Returns (master_id, worker_id, passed)."""
        master_diag = InterMachineCommDiagnostic(
            master_addr=master_addr, master_port=port,
        )
        worker_diag = InterMachineCommDiagnostic(
            master_addr=master_addr, master_port=port,
        )

        master_agent = agents[master_id]
        worker_agent = agents[worker_id]

        self._inject_diagnostic(master_agent, master_diag)
        self._inject_diagnostic(worker_agent, worker_diag)

        try:
            master_result, worker_result = await asyncio.gather(
                self._call_agent_diagnostic(
                    agent=master_agent,
                    node_id=master_id,
                    diagnostic_type=_INTER_MACHINE_TYPE,
                    timeout_seconds=timeout_seconds,
                ),
                self._call_agent_diagnostic(
                    agent=worker_agent,
                    node_id=worker_id,
                    diagnostic_type=_INTER_MACHINE_TYPE,
                    timeout_seconds=timeout_seconds,
                ),
            )
            passed = master_result.passed and worker_result.passed
        except Exception:
            logger.warning(
                "inter_machine_pair_failed master=%s worker=%s",
                master_id, worker_id,
                exc_info=True,
            )
            passed = False
        finally:
            self._remove_diagnostic(master_agent, _INTER_MACHINE_TYPE)
            self._remove_diagnostic(worker_agent, _INTER_MACHINE_TYPE)

        logger.info(
            "inter_machine_pair_result master=%s worker=%s passed=%s",
            master_id, worker_id, passed,
        )
        return master_id, worker_id, passed

    @staticmethod
    def _cross_compare(
        node_ids: list[str],
        pair_results: list[tuple[str, str, bool]],
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
        for master_id, worker_id, passed in pair_results:
            if not passed:
                failure_count[master_id] += 1
                failure_count[worker_id] += 1

        max_count = max(failure_count.values())
        if max_count == 0:
            return []

        min_count = min(failure_count.values())
        if min_count == max_count:
            logger.warning("inter_machine_all_failed — cannot localize bad node")
            return []

        bad_nodes = sorted(
            nid for nid, count in failure_count.items() if count == max_count
        )
        return bad_nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_diagnostic(agent: Any, diagnostic: BaseDiagnostic) -> None:
        set_fn = getattr(agent, "set_diagnostic", None)
        if set_fn is not None:
            set_fn(diagnostic)

    @staticmethod
    def _remove_diagnostic(agent: Any, diagnostic_type: str) -> None:
        remove_fn = getattr(agent, "remove_diagnostic", None)
        if remove_fn is not None:
            remove_fn(diagnostic_type)

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
            return DiagnosticResult(
                diagnostic_type=diagnostic_type,
                node_id=node_id,
                passed=False,
                details="agent call raised exception",
            )
