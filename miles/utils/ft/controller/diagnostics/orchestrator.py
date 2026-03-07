from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from miles.utils.ft.agents.diagnostics.gpu_diagnostic import GpuDiagnostic
from miles.utils.ft.agents.diagnostics.nccl.inter_machine import (
    InterMachineCommDiagnostic,
)
from miles.utils.ft.controller.diagnostics.nccl.orchestrator import (
    InterMachineOrchestrator,
)
from miles.utils.ft.controller.diagnostics.stack_trace import (
    collect_stack_trace_suspects,
)
from miles.utils.ft.models.diagnostics import DiagnosticResult, UnknownDiagnosticError
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.agents import NodeAgentProtocol
from miles.utils.ft.protocols.platform import DiagnosticOrchestratorProtocol

logger = logging.getLogger(__name__)


class DiagnosticOrchestrator(DiagnosticOrchestratorProtocol):
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
        pipeline_timeout_seconds: int = 900,
        node_addresses: dict[str, str] | None = None,
    ) -> None:
        self._agents = agents
        self._pipeline = pipeline or []
        self._default_timeout_seconds = default_timeout_seconds
        self._pipeline_timeout_seconds = pipeline_timeout_seconds
        self._inter_machine = InterMachineOrchestrator(
            node_agents=agents,
            node_addresses=node_addresses,
        )

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> Decision:
        logger.info(
            "diagnostic_pipeline_start trigger=%s suspect_nodes=%s pipeline=%s",
            trigger_reason, suspect_node_ids, self._pipeline,
        )

        try:
            return await asyncio.wait_for(
                self._run_diagnostic_pipeline_inner(
                    trigger_reason=trigger_reason,
                    suspect_node_ids=suspect_node_ids,
                    rank_pids_provider=rank_pids_provider,
                ),
                timeout=self._pipeline_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_pipeline_timeout timeout=%d trigger=%s",
                self._pipeline_timeout_seconds, trigger_reason,
            )
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"diagnostic pipeline timed out after {self._pipeline_timeout_seconds}s",
                trigger=trigger_reason,
            )

    async def _run_diagnostic_pipeline_inner(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> Decision:
        if trigger_reason == TriggerType.HANG and rank_pids_provider is not None:
            suspect_from_trace = await collect_stack_trace_suspects(
                agents=self._agents,
                rank_pids_provider=rank_pids_provider,
                default_timeout_seconds=self._default_timeout_seconds,
            )
            if suspect_from_trace:
                if suspect_node_ids is not None:
                    suspect_node_ids = sorted(set(suspect_node_ids) | set(suspect_from_trace))
                else:
                    suspect_node_ids = suspect_from_trace

        if not self._pipeline:
            logger.info("diagnostic_pipeline_empty — no diagnostics configured")
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="no diagnostics configured (empty pipeline)",
                trigger=trigger_reason,
            )

        if suspect_node_ids is not None:
            suspect_set = set(suspect_node_ids)
            remaining_agents: dict[str, NodeAgentProtocol] = {
                nid: agent for nid, agent in self._agents.items()
                if nid in suspect_set
            }
        else:
            remaining_agents: dict[str, NodeAgentProtocol] = dict(self._agents)

        for diagnostic_type in self._pipeline:
            if not remaining_agents:
                break

            bad_node_ids, remaining_agents = await self._run_pipeline_step(
                diagnostic_type, remaining_agents,
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
                    trigger=trigger_reason,
                )

        logger.info("diagnostic_pipeline_all_passed trigger=%s", trigger_reason)
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="all diagnostics passed — no bad nodes found",
            trigger=trigger_reason,
        )

    # ------------------------------------------------------------------
    # Single pipeline step dispatch
    # ------------------------------------------------------------------

    async def _run_pipeline_step(
        self,
        diagnostic_type: str,
        remaining_agents: dict[str, NodeAgentProtocol],
    ) -> tuple[list[str], dict[str, NodeAgentProtocol]]:
        if diagnostic_type == InterMachineCommDiagnostic.diagnostic_type:
            bad_node_ids = await self._inter_machine.run(
                node_ids=list(remaining_agents.keys()),
                timeout_seconds=self._default_timeout_seconds,
            )
            return bad_node_ids, remaining_agents

        if diagnostic_type == GpuDiagnostic.diagnostic_type:
            return await self._run_gpu_step(remaining_agents)

        return await self._run_step(
            diagnostic_type=diagnostic_type,
            agents=remaining_agents,
            timeout_seconds=self._default_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Single-node step (gpu, intra_machine, etc.)
    # ------------------------------------------------------------------

    async def _gather_diagnostic_results(
        self,
        diagnostic_type: str,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> dict[str, DiagnosticResult]:
        """Run one diagnostic on all agents and return the raw results."""
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
        return dict(zip(node_ids, raw_results))

    @staticmethod
    def _partition_results(
        results: dict[str, DiagnosticResult],
        agents: dict[str, NodeAgentProtocol],
        diagnostic_type: str,
    ) -> tuple[list[str], dict[str, NodeAgentProtocol]]:
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

    async def _run_step(
        self,
        diagnostic_type: str,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> tuple[list[str], dict[str, NodeAgentProtocol]]:
        """Run one diagnostic step on all agents.

        Returns (bad_node_ids, remaining_agents_without_bad_nodes).
        """
        results = await self._gather_diagnostic_results(
            diagnostic_type=diagnostic_type,
            agents=agents,
            timeout_seconds=timeout_seconds,
        )
        return self._partition_results(
            results=results, agents=agents, diagnostic_type=diagnostic_type,
        )

    # ------------------------------------------------------------------
    # GPU step with cross-node hash comparison
    # ------------------------------------------------------------------

    async def _run_gpu_step(
        self,
        remaining_agents: dict[str, NodeAgentProtocol],
    ) -> tuple[list[str], dict[str, NodeAgentProtocol]]:
        """Run GPU diagnostic with cross-node compute hash comparison.

        First partitions by local pass/fail (nvml + compute errors).
        Then compares compute hashes across locally-passed nodes to detect
        SDC outliers via majority vote (bitwise alignment test).
        """
        results = await self._gather_diagnostic_results(
            diagnostic_type=GpuDiagnostic.diagnostic_type,
            agents=remaining_agents,
            timeout_seconds=self._default_timeout_seconds,
        )

        bad_node_ids, remaining = self._partition_results(
            results=results, agents=remaining_agents,
            diagnostic_type=GpuDiagnostic.diagnostic_type,
        )

        passed_results = {
            nid: results[nid] for nid in remaining
        }
        hash_outliers = self._find_gpu_hash_outlier_nodes(passed_results)
        if hash_outliers:
            bad_node_ids.extend(hash_outliers)
            remaining = {
                nid: agent for nid, agent in remaining.items()
                if nid not in set(hash_outliers)
            }

        return bad_node_ids, remaining

    @staticmethod
    def _find_gpu_hash_outlier_nodes(
        results: dict[str, DiagnosticResult],
    ) -> list[str]:
        """Compare compute hashes across nodes, return nodes with minority hashes.

        For each GPU index, the most common hash is considered correct
        (majority vote). Nodes whose hash differs are outliers.

        If no clear majority exists (majority_count <= total/2), we assume
        non-determinism rather than SDC and skip that GPU index.
        """
        node_gpu_hashes: dict[str, dict[str, str]] = {}
        for node_id, result in results.items():
            if result.metadata and "compute_hashes" in result.metadata:
                node_gpu_hashes[node_id] = result.metadata["compute_hashes"]

        if len(node_gpu_hashes) < 2:
            return []

        all_gpu_indices: set[str] = set()
        for hashes in node_gpu_hashes.values():
            all_gpu_indices.update(hashes.keys())

        outlier_nodes: set[str] = set()

        for gpu_idx in sorted(all_gpu_indices):
            hash_to_nodes: dict[str, list[str]] = {}
            for node_id, hashes in node_gpu_hashes.items():
                h = hashes.get(gpu_idx, "")
                if h:
                    hash_to_nodes.setdefault(h, []).append(node_id)

            if len(hash_to_nodes) <= 1:
                continue

            total = sum(len(nodes) for nodes in hash_to_nodes.values())
            majority_hash = max(hash_to_nodes, key=lambda h: len(hash_to_nodes[h]))
            majority_count = len(hash_to_nodes[majority_hash])

            if majority_count <= total / 2:
                logger.warning(
                    "gpu_hash_no_majority gpu_idx=%s hash_distribution=%s — "
                    "possible non-determinism, skipping this GPU index",
                    gpu_idx,
                    {h[:12]: len(n) for h, n in hash_to_nodes.items()},
                )
                continue

            for h, nodes in hash_to_nodes.items():
                if h != majority_hash:
                    logger.info(
                        "gpu_hash_outlier gpu_idx=%s nodes=%s "
                        "outlier_hash=%s majority_hash=%s (%d/%d nodes agree)",
                        gpu_idx, nodes, h[:12], majority_hash[:12],
                        majority_count, total,
                    )
                    outlier_nodes.update(nodes)

        return sorted(outlier_nodes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _RPC_TIMEOUT_BUFFER_SECONDS = 30

    async def _call_agent_diagnostic(
        self,
        agent: NodeAgentProtocol,
        node_id: str,
        diagnostic_type: str,
        timeout_seconds: int,
    ) -> DiagnosticResult:
        try:
            return await asyncio.wait_for(
                agent.run_diagnostic(
                    diagnostic_type, timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds + self._RPC_TIMEOUT_BUFFER_SECONDS,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_agent_rpc_timeout node=%s type=%s timeout=%d",
                node_id, diagnostic_type, timeout_seconds,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type, node_id=node_id,
                details=f"agent RPC timed out after {timeout_seconds + self._RPC_TIMEOUT_BUFFER_SECONDS}s",
            )

        except UnknownDiagnosticError:
            logger.error(
                "diagnostic_type_not_registered node=%s type=%s — "
                "this is a pipeline configuration error, treating as fail",
                node_id, diagnostic_type,
                exc_info=True,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type, node_id=node_id,
                details=f"config error: diagnostic type '{diagnostic_type}' not registered on node",
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
