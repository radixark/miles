from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS, ClusterExecutorProtocol, NodeAgentProtocol
from miles.utils.ft.agents.types import DiagnosticPipelineResult
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol

logger = logging.getLogger(__name__)


class DiagnosticOrchestrator(DiagnosticOrchestratorProtocol):
    """Layered progressive diagnostic pipeline.

    Runs registered diagnostic executors in order. Each executor receives
    the full agent set independently. The pipeline stops on the first
    executor that returns non-empty bad_node_ids.
    """

    def __init__(
        self,
        node_agents: dict[str, NodeAgentProtocol],
        pipeline: list[ClusterExecutorProtocol] | None = None,
        default_timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        pipeline_timeout_seconds: int = 900,
    ) -> None:
        self._node_agents = node_agents
        self._pipeline = pipeline or []
        self._default_timeout_seconds = default_timeout_seconds
        self._pipeline_timeout_seconds = pipeline_timeout_seconds

    async def run_diagnostic_pipeline(
        self,
        pre_executors: list[ClusterExecutorProtocol] | None = None,
    ) -> DiagnosticPipelineResult:
        logger.info(
            "diagnostic_pipeline_start pipeline_steps=%d pre_executors=%d",
            len(self._pipeline),
            len(pre_executors) if pre_executors else 0,
        )

        try:
            return await asyncio.wait_for(
                self._run_diagnostic_pipeline_inner(pre_executors=pre_executors),
                timeout=self._pipeline_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_pipeline_timeout timeout=%d",
                self._pipeline_timeout_seconds,
            )
            return DiagnosticPipelineResult(
                bad_node_ids=[],
                reason=f"diagnostic pipeline timed out after {self._pipeline_timeout_seconds}s",
            )

    async def _run_diagnostic_pipeline_inner(
        self,
        pre_executors: list[ClusterExecutorProtocol] | None = None,
    ) -> DiagnosticPipelineResult:
        all_executors = (pre_executors or []) + self._pipeline

        if not all_executors:
            logger.info("diagnostic_pipeline_empty — no diagnostics configured")
            return DiagnosticPipelineResult(
                bad_node_ids=[],
                reason="no diagnostics configured (empty pipeline)",
            )

        for executor in all_executors:
            try:
                bad_node_ids = await executor.execute(
                    node_agents=dict(self._node_agents),
                    timeout_seconds=self._default_timeout_seconds,
                )
            except Exception:
                logger.error(
                    "diagnostic_step_failed executor=%s",
                    type(executor).__name__,
                    exc_info=True,
                )
                continue

            if bad_node_ids:
                logger.info(
                    "diagnostic_step_found_bad bad_nodes=%s",
                    bad_node_ids,
                )
                return DiagnosticPipelineResult(
                    bad_node_ids=sorted(bad_node_ids),
                    reason=f"diagnostic failed on nodes: {bad_node_ids}",
                )

        logger.info("diagnostic_pipeline_all_passed")
        return DiagnosticPipelineResult(
            bad_node_ids=[],
            reason="all diagnostics passed — no bad nodes found",
        )
