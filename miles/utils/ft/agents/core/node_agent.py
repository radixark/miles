from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.utils.metric_collection_loop import MetricCollectionLoop
from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.models.diagnostics import DiagnosticResult, UnknownDiagnosticError
from miles.utils.ft.protocols.agents import DiagnosticProtocol, NodeAgentProtocol

logger = logging.getLogger(__name__)


class FtNodeAgent(NodeAgentProtocol):
    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector] | None = None,
        collect_interval_seconds: float | None = None,
        diagnostics: list[DiagnosticProtocol] | None = None,
    ) -> None:
        self._node_id = node_id

        prepared_collectors = list(collectors or [])
        if collect_interval_seconds is not None:
            for collector in prepared_collectors:
                collector.collect_interval = collect_interval_seconds

        self._diagnostics: dict[str, DiagnosticProtocol] = {d.diagnostic_type: d for d in (diagnostics or [])}

        self._exporter = PrometheusExporter()
        self._collection_loop = MetricCollectionLoop(
            node_id=node_id,
            collectors=prepared_collectors,
            exporter=self._exporter,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    async def start(self) -> None:
        await self._collection_loop.start()

    async def stop(self) -> None:
        await self._collection_loop.stop()
        self._exporter.shutdown()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        diagnostic = self._diagnostics.get(diagnostic_type)
        if diagnostic is None:
            raise UnknownDiagnosticError(
                f"node {self._node_id}: unknown diagnostic type '{diagnostic_type}', "
                f"registered types: {sorted(self._diagnostics.keys())}"
            )

        try:
            return await asyncio.wait_for(
                diagnostic.run(
                    node_id=self._node_id,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                ),
                timeout=timeout_seconds + 5,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_timeout type=%s node_id=%s timeout=%d",
                diagnostic_type,
                self._node_id,
                timeout_seconds,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details=f"diagnostic timed out after {timeout_seconds}s",
            )
        except Exception:
            logger.warning(
                "diagnostic_error type=%s node_id=%s",
                diagnostic_type,
                self._node_id,
                exc_info=True,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details="diagnostic raised exception",
            )

