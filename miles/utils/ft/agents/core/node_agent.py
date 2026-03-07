from __future__ import annotations

import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.runner import DiagnosticRunner
from miles.utils.ft.agents.utils.metric_collection_loop import MetricCollectionLoop
from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS, DiagnosticProtocol, NodeAgentProtocol

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
        self._runner = DiagnosticRunner(node_id=node_id, diagnostics=diagnostics)

        prepared_collectors = list(collectors or [])
        if collect_interval_seconds is not None:
            for collector in prepared_collectors:
                collector.collect_interval = collect_interval_seconds

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

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult:
        return await self._runner.run_diagnostic(
            diagnostic_type=diagnostic_type,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

