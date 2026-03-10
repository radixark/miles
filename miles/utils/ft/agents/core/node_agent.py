from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import (
    DIAGNOSTIC_TIMEOUT_SECONDS,
    AgentMetadataProvider,
    NodeAgentProtocol,
    NodeExecutorProtocol,
)
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.dispatcher import NodeDiagnosticDispatcher
from miles.utils.ft.agents.metrics.metric_collection_loop import MetricCollectionLoop
from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.agents.types import DiagnosticResult

logger = logging.getLogger(__name__)


class FtNodeAgent(NodeAgentProtocol):
    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector] | None = None,
        collect_interval_seconds: float | None = None,
        diagnostics: list[NodeExecutorProtocol] | None = None,
        metadata_provider: AgentMetadataProvider | None = None,
    ) -> None:
        self._node_id = node_id
        self._metadata_provider = metadata_provider
        self._metadata: dict[str, str] = {}
        self._dispatcher = NodeDiagnosticDispatcher(node_id=node_id, diagnostics=diagnostics)

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

    @property
    def metadata(self) -> dict[str, str]:
        return self._metadata

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    async def start(self) -> None:
        if self._metadata_provider is not None:
            self._metadata = self._metadata_provider.get_metadata()

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
        return await self._dispatcher.run_diagnostic(
            diagnostic_type=diagnostic_type,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
