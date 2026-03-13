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
        logger.info(
            "node_agent: initialized: node_id=%s, num_collectors=%d, num_diagnostics=%d, collect_interval_override=%s",
            node_id,
            len(prepared_collectors),
            len(diagnostics or []),
            collect_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, str]:
        if self._metadata_provider is not None:
            return self._metadata_provider.get_metadata()
        logger.debug("node_agent: no metadata provider, returning empty metadata: node_id=%s", self._node_id)
        return {}

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    def wait_for_exporter_ready(self, timeout_seconds: float = 5.0) -> None:
        self._exporter.wait_until_ready(timeout_seconds=timeout_seconds)

    async def start(self) -> None:
        logger.info("node_agent: starting: node_id=%s", self._node_id)
        await self._collection_loop.start()

    async def stop(self) -> None:
        logger.info("node_agent: stopping: node_id=%s", self._node_id)
        await self._collection_loop.stop()
        self._exporter.shutdown()
        logger.info("node_agent: stopped: node_id=%s", self._node_id)

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult:
        logger.info(
            "node_agent: running diagnostic: node_id=%s, type=%s, timeout=%d",
            self._node_id,
            diagnostic_type,
            timeout_seconds,
        )
        result = await self._dispatcher.run_diagnostic(
            diagnostic_type=diagnostic_type,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        logger.info(
            "node_agent: diagnostic complete: node_id=%s, type=%s, passed=%s",
            self._node_id,
            diagnostic_type,
            result.passed,
        )
        return result
