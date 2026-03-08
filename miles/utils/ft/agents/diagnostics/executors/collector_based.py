from __future__ import annotations

from collections.abc import Callable

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.models.fault import NodeFault
from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS
from miles.utils.ft.protocols.metrics import MetricQueryProtocol


class CollectorBasedNodeExecutor(BaseNodeExecutor):
    diagnostic_type: str

    def __init__(
        self,
        diagnostic_type: str,
        collector: BaseCollector,
        check_fn: Callable[[MetricQueryProtocol], list[NodeFault]],
    ) -> None:
        self.diagnostic_type = diagnostic_type
        self._collector = collector
        self._check_fn = check_fn

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult:
        output = await self._collector.collect()
        store = InMemoryMetricStore()
        store.ingest_samples(target_id=node_id, samples=output.metrics)
        faults = self._check_fn(store)
        if faults:
            return self._fail(node_id, "; ".join(f.reason for f in faults))
        return self._pass(node_id, "all checks passed")
