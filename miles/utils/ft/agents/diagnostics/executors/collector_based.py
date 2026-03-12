from __future__ import annotations

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import ActionType, MetricStore


class CollectorBasedNodeExecutor(BaseNodeExecutor):
    diagnostic_type: str

    def __init__(
        self,
        diagnostic_type: str,
        collector: BaseCollector,
        detector: BaseFaultDetector,
    ) -> None:
        self.diagnostic_type = diagnostic_type
        self._collector = collector
        self._detector = detector

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult:
        output = await self._collector.collect()
        store = InMemoryMetricStore()
        store.ingest_samples(target_id=node_id, samples=output.metrics)

        ctx = DetectorContext(metric_store=MetricStore(time_series_store=store, mini_wandb=MiniWandb()))
        decision = self._detector.evaluate(ctx)

        if decision.action != ActionType.NONE:
            return self._fail(node_id, decision.reason)
        return self._pass(node_id, decision.reason)
