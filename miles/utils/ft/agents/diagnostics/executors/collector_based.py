from __future__ import annotations

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.types import SampleEvaluator
from miles.utils.ft.utils.diagnostic_types import DiagnosticResult


class CollectorBasedNodeExecutor(BaseNodeExecutor):
    diagnostic_type: str

    def __init__(
        self,
        diagnostic_type: str,
        collector: BaseCollector,
        evaluator: SampleEvaluator,
    ) -> None:
        self.diagnostic_type = diagnostic_type
        self._collector = collector
        self._evaluator = evaluator

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult:
        output = await self._collector.collect()

        passed, reason = self._evaluator(node_id, output.metrics)

        if not passed:
            return self._fail(node_id, reason)
        return self._pass(node_id, reason)
