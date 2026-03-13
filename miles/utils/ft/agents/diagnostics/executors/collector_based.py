from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.types import SampleEvaluator
from miles.utils.ft.utils.diagnostic_types import DiagnosticResult

logger = logging.getLogger(__name__)


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
        logger.info(
            "diagnostics: running collector-based diagnostic: type=%s, node=%s, timeout=%s",
            self.diagnostic_type,
            node_id,
            timeout_seconds,
        )
        try:
            output = await asyncio.wait_for(
                self._collector.collect(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostics: collector timed out: type=%s, node=%s, timeout=%s",
                self.diagnostic_type,
                node_id,
                timeout_seconds,
            )
            return self._fail(node_id, f"collector timed out after {timeout_seconds}s")
        except Exception:
            logger.error(
                "diagnostics: collector raised exception: type=%s, node=%s",
                self.diagnostic_type,
                node_id,
                exc_info=True,
            )
            raise

        passed, reason = self._evaluator(node_id, output.metrics)
        logger.debug(
            "diagnostics: evaluator result: type=%s, node=%s, passed=%s, reason=%s",
            self.diagnostic_type,
            node_id,
            passed,
            reason,
        )

        if not passed:
            return self._fail(node_id, reason)
        return self._pass(node_id, reason)
