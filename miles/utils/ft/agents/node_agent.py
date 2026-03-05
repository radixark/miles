from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.prometheus_exporter import PrometheusExporter
from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models import DiagnosticResult, UnknownDiagnosticError

logger = logging.getLogger(__name__)


class FtNodeAgent:
    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector] | None = None,
        collect_interval_seconds: float | None = None,
        diagnostics: list[BaseDiagnostic] | None = None,
    ) -> None:
        self._node_id = node_id
        self._collectors = collectors or []
        self._stopped = False

        if collect_interval_seconds is not None:
            for collector in self._collectors:
                collector.collect_interval = collect_interval_seconds

        self._diagnostics: dict[str, BaseDiagnostic] = {
            d.diagnostic_type: d for d in (diagnostics or [])
        }

        self._exporter = PrometheusExporter()
        self._collector_tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    async def start(self) -> None:
        if self._stopped or self._collector_tasks:
            return

        loop = asyncio.get_running_loop()
        for collector in self._collectors:
            task = loop.create_task(self._run_single_collector(collector))
            self._collector_tasks.append(task)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        for task in self._collector_tasks:
            task.cancel()
        await asyncio.gather(*self._collector_tasks, return_exceptions=True)
        self._collector_tasks.clear()

        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.warning(
                    "Collector %s.close() failed on node %s",
                    type(collector).__name__,
                    self._node_id,
                    exc_info=True,
                )

        self._exporter.shutdown()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def set_diagnostic(self, diagnostic: BaseDiagnostic) -> None:
        """Register (or replace) a diagnostic by its diagnostic_type."""
        self._diagnostics[diagnostic.diagnostic_type] = diagnostic

    def remove_diagnostic(self, diagnostic_type: str) -> None:
        """Remove a diagnostic by type. No-op if not present."""
        self._diagnostics.pop(diagnostic_type, None)

    async def run_diagnostic(
        self, diagnostic_type: str, timeout_seconds: int = 120,
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
                    node_id=self._node_id, timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds + 5,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_timeout type=%s node_id=%s timeout=%d",
                diagnostic_type, self._node_id, timeout_seconds,
            )
            return DiagnosticResult(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                passed=False,
                details=f"diagnostic timed out after {timeout_seconds}s",
            )
        except Exception:
            logger.warning(
                "diagnostic_error type=%s node_id=%s",
                diagnostic_type, self._node_id,
                exc_info=True,
            )
            return DiagnosticResult(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                passed=False,
                details="diagnostic raised exception",
            )

    # ------------------------------------------------------------------
    # Per-collector background task
    # ------------------------------------------------------------------

    async def _run_single_collector(self, collector: BaseCollector) -> None:
        collector_name = type(collector).__name__
        while True:
            try:
                result = await collector.collect()
                self._exporter.update_metrics(result.metrics)
            except Exception:
                logger.warning(
                    "Collector %s failed on node %s",
                    collector_name,
                    self._node_id,
                    exc_info=True,
                )

            await asyncio.sleep(collector.collect_interval)
