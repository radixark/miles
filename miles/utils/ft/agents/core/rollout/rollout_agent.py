from __future__ import annotations

import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import EngineHealthChecker
from miles.utils.ft.agents.core.rollout.health_checker import CellEntry, RolloutHealthChecker
from miles.utils.ft.agents.core.rollout.metrics_exporter import RolloutMetricsExporter

logger = logging.getLogger(__name__)


class FtRolloutAgent:
    def __init__(
        self,
        *,
        cell_ids: list[str],
        get_engines: Callable[[str], list[object]],
        health_checker: EngineHealthChecker,
        check_interval: float = 10.0,
    ) -> None:
        self._metrics_exporter = RolloutMetricsExporter()
        self._health_checker = RolloutHealthChecker(
            cells=[
                CellEntry(
                    cell_id=cid,
                    get_engines=lambda _c=cid: get_engines(_c),
                )
                for cid in cell_ids
            ],
            engine_health_fn=health_checker,
            report_fn=self._metrics_exporter.update,
            check_interval=check_interval,
        )

        logger.info(
            "rollout: agent started: address=%s, cells=%s",
            self.address,
            self._health_checker.cell_ids,
        )

    @property
    def address(self) -> str:
        return self._metrics_exporter.address

    async def shutdown(self) -> None:
        logger.info("rollout: agent shutting down")
        await self._health_checker.shutdown()
        self._metrics_exporter.shutdown()
        logger.info("rollout: agent shutdown complete")

    def pause(self) -> None:
        logger.info("rollout: agent pausing health checks")
        self._health_checker.pause()

    def resume(self) -> None:
        logger.info("rollout: agent resuming health checks")
        self._health_checker.resume()
