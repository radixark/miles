from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import EngineHealthChecker
from miles.utils.ft.agents.core.rollout.health_checker import CellEntry, RolloutHealthChecker
from miles.utils.ft.agents.core.rollout.metrics_exporter import RolloutMetricsExporter

logger = logging.getLogger(__name__)


class FtRolloutAgent:
    def __init__(
        self,
        rollout_manager: object,
        *,
        health_checker: EngineHealthChecker,
        check_interval: float = 10.0,
    ) -> None:
        self._metrics_exporter = RolloutMetricsExporter()
        self._health_checker = RolloutHealthChecker(
            # TODO: will have many cells
            cells=[
                CellEntry(
                    cell_id="default",
                    get_engines=lambda: list(rollout_manager.all_rollout_engines),  # type: ignore[attr-defined]
                ),
            ],
            engine_health_fn=health_checker,
            report_fn=self._metrics_exporter.update,
            check_interval=check_interval,
        )

        logger.info(
            "ft_rollout_agent_started address=%s cells=%s",
            self.address, self._health_checker.cell_ids,
        )

    @property
    def address(self) -> str:
        return self._metrics_exporter.address

    async def shutdown(self) -> None:
        await self._health_checker.shutdown()
        self._metrics_exporter.shutdown()
        logger.info("ft_rollout_agent_shutdown")

    def pause(self) -> None:
        self._health_checker.pause()

    def resume(self) -> None:
        self._health_checker.resume()
