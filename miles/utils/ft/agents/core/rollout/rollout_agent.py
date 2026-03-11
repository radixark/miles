from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import EngineHealthChecker
from miles.utils.ft.agents.core.rollout.health_checker import RolloutHealthChecker
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
        self._health_checker = self._build_health_checker(
            rollout_manager, health_checker=health_checker,
        )

        self._check_interval = check_interval
        self._paused = False
        self._metrics_exporter = RolloutMetricsExporter()

        self._health_loop_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            "ft_rollout_agent_started address=%s cells=%s",
            self.address, self._health_checker.cell_ids,
        )

    @staticmethod
    def _build_health_checker(
        rollout_manager: object,
        *,
        health_checker: EngineHealthChecker,
    ) -> RolloutHealthChecker:
        checker = RolloutHealthChecker(engine_health_fn=health_checker)
        checker.add_cell(
            cell_id="default",
            get_engines=lambda: list(rollout_manager.all_rollout_engines),  # type: ignore[attr-defined]
        )
        return checker

    @property
    def address(self) -> str:
        return self._metrics_exporter.address

    # --- Lifecycle ---

    async def shutdown(self) -> None:
        if self._health_loop_task is not None:
            self._health_loop_task.cancel()
            try:
                await self._health_loop_task
            except asyncio.CancelledError:
                pass
        self._metrics_exporter.shutdown()
        logger.info("ft_rollout_agent_shutdown")

    # --- Health check loop ---

    async def _health_check_loop(self) -> None:
        while True:
            if not self._paused:
                await asyncio.gather(
                    *(
                        self._check_one_cell(cell_id)
                        for cell_id in self._health_checker.cell_ids
                    )
                )
            await asyncio.sleep(self._check_interval)

    async def _check_one_cell(self, cell_id: str) -> None:
        try:
            is_healthy = await self._health_checker.check_cell(cell_id)
            self._metrics_exporter.update(cell_id=cell_id, is_healthy=is_healthy)
        except Exception:
            logger.warning(
                "health_check_failed cell_id=%s", cell_id, exc_info=True
            )

    def pause(self) -> None:
        self._paused = True
        logger.info("ft_rollout_agent_paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("ft_rollout_agent_resumed")
