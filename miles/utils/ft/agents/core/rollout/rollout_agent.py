from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import EngineHealthChecker
from miles.utils.ft.agents.core.rollout.rollout_cell_agent import RolloutCellAgent
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
        self._cells = self._build_cells(rollout_manager, health_checker=health_checker)

        self._check_interval = check_interval
        self._paused = False
        self._metrics_exporter = RolloutMetricsExporter()

        self._health_loop_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            "ft_rollout_agent_started address=%s cells=%d",
            self.address, len(self._cells),
        )

    # TODO this is surely incorrect. need to know real cells when rollout.py is refactored
    @staticmethod
    def _build_cells(
        rollout_manager: object,
        *,
        health_checker: EngineHealthChecker,
    ) -> dict[str, RolloutCellAgent]:
        return {"default": RolloutCellAgent(
            cell_id="default",
            engines=list(rollout_manager.all_rollout_engines),  # type: ignore[attr-defined]
            health_checker=health_checker,
        )}

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
                    *(self._check_one_cell(cell) for cell in self._cells.values())
                )
            await asyncio.sleep(self._check_interval)

    async def _check_one_cell(self, cell: RolloutCellAgent) -> None:
        try:
            result = await cell.check_health()
            self._metrics_exporter.update(result)
        except Exception:
            logger.warning(
                "health_check_failed cell_id=%s", cell.cell_id, exc_info=True
            )

    def pause(self) -> None:
        self._paused = True
        logger.info("ft_rollout_agent_paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("ft_rollout_agent_resumed")

