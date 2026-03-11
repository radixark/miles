from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import EngineHealthChecker, JobStatus
from miles.utils.ft.agents.core.rollout.rollout_cell_agent import RolloutCellAgent
from miles.utils.ft.agents.core.rollout.metrics_exporter import RolloutMetricsExporter

logger = logging.getLogger(__name__)


class FtRolloutAgent:
    def __init__(
        self,
        rollout_manager: object | None = None,
        *,
        cells: dict[str, RolloutCellAgent] | None = None,
        health_checker: EngineHealthChecker | None = None,
        check_interval: float = 10.0,
    ) -> None:
        if rollout_manager is not None:
            if health_checker is None:
                raise ValueError("health_checker is required when rollout_manager is provided")
            self._rollout_manager = rollout_manager
            self._cells = self._build_cells(rollout_manager, health_checker=health_checker)
        elif cells is not None:
            self._rollout_manager = None
            self._cells = cells
        else:
            raise ValueError("Either rollout_manager or cells must be provided")

        self._check_interval = check_interval
        self._paused = False
        self._metrics_exporter = RolloutMetricsExporter()

        self._health_loop_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            "ft_rollout_agent_started address=%s cells=%d",
            self.address, len(self._cells),
        )

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

    # --- Control plane: subsystem level (Level 2) ---

    async def stop_all(self) -> None:
        for cell in self._cells.values():
            await cell.stop()

    async def rebuild(self) -> str:
        raise NotImplementedError("Full rebuild depends on rollout architecture")

    async def get_status(self) -> JobStatus:
        all_healthy = all(cell.is_healthy() for cell in self._cells.values())
        return JobStatus.RUNNING if all_healthy else JobStatus.FAILED

    # --- Control plane: cell level (routes to RolloutManager when available) ---

    async def stop_cell(self, cell_id: str) -> None:
        if self._rollout_manager is not None:
            await self._rollout_manager.stop_cell(cell_id)  # type: ignore[attr-defined]
        else:
            await self._cells[cell_id].stop()

    async def start_cell(self, cell_id: str) -> int:
        if self._rollout_manager is not None:
            return await self._rollout_manager.start_cell(cell_id)  # type: ignore[attr-defined]
        return await self._cells[cell_id].start()

    async def get_cell_status(self, cell_id: str) -> JobStatus:
        if self._rollout_manager is not None:
            return await self._rollout_manager.get_cell_status(cell_id)  # type: ignore[attr-defined]
        return JobStatus.RUNNING if self._cells[cell_id].is_healthy() else JobStatus.FAILED

    # --- Public API for M12 integration ---

    def update_cell_engines(self, cell_id: str, engines: list[object]) -> None:
        self._cells[cell_id].update_engines(engines)

    def get_cell_ids(self) -> list[str]:
        return list(self._cells.keys())

    def get_cell_agent(self, cell_id: str) -> RolloutCellAgent:
        return self._cells[cell_id]
