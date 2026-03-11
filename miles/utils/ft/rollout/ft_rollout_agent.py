from __future__ import annotations

import asyncio
import logging

from prometheus_client import CollectorRegistry, Gauge

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.rollout.cell_agent import CellHealthResult, RolloutCellAgent
from miles.utils.ft.rollout.metrics_server import MetricsServer

logger = logging.getLogger(__name__)


class FtRolloutAgent:
    def __init__(
        self,
        *,
        cells: dict[str, RolloutCellAgent],
        check_interval: float = 10.0,
        metrics_port: int = 0,
    ) -> None:
        self._cells = cells
        self._check_interval = check_interval
        self._paused = False
        self._health_loop_task: asyncio.Task[None] | None = None

        self._registry = CollectorRegistry()
        self._engine_alive = Gauge(
            "rollout_engine_alive",
            "1=alive, 0=dead",
            labelnames=["cell_id", "engine_index"],
            registry=self._registry,
        )
        self._cell_alive = Gauge(
            "rollout_cell_alive",
            "1=all engines alive, 0=any dead",
            labelnames=["cell_id"],
            registry=self._registry,
        )

        self._metrics_server = MetricsServer(
            registry=self._registry,
            port=metrics_port,
        )

    @property
    def address(self) -> str:
        return self._metrics_server.address

    # --- Lifecycle ---

    async def start(self) -> None:
        await self._metrics_server.start()
        self._health_loop_task = asyncio.create_task(self._health_check_loop())
        logger.info("ft_rollout_agent_started address=%s cells=%d", self.address, len(self._cells))

    async def shutdown(self) -> None:
        if self._health_loop_task is not None:
            self._health_loop_task.cancel()
            try:
                await self._health_loop_task
            except asyncio.CancelledError:
                pass
        await self._metrics_server.shutdown()
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
            self._update_metrics(result)
        except Exception:
            logger.warning(
                "health_check_failed cell_id=%s", cell.cell_id, exc_info=True
            )

    def _update_metrics(self, result: CellHealthResult) -> None:
        self._cell_alive.labels(cell_id=result.cell_id).set(
            1.0 if result.is_healthy else 0.0
        )
        dead_set = set(result.dead_engine_indices)
        for i in range(result.total_engines):
            self._engine_alive.labels(
                cell_id=result.cell_id, engine_index=str(i)
            ).set(0.0 if i in dead_set else 1.0)

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

    # --- Control plane: cell level (called by RayRolloutActuator) ---

    async def stop_cell(self, cell_id: str) -> None:
        await self._cells[cell_id].stop()

    async def start_cell(self, cell_id: str) -> int:
        return await self._cells[cell_id].start()

    async def get_cell_status(self, cell_id: str) -> JobStatus:
        return JobStatus.RUNNING if self._cells[cell_id].is_healthy() else JobStatus.FAILED

    # --- Public API for M12 integration ---

    def get_cell_ids(self) -> list[str]:
        return list(self._cells.keys())

    def get_cell_agent(self, cell_id: str) -> RolloutCellAgent:
        return self._cells[cell_id]

    async def register_with_controller(self, controller_handle: object) -> None:
        await controller_handle.add_scrape_target.remote(  # type: ignore[attr-defined]
            target_id="rollout-ft-agent",
            address=self.address,
        )
        logger.info("registered_with_controller address=%s", self.address)
