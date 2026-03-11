from __future__ import annotations

import asyncio
import logging

from prometheus_client import CollectorRegistry, Gauge

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.rollout.atom_agent import AtomHealthResult, RolloutAtomAgent
from miles.utils.ft.rollout.metrics_server import MetricsServer

logger = logging.getLogger(__name__)


class FtRolloutAgent:
    def __init__(
        self,
        *,
        atoms: dict[str, RolloutAtomAgent],
        check_interval: float = 10.0,
        metrics_port: int = 0,
    ) -> None:
        self._atoms = atoms
        self._check_interval = check_interval
        self._paused = False
        self._health_loop_task: asyncio.Task[None] | None = None

        self._registry = CollectorRegistry()
        self._engine_alive = Gauge(
            "rollout_engine_alive",
            "1=alive, 0=dead",
            labelnames=["atom_id", "engine_index"],
            registry=self._registry,
        )
        self._atom_alive = Gauge(
            "rollout_atom_alive",
            "1=all engines alive, 0=any dead",
            labelnames=["atom_id"],
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
        logger.info("ft_rollout_agent_started address=%s atoms=%d", self.address, len(self._atoms))

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
                for atom in self._atoms.values():
                    try:
                        result = await atom.check_health()
                        self._update_metrics(result)
                    except Exception:
                        logger.warning(
                            "health_check_failed atom_id=%s", atom.atom_id, exc_info=True
                        )
            await asyncio.sleep(self._check_interval)

    def _update_metrics(self, result: AtomHealthResult) -> None:
        self._atom_alive.labels(atom_id=result.atom_id).set(
            1.0 if result.is_healthy else 0.0
        )
        alive_set = set(range(result.total_engines)) - set(result.dead_engine_indices)
        for i in range(result.total_engines):
            self._engine_alive.labels(
                atom_id=result.atom_id, engine_index=str(i)
            ).set(1.0 if i in alive_set else 0.0)

    def pause(self) -> None:
        self._paused = True
        logger.info("ft_rollout_agent_paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("ft_rollout_agent_resumed")

    # --- Control plane: subsystem level (Level 2) ---

    async def stop_all(self) -> None:
        for atom in self._atoms.values():
            await atom.stop()

    async def rebuild(self) -> str:
        raise NotImplementedError("Full rebuild depends on rollout architecture")

    async def get_status(self) -> JobStatus:
        all_healthy = all(atom.is_healthy() for atom in self._atoms.values())
        return JobStatus.RUNNING if all_healthy else JobStatus.FAILED

    # --- Control plane: atom level (called by RayRolloutActuator) ---

    async def stop_atom(self, atom_id: str) -> None:
        await self._atoms[atom_id].stop()

    async def start_atom(self, atom_id: str) -> int:
        return await self._atoms[atom_id].start()

    async def get_atom_status(self, atom_id: str) -> JobStatus:
        return JobStatus.RUNNING if self._atoms[atom_id].is_healthy() else JobStatus.FAILED

    # --- Public API for M12 integration ---

    def get_atom_ids(self) -> list[str]:
        return list(self._atoms.keys())

    def get_atom_agent(self, atom_id: str) -> RolloutAtomAgent:
        return self._atoms[atom_id]

    async def register_with_controller(self, controller_handle: object) -> None:
        await controller_handle.add_scrape_target.remote(  # type: ignore[attr-defined]
            target_id="rollout-ft-agent",
            address=self.address,
        )
        logger.info("registered_with_controller address=%s", self.address)
