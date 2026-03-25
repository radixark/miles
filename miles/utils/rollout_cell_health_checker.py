from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.prometheus_utils import get_prometheus

logger = logging.getLogger(__name__)

_METRIC_NAME = "miles_rollout_cell_alive"


@dataclass(frozen=True)
class CellEntry:
    cell_id: str
    get_engines: Callable[[], list[object]]


class RolloutCellHealthChecker:
    """Async health checker that periodically probes rollout cells and reports a Prometheus gauge.

    Each cell's lead engine is probed via ``engine.health_generate.remote()``.
    The gauge ``miles_rollout_cell_alive`` is set to 1.0 (healthy) or 0.0 (unhealthy).
    """

    @classmethod
    def maybe_create(cls, servers: dict, args: object) -> RolloutCellHealthChecker | None:
        """Create and start a checker if prometheus is enabled and there are cells to monitor."""
        if not args.use_prometheus:
            return None

        cells = [
            CellEntry(
                cell_id=f"{srv_name}-{group.worker_type}",
                get_engines=lambda g=group: g.engines,
            )
            for srv_name, srv in servers.items()
            for group in srv.server_groups
        ]
        if not cells:
            return None

        checker = cls(
            cells=cells,
            session_id=args.session_id,
            check_interval=args.rollout_health_check_interval,
            timeout=args.rollout_health_check_timeout,
        )
        checker.start()
        return checker

    def __init__(
        self,
        *,
        cells: list[CellEntry],
        session_id: str,
        check_interval: float = 30.0,
        timeout: float = 30.0,
    ) -> None:
        self._cells = {entry.cell_id: entry for entry in cells}
        self._session_id = session_id
        self._check_interval = check_interval
        self._timeout = timeout
        self._paused = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the health check loop. Must be called from an async context."""
        if self._task is not None:
            return
        self._task = asyncio.ensure_future(self._loop())
        logger.info("rollout cell health checker started: num_cells=%d", len(self._cells))

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    async def shutdown(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            if self._paused:
                for entry in self._cells.values():
                    self._report(cell_id=entry.cell_id, is_healthy=None)
            else:
                await asyncio.gather(*(self._check_one_cell(e) for e in self._cells.values()))

            await asyncio.sleep(self._check_interval)

    async def _check_one_cell(self, entry: CellEntry) -> None:
        is_healthy = False
        try:
            is_healthy = await _probe_cell(engines=entry.get_engines(), timeout=self._timeout)
        except Exception:
            logger.warning("Health probe failed for cell %s", entry.cell_id, exc_info=True)

        self._report(cell_id=entry.cell_id, is_healthy=is_healthy)

    def _report(self, *, cell_id: str, is_healthy: bool | None) -> None:
        handle = get_prometheus()
        if handle is None:
            return

        try:
            handle.set_gauge.remote(
                _METRIC_NAME,
                -1.0 if is_healthy is None else (1.0 if is_healthy else 0.0),
                extra_labels={"session_id": self._session_id, "cell_id": cell_id},
            )
        except Exception:
            logger.warning("Failed to report cell health for %s", cell_id, exc_info=True)


async def _probe_cell(*, engines: list[object], timeout: float) -> bool:
    if not engines:
        return False

    lead_engine = engines[0]
    if lead_engine is None:
        return False

    await asyncio.wait_for(lead_engine.health_generate.remote(), timeout=timeout)  # type: ignore[union-attr]
    return True
