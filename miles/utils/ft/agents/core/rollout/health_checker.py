from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.ft.adapters.types import EngineHealthChecker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate health checker — owns the loop, delegates to per-cell checkers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CellEntry:
    get_engines: Callable[[], list[object]]
    checker: RolloutCellHealthChecker


class RolloutHealthChecker:
    """Periodically probes all rollout cells and reports results via *report_fn*.

    Owns one :class:`RolloutCellHealthChecker` per cell and drives the
    async health-check loop.  Callers inject a ``report_fn(*, cell_id, is_healthy)``
    callback (typically the metrics exporter) to decouple checking from reporting.
    """

    def __init__(
        self,
        *,
        cells: dict[str, Callable[[], list[object]]],
        engine_health_fn: EngineHealthChecker,
        report_fn: Callable[..., None],
        check_interval: float = 10.0,
        timeout: float = 10.0,
    ) -> None:
        self._report_fn = report_fn
        self._check_interval = check_interval
        self._paused = False

        self._cells: dict[str, _CellEntry] = {
            cell_id: _CellEntry(
                get_engines=get_engines,
                checker=RolloutCellHealthChecker(
                    cell_id=cell_id,
                    engine_health_fn=engine_health_fn,
                    timeout=timeout,
                ),
            )
            for cell_id, get_engines in cells.items()
        }

        self._task = asyncio.create_task(self._loop())

    @property
    def cell_ids(self) -> list[str]:
        return list(self._cells.keys())

    def pause(self) -> None:
        self._paused = True
        logger.info("rollout_health_checker_paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("rollout_health_checker_resumed")

    async def shutdown(self) -> None:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        logger.info("rollout_health_checker_shutdown")

    # --- Private ---

    async def _loop(self) -> None:
        while True:
            if not self._paused:
                await asyncio.gather(
                    *(self._check_one_cell(cid) for cid in self._cells)
                )
            await asyncio.sleep(self._check_interval)

    async def _check_one_cell(self, cell_id: str) -> None:
        try:
            entry = self._cells[cell_id]
            is_healthy = await entry.checker.check_health(
                engines=entry.get_engines(),
            )
            self._report_fn(cell_id=cell_id, is_healthy=is_healthy)
        except Exception:
            logger.warning(
                "health_check_failed cell_id=%s", cell_id, exc_info=True,
            )


# ---------------------------------------------------------------------------
# Per-cell health checker
# ---------------------------------------------------------------------------


class RolloutCellHealthChecker:
    """Probes a single rollout cell (TP group) by checking engines[0].

    All engines in a cell form a single TP group; only engines[0] exposes
    an HTTP health endpoint, so we probe that one and consider the entire
    cell alive or dead based on its response.
    """

    def __init__(
        self,
        *,
        cell_id: str,
        engine_health_fn: EngineHealthChecker,
        timeout: float = 10.0,
    ) -> None:
        self._cell_id = cell_id
        self._engine_health_fn = engine_health_fn
        self._timeout = timeout

    async def check_health(self, *, engines: list[object]) -> bool:
        """Probe engines[0] and return True if healthy, False otherwise."""
        if not engines:
            raise ValueError("engines must not be empty")

        return await self._probe_engine(engine=engines[0])

    async def _probe_engine(self, *, engine: object) -> bool:
        try:
            await asyncio.wait_for(
                self._engine_health_fn(engine),
                timeout=self._timeout,
            )
            return True
        except Exception:
            logger.info(
                "engine_health_check_failed cell_id=%s",
                self._cell_id, exc_info=True,
            )
            return False
