from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.ft.adapters.types import EngineHealthChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CellEntry:
    cell_id: str
    get_engines: Callable[[], list[object]]


class RolloutHealthChecker:
    """Periodically probes all rollout cells and reports results via *report_fn*.

    Drives the async health-check loop.  Callers inject a
    ``report_fn(*, cell_id, is_healthy)`` callback (typically the metrics
    exporter) to decouple checking from reporting.
    """

    def __init__(
        self,
        *,
        cells: dict[str, Callable[[], list[object]]],
        engine_health_fn: EngineHealthChecker,
        report_fn: Callable[..., None],
        check_interval: float = 30.0,
        timeout: float = 30.0,
    ) -> None:
        self._engine_health_fn = engine_health_fn
        self._timeout = timeout
        self._report_fn = report_fn
        self._check_interval = check_interval
        self._paused = False

        self._cells: dict[str, _CellEntry] = {
            cell_id: _CellEntry(cell_id=cell_id, get_engines=get_engines)
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
                    *(self._check_one_cell(entry) for entry in self._cells.values())
                )
            await asyncio.sleep(self._check_interval)

    async def _check_one_cell(self, entry: _CellEntry) -> None:
        try:
            is_healthy = await _probe_cell(
                engines=entry.get_engines(),
                engine_health_fn=self._engine_health_fn,
                cell_id=entry.cell_id,
                timeout=self._timeout,
            )
            self._report_fn(cell_id=entry.cell_id, is_healthy=is_healthy)
        except Exception:
            logger.warning(
                "health_check_failed cell_id=%s", entry.cell_id, exc_info=True,
            )


async def _probe_cell(
    *,
    engines: list[object],
    engine_health_fn: EngineHealthChecker,
    cell_id: str,
    timeout: float,
) -> bool:
    """Probe engines[0] of a cell and return True if healthy.

    All engines in a cell form a single TP group; only engines[0] exposes
    an HTTP health endpoint, so we probe that one and consider the entire
    cell alive or dead based on its response.
    """
    if not engines:
        raise ValueError("engines must not be empty")

    try:
        await asyncio.wait_for(engine_health_fn(engines[0]), timeout=timeout)
        return True
    except Exception:
        logger.info("engine_health_check_failed cell_id=%s", cell_id, exc_info=True)
        return False
