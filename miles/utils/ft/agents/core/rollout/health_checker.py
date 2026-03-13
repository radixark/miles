from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.ft.adapters.types import EngineHealthChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellEntry:
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
        cells: list[CellEntry],
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

        self._cells: dict[str, CellEntry] = {entry.cell_id: entry for entry in cells}

        logger.info(
            "rollout: health checker initialized: num_cells=%d, cell_ids=%s, check_interval=%s, timeout=%s",
            len(self._cells), list(self._cells.keys()), check_interval, timeout,
        )

        # Accepted product decision: rollout-agent construction is only
        # supported from an already-running event loop. We intentionally do
        # not offer a sync-safe/lazy-start variant for callers outside async
        # contexts; future audits should treat that as a non-goal unless
        # product requirements change.
        self._task = asyncio.create_task(self._loop())

    @property
    def cell_ids(self) -> list[str]:
        return list(self._cells.keys())

    def pause(self) -> None:
        self._paused = True
        logger.info("rollout: health checker paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("rollout: health checker resumed")

    async def shutdown(self) -> None:
        logger.info("rollout: health checker shutting down")
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        logger.info("rollout: health checker shutdown complete")

    # --- Private ---

    async def _loop(self) -> None:
        while True:
            if not self._paused:
                logger.debug("rollout: health check loop iteration: num_cells=%d", len(self._cells))
                await asyncio.gather(*(self._check_one_cell(entry) for entry in self._cells.values()))
            else:
                logger.debug("rollout: health check loop skipped, paused")
            await asyncio.sleep(self._check_interval)

    async def _check_one_cell(self, entry: CellEntry) -> None:
        is_healthy = False
        try:
            is_healthy = await _probe_cell(
                engines=entry.get_engines(),
                engine_health_fn=self._engine_health_fn,
                cell_id=entry.cell_id,
                timeout=self._timeout,
            )
        except Exception:
            logger.warning(
                "rollout: health check failed: cell_id=%s",
                entry.cell_id,
                exc_info=True,
            )
        logger.debug("rollout: health check result: cell_id=%s, is_healthy=%s", entry.cell_id, is_healthy)
        self._report_fn(cell_id=entry.cell_id, is_healthy=is_healthy)


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

    # Miles will make the engine object none when it is killed (stopped)
    lead_engine = engines[0]
    if lead_engine is None:
        logger.debug("rollout: lead engine is None, cell unhealthy: cell_id=%s", cell_id)
        return False

    try:
        await asyncio.wait_for(engine_health_fn(lead_engine), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning("rollout: engine health check timed out: cell_id=%s, timeout=%s", cell_id, timeout)
        return False
    except Exception:
        logger.info("rollout: engine health check failed: cell_id=%s", cell_id, exc_info=True)
        return False
