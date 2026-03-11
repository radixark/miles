from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.ft.adapters.types import EngineHealthChecker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate health checker for all cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CellEntry:
    get_engines: Callable[[], list[object]]
    checker: RolloutCellHealthChecker


class RolloutHealthChecker:
    """Manages health checking across all rollout cells.

    Owns one :class:`RolloutCellHealthChecker` per cell and knows how to
    retrieve the engine list for each cell via a callback.
    """

    def __init__(
        self,
        *,
        engine_health_fn: EngineHealthChecker,
        timeout: float = 10.0,
    ) -> None:
        self._engine_health_fn = engine_health_fn
        self._timeout = timeout
        self._cells: dict[str, _CellEntry] = {}

    def add_cell(
        self,
        cell_id: str,
        get_engines: Callable[[], list[object]],
    ) -> None:
        self._cells[cell_id] = _CellEntry(
            get_engines=get_engines,
            checker=RolloutCellHealthChecker(
                cell_id=cell_id,
                engine_health_fn=self._engine_health_fn,
                timeout=self._timeout,
            ),
        )

    @property
    def cell_ids(self) -> list[str]:
        return list(self._cells.keys())

    async def check_cell(self, cell_id: str) -> bool:
        entry = self._cells[cell_id]
        return await entry.checker.check_health(engines=entry.get_engines())


# ---------------------------------------------------------------------------
# Per-cell health checker
# ---------------------------------------------------------------------------


class RolloutCellHealthChecker:
    """Encapsulates health-checking logic for a rollout cell's engines.

    All engines in a cell form a single TP group; only engines[0] exposes
    an HTTP health endpoint, so we probe that one and consider the entire
    cell alive or dead based on its response.

    Caches the most recent boolean result so that ``is_healthy`` can be
    queried cheaply between checks.
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
        self._last_healthy: bool | None = None

    async def check_health(self, *, engines: list[object]) -> bool:
        """Probe engines[0] and return True if healthy, False otherwise."""
        if not engines:
            raise ValueError("engines must not be empty")

        healthy = await self._probe_engine(engine=engines[0])
        self._last_healthy = healthy
        return healthy

    def is_healthy(self) -> bool:
        """Based on the most recent check_health result. Returns False if never checked."""
        return self._last_healthy is True

    def invalidate(self) -> None:
        """Clear cached result, forcing is_healthy() to return False until next check."""
        self._last_healthy = None

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
