from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from miles.utils.ft.adapters.types import EngineHealthChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellHealthResult:
    cell_id: str
    total_engines: int
    alive_engines: int
    dead_engine_indices: tuple[int, ...]
    is_healthy: bool
    checked_at: float


class RolloutCellHealthChecker:
    """Encapsulates health-checking logic for a rollout cell's engines.

    Stateless w.r.t. engines: the caller passes the engine list on each
    ``check_health`` call.  The checker caches the most recent result so
    that ``is_healthy`` can be queried cheaply between checks.
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
        self._last_result: CellHealthResult | None = None

    async def check_health(self, *, engines: list[object]) -> CellHealthResult:
        """Run health check against engines[0] and treat the whole cell accordingly.

        All engines in a cell form a single TP group; only engines[0] exposes
        an HTTP health endpoint, so we probe that one and consider the entire
        cell alive or dead based on its response.
        """
        if not engines:
            raise ValueError("engines must not be empty")

        healthy = await self._check_single_engine(engine=engines[0], index=0)

        total = len(engines)
        alive_count = total if healthy else 0
        dead_indices = () if healthy else tuple(range(total))

        result = CellHealthResult(
            cell_id=self._cell_id,
            total_engines=total,
            alive_engines=alive_count,
            dead_engine_indices=dead_indices,
            is_healthy=healthy,
            checked_at=time.time(),
        )
        self._last_result = result
        return result

    def is_healthy(self) -> bool:
        """Based on the most recent check_health result. Returns False if never checked."""
        if self._last_result is None:
            return False
        return self._last_result.is_healthy

    def invalidate(self) -> None:
        """Clear cached result, forcing is_healthy() to return False until next check."""
        self._last_result = None

    @property
    def last_result(self) -> CellHealthResult | None:
        return self._last_result

    async def _check_single_engine(self, *, engine: object, index: int) -> bool:
        """Check whether a single engine is alive via the injected health function."""
        try:
            await asyncio.wait_for(
                self._engine_health_fn(engine),
                timeout=self._timeout,
            )
            return True
        except Exception:
            logger.debug(
                "engine_health_check_failed cell_id=%s index=%d",
                self._cell_id, index, exc_info=True,
            )
            return False
