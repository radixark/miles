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


class RolloutCellAgent:
    """Manages a group of engines forming a fault domain.

    E.g. EP72 = 9 engines across 9 nodes. Any single engine death makes
    the entire group unhealthy, requiring a full stop/start cycle.

    This is an internal object held by FtRolloutAgent (not a Ray actor).
    """

    def __init__(
        self,
        *,
        cell_id: str,
        engines: list[object],
        health_checker: EngineHealthChecker,
        health_check_timeout: float = 10.0,
    ) -> None:
        self._cell_id = cell_id
        self._engines = list(engines)
        self._health_checker = health_checker
        self._health_check_timeout = health_check_timeout
        self._node_ids: set[str] = set()
        self._last_result: CellHealthResult | None = None

    @property
    def cell_id(self) -> str:
        return self._cell_id

    # --- Health ---

    async def check_health(self) -> CellHealthResult:
        """Run health check on all engines concurrently, return aggregated result."""
        results = await asyncio.gather(
            *(self._check_single_engine(engine=engine, index=i)
              for i, engine in enumerate(self._engines))
        )

        alive_count = sum(results)
        dead_indices = tuple(i for i, ok in enumerate(results) if not ok)

        result = CellHealthResult(
            cell_id=self._cell_id,
            total_engines=len(self._engines),
            alive_engines=alive_count,
            dead_engine_indices=dead_indices,
            is_healthy=(alive_count == len(self._engines)),
            checked_at=time.time(),
        )
        self._last_result = result
        return result

    def is_healthy(self) -> bool:
        """Based on the most recent check_health result. Returns False if never checked."""
        if self._last_result is None:
            return False
        return self._last_result.is_healthy

    async def _check_single_engine(self, *, engine: object, index: int) -> bool:
        """Check whether a single engine is alive via the injected health_checker."""
        try:
            await asyncio.wait_for(
                self._health_checker(engine),
                timeout=self._health_check_timeout,
            )
            return True
        except Exception:
            logger.debug(
                "engine_health_check_failed cell_id=%s index=%d",
                self._cell_id, index, exc_info=True,
            )
            return False

    # --- Lifecycle ---

    async def stop(self) -> None:
        """Stop all engines in this cell."""
        raise NotImplementedError("Depends on rollout architecture")

    async def start(self) -> int:
        """Rebuild engines for this cell. Returns alive engine count."""
        raise NotImplementedError("Depends on rollout architecture")

    # --- Node tracking ---

    def get_node_ids(self) -> set[str]:
        return set(self._node_ids)

    def get_engine_count(self) -> int:
        return len(self._engines)

    def get_alive_engine_count(self) -> int:
        if self._last_result is None:
            return 0
        return self._last_result.alive_engines

    def update_engines(self, engines: list[object]) -> None:
        """Replace engine handles after recovery. Invalidates last result until next check."""
        self._engines = list(engines)
        self._last_result = None

    def set_node_ids(self, node_ids: set[str]) -> None:
        """Called by FtRolloutAgent after startup or rebuild."""
        self._node_ids = set(node_ids)
