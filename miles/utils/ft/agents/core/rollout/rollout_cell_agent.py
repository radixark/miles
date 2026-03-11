from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import EngineHealthChecker
from miles.utils.ft.agents.core.rollout.health_checker import CellHealthResult, RolloutCellHealthChecker

logger = logging.getLogger(__name__)


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
        self._health_checker = RolloutCellHealthChecker(
            cell_id=cell_id,
            engine_health_fn=health_checker,
            timeout=health_check_timeout,
        )
        self._node_ids: set[str] = set()

    @property
    def cell_id(self) -> str:
        return self._cell_id

    # --- Health ---

    async def check_health(self) -> CellHealthResult:
        """Run health check on all engines concurrently, return aggregated result."""
        return await self._health_checker.check_health(engines=self._engines)

    def is_healthy(self) -> bool:
        """Based on the most recent check_health result. Returns False if never checked."""
        return self._health_checker.is_healthy()

    # --- Node tracking ---

    def get_node_ids(self) -> set[str]:
        return set(self._node_ids)

    def get_engine_count(self) -> int:
        return len(self._engines)

    def get_alive_engine_count(self) -> int:
        last_result = self._health_checker.last_result
        if last_result is None:
            return 0
        return last_result.alive_engines

    def update_engines(self, engines: list[object]) -> None:
        """Replace engine handles after recovery. Invalidates last result until next check."""
        self._engines = list(engines)
        self._health_checker.invalidate()

    def set_node_ids(self, node_ids: set[str]) -> None:
        """Called by FtRolloutAgent after startup."""
        self._node_ids = set(node_ids)
