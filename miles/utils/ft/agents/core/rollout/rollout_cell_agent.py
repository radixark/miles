from __future__ import annotations

import logging
from collections.abc import Callable

from miles.utils.ft.adapters.types import EngineHealthChecker
from miles.utils.ft.agents.core.rollout.health_checker import RolloutCellHealthChecker

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
        get_engines: Callable[[], list[object]],
        health_checker: EngineHealthChecker,
        health_check_timeout: float = 10.0,
    ) -> None:
        self._cell_id = cell_id
        self._get_engines = get_engines
        self._health_checker = RolloutCellHealthChecker(
            cell_id=cell_id,
            engine_health_fn=health_checker,
            timeout=health_check_timeout,
        )

    @property
    def cell_id(self) -> str:
        return self._cell_id

    # --- Health ---

    async def check_health(self) -> bool:
        """Probe the cell's engines and return True if healthy."""
        return await self._health_checker.check_health(engines=self._get_engines())

