from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtomHealthResult:
    atom_id: str
    total_engines: int
    alive_engines: int
    dead_engine_indices: list[int]
    is_healthy: bool
    checked_at: float


class RolloutAtomAgent:
    """Manages a group of engines forming a fault domain.

    E.g. EP72 = 9 engines across 9 nodes. Any single engine death makes
    the entire group unhealthy, requiring a full stop/start cycle.

    This is an internal object held by FtRolloutAgent (not a Ray actor).
    """

    def __init__(
        self,
        *,
        atom_id: str,
        engines: list[object],
        health_check_timeout: float = 10.0,
    ) -> None:
        self._atom_id = atom_id
        self._engines = list(engines)
        self._health_check_timeout = health_check_timeout
        self._node_ids: set[str] = set()
        self._last_result: AtomHealthResult | None = None

    @property
    def atom_id(self) -> str:
        return self._atom_id

    # --- Health ---

    async def check_health(self) -> AtomHealthResult:
        """Run health check on all engines, return aggregated result."""
        alive_count = 0
        dead_indices: list[int] = []

        for i, engine in enumerate(self._engines):
            ok = await self._check_single_engine(engine=engine, index=i)
            if ok:
                alive_count += 1
            else:
                dead_indices.append(i)

        result = AtomHealthResult(
            atom_id=self._atom_id,
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
        """Check whether a single engine is alive.

        Default implementation calls engine.health_check.remote() (Ray actor interface).
        Tests override via MockRolloutAtomAgent subclass.
        """
        try:
            await engine.health_check.remote()  # type: ignore[attr-defined]
            return True
        except Exception:
            logger.debug(
                "engine_health_check_failed atom_id=%s index=%d",
                self._atom_id, index, exc_info=True,
            )
            return False

    # --- Lifecycle ---

    async def stop(self) -> None:
        """Stop all engines in this atom."""
        raise NotImplementedError("Depends on rollout architecture")

    async def start(self) -> int:
        """Rebuild engines for this atom. Returns alive engine count."""
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

    def set_node_ids(self, node_ids: set[str]) -> None:
        """Called by FtRolloutAgent after startup or rebuild."""
        self._node_ids = set(node_ids)
