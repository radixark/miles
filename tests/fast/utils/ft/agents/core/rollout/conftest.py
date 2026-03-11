from __future__ import annotations

from miles.utils.ft.agents.core.rollout.health_checker import RolloutCellHealthChecker
from miles.utils.ft.agents.core.rollout.rollout_cell_agent import RolloutCellAgent


async def _noop_health_checker(engine: object) -> None:
    pass


class MockEngine:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive


async def mock_health_checker(engine: object) -> None:
    if not engine.alive:  # type: ignore[attr-defined]
        raise ConnectionError("engine dead")


class _MockHealthChecker(RolloutCellHealthChecker):
    """Override _probe_engine so the first engine's liveness is configurable."""

    def __init__(self, *, cell_id: str, healthy: bool) -> None:
        super().__init__(cell_id=cell_id, engine_health_fn=_noop_health_checker, timeout=10.0)
        self.healthy = healthy

    async def _probe_engine(self, *, engine: object) -> bool:
        return self.healthy


class MockRolloutCellAgent(RolloutCellAgent):
    """Uses _MockHealthChecker to avoid real health check calls."""

    def __init__(self, *, cell_id: str, healthy: bool = True) -> None:
        super().__init__(
            cell_id=cell_id,
            get_engines=lambda: [object()],
            health_checker=_noop_health_checker,
        )
        self._health_checker = _MockHealthChecker(cell_id=cell_id, healthy=healthy)
