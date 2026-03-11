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
    """Override _check_single_engine with configurable engine_alive."""

    def __init__(self, *, cell_id: str, engine_alive: list[bool]) -> None:
        super().__init__(cell_id=cell_id, engine_health_fn=_noop_health_checker, timeout=10.0)
        self.engine_alive = list(engine_alive)

    async def _check_single_engine(self, *, engine: object, index: int) -> bool:
        return self.engine_alive[index]


class MockRolloutCellAgent(RolloutCellAgent):
    """Uses _MockHealthChecker to avoid real health check calls."""

    def __init__(self, *, cell_id: str, engine_alive: list[bool]) -> None:
        super().__init__(
            cell_id=cell_id,
            engines=list(range(len(engine_alive))),
            health_checker=_noop_health_checker,
        )
        mock = _MockHealthChecker(cell_id=cell_id, engine_alive=engine_alive)
        self._health_checker = mock
