from __future__ import annotations

from miles.utils.ft.agents.core.rollout.rollout_cell_agent import RolloutCellAgent


async def _noop_health_checker(engine: object) -> None:
    pass


class MockRolloutCellAgent(RolloutCellAgent):
    """Overrides _check_single_engine to avoid real health check calls."""

    def __init__(self, *, cell_id: str, engine_alive: list[bool]) -> None:
        super().__init__(
            cell_id=cell_id,
            engines=list(range(len(engine_alive))),
            health_checker=_noop_health_checker,
        )
        self._engine_alive = list(engine_alive)

    async def _check_single_engine(self, *, engine: object, index: int) -> bool:
        return self._engine_alive[index]

    async def stop(self) -> None:
        pass

    async def start(self) -> int:
        return sum(self._engine_alive)
