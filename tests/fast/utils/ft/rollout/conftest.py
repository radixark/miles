from __future__ import annotations

from miles.utils.ft.rollout.atom_agent import RolloutAtomAgent


class MockRolloutAtomAgent(RolloutAtomAgent):
    """Overrides _check_single_engine to avoid Ray dependency."""

    def __init__(self, *, atom_id: str, engine_alive: list[bool]) -> None:
        super().__init__(atom_id=atom_id, engines=list(range(len(engine_alive))))
        self._engine_alive = list(engine_alive)

    async def _check_single_engine(self, *, engine: object, index: int) -> bool:
        return self._engine_alive[index]

    async def stop(self) -> None:
        pass

    async def start(self) -> int:
        return sum(self._engine_alive)
