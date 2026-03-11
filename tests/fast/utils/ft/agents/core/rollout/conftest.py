from __future__ import annotations

from miles.utils.ft.agents.core.rollout.health_checker import RolloutHealthChecker


class MockEngine:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive


async def mock_health_checker(engine: object) -> None:
    if not engine.alive:  # type: ignore[attr-defined]
        raise ConnectionError("engine dead")


class MockRolloutHealthChecker(RolloutHealthChecker):
    """RolloutHealthChecker with configurable per-cell results.

    For cells registered via ``add_cell``, health checking works normally
    (delegates to the real ``RolloutCellHealthChecker``).

    For cells registered via ``set_result``, ``check_cell`` returns the
    preconfigured boolean directly.
    """

    def __init__(self, cell_results: dict[str, bool] | None = None) -> None:
        async def _noop(engine: object) -> None:
            pass

        super().__init__(engine_health_fn=_noop)
        self._fixed_results: dict[str, bool] = {}

        if cell_results is not None:
            for cell_id, healthy in cell_results.items():
                self.set_result(cell_id, healthy)

    def set_result(self, cell_id: str, healthy: bool) -> None:
        self._fixed_results[cell_id] = healthy
        if cell_id not in self._cells:
            self.add_cell(cell_id=cell_id, get_engines=lambda: [object()])

    async def check_cell(self, cell_id: str) -> bool:
        if cell_id in self._fixed_results:
            return self._fixed_results[cell_id]
        return await super().check_cell(cell_id)
