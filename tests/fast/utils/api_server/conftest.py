from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable

import httpx
import pytest

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellSpec, CellStatus
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.ft_utils.api_server.server import _create_api_app
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class MockCellState:
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
        suspend_error: Exception | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self.phase = phase
        self.conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self.is_suspended = is_suspended
        self.suspend_error = suspend_error
        self.resume_error = resume_error
        self.suspend_calls: int = 0
        self.resume_calls: int = 0


class MockHandler(_CellHandler):
    def __init__(self, cell_type: str) -> None:
        self._cell_type = cell_type
        self.cells: dict[str, MockCellState] = {}
        self.injected: list[tuple[str, FailureMode, int]] = []
        self.supports_inject_fault = False
        self.inject_fault_error: Exception | None = None

    @property
    def cell_type(self) -> str:
        return self._cell_type

    def add(self, cell_id: str = "0", **overrides) -> MockCellState:
        state = MockCellState(**overrides)
        self.cells[cell_id] = state
        return state

    async def list_cell_ids(self) -> list[str]:
        return list(self.cells)

    async def list_cells(self) -> list[Cell]:
        cell_ids = await self.list_cell_ids()
        return list(await asyncio.gather(*(self.get_cell(cell_id) for cell_id in cell_ids)))

    async def get_cell(self, cell_id: str) -> Cell:
        state = self.cells[cell_id]
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=state.is_suspended),
            status=CellStatus(
                phase=state.phase,
                conditions=[CellCondition(**c) for c in state.conditions],
                workers_hash="pseudo-hash-0",
            ),
        )

    async def suspend(self, cell_id: str) -> None:
        state = self.cells[cell_id]
        if state.suspend_error:
            raise state.suspend_error
        state.suspend_calls += 1
        state.is_suspended = True
        state.phase = "Suspended"
        state.conditions = [
            {"type": "Allocated", "status": "False"},
            {"type": "Healthy", "status": "False"},
        ]

    async def resume(self, cell_id: str) -> None:
        state = self.cells[cell_id]
        if state.resume_error:
            raise state.resume_error
        state.resume_calls += 1
        state.is_suspended = False
        state.phase = "Running"
        state.conditions = [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        if not self.supports_inject_fault:
            raise NotImplementedError(f"{type(self).__name__} does not support fault injection")
        if self.inject_fault_error is not None:
            raise self.inject_fault_error
        self.injected.append((cell_id, mode, sub_index))


class MockGatedHandler(MockHandler):
    def __init__(self, cell_type: str, *, gate: asyncio.Event) -> None:
        super().__init__(cell_type)
        self._gate = gate

    async def list_cells(self) -> list[Cell]:
        await self._gate.wait()
        return await super().list_cells()


class MockGateOpeningHandler(MockHandler):
    def __init__(self, cell_type: str, *, gate: asyncio.Event) -> None:
        super().__init__(cell_type)
        self._gate = gate

    async def list_cells(self) -> list[Cell]:
        self._gate.set()
        return await super().list_cells()


class MockRemoteCall:
    def __init__(
        self,
        return_value: object,
        effect: Callable[..., None] | None = None,
        factory: Callable[..., object] | None = None,
    ) -> None:
        self._return_value = return_value
        self._effect = effect
        self._factory = factory
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        self.calls.append((args, kwargs))
        if self._effect is not None:
            self._effect(*args, **kwargs)
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._factory(*args, **kwargs) if self._factory is not None else self._return_value)
        return future


class MockInferenceController:
    def __init__(self, statuses: dict[str, CellStatus] | None = None) -> None:
        self._statuses = dict(statuses or {})
        self.status_calls: int = 0

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        self.status_calls += 1
        return dict(self._statuses)

    def observe_cell(self, cell_id: str, status: CellStatus) -> None:
        self._statuses[cell_id] = status


class MockWorkerManager:
    def __init__(self, summaries: dict[str, CellInfo] | None = None) -> None:
        self._summaries = dict(summaries or {})
        self.stopped_cells: list[list[str]] = []
        self.started_cells: list[list[str]] = []
        self.cell_info_calls: list[dict[str, object]] = []

    @property
    def get_cell_infos(self) -> MockRemoteCall:
        def _filtered(**kwargs: object) -> dict[str, CellInfo]:
            pool_ids = kwargs.get("pool_ids")
            if pool_ids is None:
                return dict(self._summaries)
            return {cell_id: info for cell_id, info in self._summaries.items() if info.pool_id in pool_ids}

        return MockRemoteCall(
            None,
            effect=lambda **kwargs: self.cell_info_calls.append(kwargs),
            factory=_filtered,
        )

    @property
    def stop_cells(self) -> MockRemoteCall:
        return MockRemoteCall(None, effect=lambda ids: self._record(self.stopped_cells, ids, suspended=True))

    @property
    def start_cells(self) -> MockRemoteCall:
        return MockRemoteCall(None, effect=lambda ids: self._record(self.started_cells, ids, suspended=False))

    def _record(self, log: list[list[str]], cell_ids: list[str], *, suspended: bool) -> None:
        log.append(list(cell_ids))
        for cell_id in cell_ids:
            previous = self._summaries[cell_id]
            self._summaries[cell_id] = dataclasses.replace(previous, alive=not suspended)


def make_cell_summaries(
    *cell_ids: str, suspended: bool = False, workers_hash: str = "pseudo-hash-0"
) -> dict[str, CellInfo]:
    return {
        cell_id: CellInfo(
            cell_id=cell_id,
            pool_id=cell_id.rsplit("-", 1)[0],
            alive=not suspended,
            worker_names=[] if suspended else [f"{cell_id}-0"],
            workers_hash=workers_hash,
            meta={"model_id": "default"},
        )
        for cell_id in cell_ids
    }


class MockTrainerCell:
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        workers_hash: str = "pseudo-hash-0",
    ) -> None:
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self.workers_hash = workers_hash

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def conditions(self) -> list[dict[str, str | None]]:
        return self._conditions

    def cell_status(self) -> CellStatus:
        from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus

        return CellStatus(
            phase=self._phase,
            conditions=[CellCondition(**c) for c in self._conditions],
            workers_hash=self.workers_hash,
        )


def make_mock_controller(cells: list[MockTrainerCell], *, pool_id: str = "trainer-engine-actor") -> object:
    from miles.ray.train.group import TrainerController

    group = object.__new__(TrainerController)
    for cell_index, cell in enumerate(cells):
        cell.cell_index = cell_index
        cell.cell_id = f"{pool_id}-{cell_index}"
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group._pool_id = pool_id
    group._indep_dp_quorum_id = 0
    return group


@pytest.fixture
def actor_handler() -> MockHandler:
    return MockHandler("actor")


@pytest.fixture
def rollout_handler() -> MockHandler:
    return MockHandler("rollout")


@pytest.fixture
def registry(actor_handler: MockHandler, rollout_handler: MockHandler) -> _CellRegistry:
    return _CellRegistry([actor_handler, rollout_handler])


@pytest.fixture
def async_client(registry: _CellRegistry) -> httpx.AsyncClient:
    app = _create_api_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")
