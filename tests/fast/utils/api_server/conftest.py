from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable

import httpx
import pytest

from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.ft_utils.api_server.server import _create_api_app
from miles.utils.workers.worker_provider.base import CellInfo


class MockHandle:
    def __init__(
        self,
        cell_id: str,
        cell_type: str,
        cell_key: str = "0",
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
        suspend_error: Exception | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.cell_key = cell_key
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self._is_suspended = is_suspended
        self._suspend_error = suspend_error
        self._resume_error = resume_error
        self.suspend_calls: int = 0
        self.resume_calls: int = 0

    async def get_cell(self) -> Cell:
        return Cell(
            metadata=CellMetadata(
                name=self.cell_id,
                labels={"miles.io/cell-type": self.cell_type, "miles.io/cell-index": self.cell_key},
            ),
            spec=CellSpec(suspend=self._is_suspended),
            status=CellStatus(
                phase=self._phase,
                conditions=[CellCondition(**c) for c in self._conditions],
            ),
        )

    async def suspend(self) -> None:
        if self._suspend_error:
            raise self._suspend_error
        self.suspend_calls += 1
        self._is_suspended = True
        self._phase = "Suspended"
        self._conditions = [
            {"type": "Allocated", "status": "False"},
            {"type": "Healthy", "status": "False"},
        ]

    async def resume(self) -> None:
        if self._resume_error:
            raise self._resume_error
        self.resume_calls += 1
        self._is_suspended = False
        self._phase = "Running"
        self._conditions = [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]


class MockRemoteCall:
    def __init__(self, return_value: object, effect: Callable[..., None] | None = None) -> None:
        self._return_value = return_value
        self._effect = effect
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        self.calls.append((args, kwargs))
        if self._effect is not None:
            self._effect(*args, **kwargs)
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._return_value)
        return future


class MockInferenceController:
    def __init__(self, statuses: dict[str, CellStatus] | None = None) -> None:
        self._statuses = dict(statuses or {})
        self.status_calls: int = 0

    def get_cell_statuses(self) -> dict[str, CellStatus]:
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
        return MockRemoteCall(dict(self._summaries), effect=lambda **kwargs: self.cell_info_calls.append(kwargs))

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


def make_cell_summaries(*cell_ids: str, suspended: bool = False) -> dict[str, CellInfo]:
    return {
        cell_id: CellInfo(
            cell_id=cell_id,
            pool_id=cell_id.rsplit("-", 1)[0],
            alive=not suspended,
            worker_names=[] if suspended else [f"{cell_id}-0"],
            workers_hash="pseudo-hash-0",
            meta={"model_id": "default"},
        )
        for cell_id in cell_ids
    }


class MockRayTrainCell:
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_stopped: bool = False,
    ) -> None:
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self._is_stopped = is_stopped

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def conditions(self) -> list[dict[str, str | None]]:
        return self._conditions

    @property
    def is_stopped(self) -> bool:
        return self._is_stopped

    def cell_status(self) -> CellStatus:
        from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus

        return CellStatus(
            phase=self._phase,
            conditions=[CellCondition(**c) for c in self._conditions],
        )


def make_mock_group(cells: list[MockRayTrainCell]) -> object:
    from miles.ray.train.group import TrainerController

    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    group._alive_cell_ids = frozenset()
    return group


@pytest.fixture
def registry() -> _CellRegistry:
    return _CellRegistry()


@pytest.fixture
def async_client(registry: _CellRegistry) -> httpx.AsyncClient:
    app = _create_api_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")
