from __future__ import annotations

import asyncio

import httpx
import pytest

from miles.utils.ft_utils.control_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus
from miles.utils.ft_utils.control_server.registry import _CellRegistry
from miles.utils.ft_utils.control_server.server import _create_control_app


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
    def __init__(self, return_value: object) -> None:
        self._return_value = return_value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        self.calls.append((args, kwargs))
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._return_value)
        return future


class MockInferenceController:
    """Plain object, not a Ray actor: the controller lives in the driver process."""

    def __init__(
        self,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
    ) -> None:
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self._is_suspended = is_suspended
        self.stopped_cells: list[str] = []
        self.started_cells: list[str] = []

    def get_cell_phase(self, cell_id: str) -> str:
        return self._phase

    def get_cell_conditions(self, cell_id: str) -> list[dict[str, str | None]]:
        return self._conditions

    def get_cell_is_suspended(self, cell_id: str) -> bool:
        return self._is_suspended

    async def stop_cell(self, cell_id: str) -> None:
        self.stopped_cells.append(cell_id)

    async def start_cell(self, cell_id: str) -> None:
        self.started_cells.append(cell_id)


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
        from miles.utils.ft_utils.control_server.models import CellCondition, CellStatus

        return CellStatus(
            phase=self._phase,
            conditions=[CellCondition(**c) for c in self._conditions],
        )


def make_mock_group(cells: list[MockRayTrainCell]) -> object:
    from miles.ray.train.group import RayTrainGroup

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
    app = _create_control_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")
