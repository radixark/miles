from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.control_server.handles import _ActorCellHandle, _CellHandle, _RolloutCellHandle
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import MockRayTrainCell, MockRolloutManager, make_mock_group


class TestActorCellHandle:
    def test_cell_id_and_type(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        assert handle.cell_id == "actor-0"
        assert handle.cell_type == "actor"

    @pytest.mark.asyncio
    async def test_get_cell_returns_full_cell_structure(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.model_dump() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-index": "0",
                },
            },
            "spec": {"suspend": False},
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Allocated",
                        "status": "True",
                        "reason": None,
                        "message": None,
                        "lastTransitionTime": None,
                    },
                    {"type": "Healthy", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_get_cell_suspended(self) -> None:
        group = make_mock_group(
            [
                MockRayTrainCell(
                    phase="Suspended",
                    conditions=[
                        {"type": "Allocated", "status": "False"},
                        {"type": "Healthy", "status": "False"},
                    ],
                    is_stopped=True,
                )
            ]
        )
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.stop_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=2)
        await handle.suspend()
        group.stop_cell.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_resume_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.start_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=1)
        await handle.resume()
        group.start_cell.assert_called_once_with(1)


class TestRolloutCellHandle:
    @pytest.mark.asyncio
    async def test_get_cell_delegates_to_manager(self) -> None:
        manager = MockRolloutManager()
        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        cell = await handle.get_cell()

        assert cell.metadata.name == "rollout-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"
        assert cell.spec.suspend is False

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_manager(self) -> None:
        manager = MockRolloutManager()
        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        await handle.suspend()
        assert manager.stop_cell.calls == [((0,), {})]

    @pytest.mark.asyncio
    async def test_resume_delegates_to_manager(self) -> None:
        manager = MockRolloutManager()
        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        await handle.resume()
        assert manager.start_cell.calls == [((0,), {})]

    def test_cell_type_is_rollout(self) -> None:
        handle = _RolloutCellHandle(rollout_manager=object(), cell_index=0)
        assert handle.cell_type == "rollout"
        assert handle.cell_id == "rollout-0"


class _FakeRemoteMethod:
    def __init__(self) -> None:
        self.remote_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> None:
        self.remote_calls.append((args, kwargs))


class _FakeActor:
    def __init__(self) -> None:
        self.inject_fault = _FakeRemoteMethod()


class _FakeInjectCell:
    def __init__(self, *, is_alive: bool = True, num_actors: int = 2) -> None:
        self._is_alive = is_alive
        self._actor = _FakeActor()
        self._num_actors = num_actors

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    def _get_actor_handles(self) -> list[_FakeActor]:
        return [self._actor for _ in range(self._num_actors)]


def _make_inject_group(cell: _FakeInjectCell) -> object:
    group = object.__new__(RayTrainGroup)
    group._cells = [cell]
    return group


class _ConcreteCellHandle(_CellHandle):
    @property
    def cell_type(self) -> str:
        return "fake"

    @property
    def cell_index(self) -> int:
        return 0

    async def get_cell(self) -> object:
        raise NotImplementedError

    async def suspend(self) -> None:
        raise NotImplementedError

    async def resume(self) -> None:
        raise NotImplementedError


class TestActorCellHandleInjectFault:
    @pytest.mark.asyncio
    async def test_inject_fault_calls_actor_with_mode_value(self) -> None:
        """inject_fault forwards mode.value to the selected actor's remote handle."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=1)

        assert cell._actor.inject_fault.remote_calls == [(("sigkill",), {})]

    @pytest.mark.asyncio
    async def test_inject_fault_raises_when_cell_not_alive(self) -> None:
        """inject_fault raises RuntimeError when the target cell is not alive."""
        cell = _FakeInjectCell(is_alive=False, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        with pytest.raises(RuntimeError, match="not alive"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=0)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_out_of_range(self) -> None:
        """inject_fault raises IndexError when sub_index exceeds the actor count."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        with pytest.raises(IndexError, match="out of range"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=2)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_negative(self) -> None:
        """inject_fault raises IndexError when sub_index is negative."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        with pytest.raises(IndexError, match="out of range"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=-1)


class TestBaseCellHandleInjectFault:
    @pytest.mark.asyncio
    async def test_base_inject_fault_raises_not_implemented(self) -> None:
        """The base _CellHandle.inject_fault raises NotImplementedError naming the subclass."""
        handle = _ConcreteCellHandle()

        with pytest.raises(NotImplementedError, match="_ConcreteCellHandle does not support fault injection"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=0)
