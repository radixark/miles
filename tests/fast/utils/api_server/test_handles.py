from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.train.group import TrainerController
from miles.utils.ft_utils.api_server.handles import _ActorCellHandle, _CellHandle, _RolloutCellHandle
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import (
    MockInferenceController,
    MockRayTrainCell,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_group,
)

ENGINE_CELL_ID = "inference-engine-0-0-0"
_PENDING_STATUS = CellStatus(phase="Pending", conditions=[CellCondition.allocated(TriState.TRUE)])
_SUSPENDED_STATUS = CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])


def _running_status(health: TriState) -> CellStatus:
    return CellStatus(
        phase="Running",
        conditions=[CellCondition.allocated(TriState.TRUE), CellCondition.from_health_checker_status(health)],
    )


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
                "observedWorkersHash": None,
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


def _make_rollout_handle(
    *,
    cell_id: str = ENGINE_CELL_ID,
    suspended: bool = False,
    health: TriState | None = TriState.TRUE,
    status: CellStatus | None = None,
) -> tuple[_RolloutCellHandle, MockWorkerManager, MockInferenceController]:
    manager = MockWorkerManager(make_cell_summaries(cell_id, suspended=suspended))
    resolved = status if status is not None else (_running_status(health) if health is not None else None)
    if suspended and status is None:
        resolved = _SUSPENDED_STATUS
    controller = MockInferenceController({cell_id: resolved} if resolved is not None else {})
    handle = _RolloutCellHandle(
        worker_manager=manager,
        inference_controller=controller,
        rollout_cell_id=cell_id,
    )
    return handle, manager, controller


class TestRolloutCellHandle:
    @pytest.mark.asyncio
    async def test_a_healthy_cell_is_reported_running(self) -> None:
        """A serving engine that answers its probe is what the heal loop must leave alone."""
        handle, _manager, _controller = _make_rollout_handle()

        cell = await handle.get_cell()

        assert cell.metadata.name == "rollout-inference-engine-0-0-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"
        assert cell.spec.suspend is False
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
        ]

    @pytest.mark.asyncio
    async def test_a_failing_probe_is_reported_unhealthy(self) -> None:
        """This is the signal the mini ft controller heals on."""
        handle, _manager, _controller = _make_rollout_handle(health=TriState.FALSE)

        cell = await handle.get_cell()

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
        ]

    @pytest.mark.asyncio
    async def test_suspension_comes_from_the_controller_status(self) -> None:
        """The controller is the only status source, so suspension is read off its document too."""
        handle, _manager, _controller = _make_rollout_handle(suspended=True)

        cell = await handle.get_cell()

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_no_health(self) -> None:
        """Its engine is gone, so nothing may claim a health verdict for it."""
        handle, _manager, _controller = _make_rollout_handle(suspended=True)

        cell = await handle.get_cell()

        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.FALSE)]

    @pytest.mark.asyncio
    async def test_the_worker_manager_is_never_read_to_build_a_status(self) -> None:
        """Merging the manager's live view into the controller's status is how one generation's verdict lands on another."""
        handle, manager, _controller = _make_rollout_handle()

        await handle.get_cell()

        assert manager.cell_info_calls == []

    @pytest.mark.asyncio
    async def test_a_cell_the_controller_does_not_track_yet_carries_no_health(self) -> None:
        """Reconcile only hands the manager's cell to the controller a poll period later."""
        handle, _manager, _controller = _make_rollout_handle(health=None)

        cell = await handle.get_cell()

        assert cell.status.phase == "Pending"
        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.TRUE)]

    @pytest.mark.asyncio
    async def test_the_controllers_status_is_passed_through_verbatim(self) -> None:
        """The handle renders what the controller computed; re-deriving phase here would let the two disagree."""
        handle, _manager, _controller = _make_rollout_handle(status=_PENDING_STATUS)

        cell = await handle.get_cell()

        assert cell.status == _PENDING_STATUS

    @pytest.mark.asyncio
    async def test_health_is_read_without_awaiting_the_controller(self) -> None:
        """The api server serves from its own event loop, so the controller is read synchronously."""
        handle, _manager, controller = _make_rollout_handle()

        await handle.get_cell()

        assert controller.status_calls == 1

    @pytest.mark.asyncio
    async def test_suspend_stops_the_cell_in_the_worker_manager(self) -> None:
        """The manager owns the processes, so healing goes through it, not the controller."""
        handler, manager, _controller = _make_rollout_handle()

        await handler.suspend()

        assert manager.stopped_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_resume_starts_the_cell_in_the_worker_manager(self) -> None:
        """Resume relaunches the cell, which reconcile then observes as a new generation."""
        handler, manager, _controller = _make_rollout_handle(suspended=True)

        await handler.resume()

        assert manager.started_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_suspended_afterwards(self) -> None:
        """The heal loop reads back the status it just asked for."""
        handler, _manager, _controller = _make_rollout_handle()

        await handler.suspend()
        cell = await handler.get_cell()

        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_resumed_cell_is_pending_without_health_until_reconcile_sees_it(self) -> None:
        """The new generation is still gated, so calling it Running would be the old process' verdict."""
        handler, _manager, _controller = _make_rollout_handle(suspended=True)

        await handler.resume()
        cell = await handler.get_cell()

        assert cell.status.phase == "Pending"
        assert cell.spec.suspend is False
        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.TRUE)]

    @pytest.mark.asyncio
    async def test_a_suspend_resume_pair_within_one_poll_interval_drops_the_old_verdict(self) -> None:
        """This is the window the mini ft heal loop lives in; a stale Healthy here means the new cell is never healed."""
        handler, _manager, controller = _make_rollout_handle(health=TriState.TRUE)

        await handler.suspend()
        await handler.resume()
        cell = await handler.get_cell()

        assert cell.status.phase == "Pending"
        assert [c.type for c in cell.status.conditions] == ["Allocated"]
        assert controller.status_calls == 1

    @pytest.mark.asyncio
    async def test_the_status_comes_back_once_reconcile_observes_the_new_generation(self) -> None:
        """The published document must recover on its own, without another suspend or resume."""
        handler, _manager, controller = _make_rollout_handle(suspended=True)
        await handler.resume()

        controller.observe_cell(ENGINE_CELL_ID, _running_status(TriState.TRUE))
        cell = await handler.get_cell()

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
        ]

    def test_cell_type_is_rollout(self) -> None:
        handle, _manager, _controller = _make_rollout_handle(cell_id="inference-engine-0-0-3")
        assert handle.cell_type == "rollout"
        assert handle.cell_id == "rollout-inference-engine-0-0-3"


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
    def cell_key(self) -> str:
        return "0"

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
