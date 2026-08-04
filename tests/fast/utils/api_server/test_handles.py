from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.train.group import TrainerController
from miles.utils.ft_utils.api_server.handles import _ActorCellHandler, _CellHandler, _RolloutCellHandler
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import (
    MockInferenceController,
    MockRayTrainCell,
    MockRemoteCall,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_group,
)

_PENDING_STATUS = CellStatus(phase="Pending", conditions=[CellCondition.allocated(TriState.TRUE)])
_SUSPENDED_STATUS = CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])


def _running_status(health: TriState) -> CellStatus:
    return CellStatus(
        phase="Running",
        conditions=[CellCondition.allocated(TriState.TRUE), CellCondition.from_health_checker_status(health)],
    )


class TestActorCellHandler:
    async def test_every_cell_of_the_group_is_listed(self) -> None:
        """The api server addresses trainer cells by their index in the group."""
        handler = _ActorCellHandler(group=make_mock_group([MockRayTrainCell(), MockRayTrainCell()]))
        assert await handler.list_cell_ids() == ["actor-0", "actor-1"]

    def test_cell_type(self) -> None:
        """The cell type is the api-server-visible label the ft controller filters on."""
        handler = _ActorCellHandler(group=make_mock_group([MockRayTrainCell()]))
        assert handler.cell_type == "actor"

    @pytest.mark.asyncio
    async def test_get_cell_returns_full_cell_structure(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handler = _ActorCellHandler(group=group)
        cell = await handler.get_cell("actor-0")

        assert cell.model_dump() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-id": "actor-0",
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
        handler = _ActorCellHandler(group=group)
        cell = await handler.get_cell("actor-0")

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.stop_cell = AsyncMock()
        handler = _ActorCellHandler(group=group)
        await handler.suspend("actor-2")
        group.stop_cell.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_resume_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.start_cell = MagicMock()
        handler = _ActorCellHandler(group=group)
        await handler.resume("actor-1")
        group.start_cell.assert_called_once_with(1)


ENGINE_CELL_ID = "inference-engine-0-0-0"


def _pool_ids_of(manager: MockWorkerManager) -> list[str]:
    return sorted({summary.pool_id for summary in manager._summaries.values()})


def _make_rollout_handler(
    *,
    cell_id: str = ENGINE_CELL_ID,
    suspended: bool = False,
    health: TriState | None = TriState.TRUE,
    status: CellStatus | None = None,
) -> tuple[_RolloutCellHandler, MockWorkerManager, MockInferenceController]:
    manager = MockWorkerManager(make_cell_summaries(cell_id, suspended=suspended))
    resolved = status if status is not None else (_running_status(health) if health is not None else None)
    if suspended and status is None:
        resolved = _SUSPENDED_STATUS
    controller = MockInferenceController({cell_id: resolved} if resolved is not None else {})
    handler = _RolloutCellHandler(
        worker_manager=manager,
        inference_controller=controller,
    )
    return handler, manager, controller


class TestRolloutCellHandler:
    @pytest.mark.asyncio
    async def test_a_healthy_cell_is_reported_running(self) -> None:
        """A serving engine that answers its probe is what the heal loop must leave alone."""
        handler, _manager, _controller = _make_rollout_handler()

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.metadata.name == "inference-engine-0-0-0"
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
        handler, _manager, _controller = _make_rollout_handler(health=TriState.FALSE)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
        ]

    @pytest.mark.asyncio
    async def test_suspension_comes_from_the_controller_status(self) -> None:
        """The controller is the only status source, so suspension is read off its document too."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_no_health(self) -> None:
        """Its engine is gone, so nothing may claim a health verdict for it."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.FALSE)]

    @pytest.mark.asyncio
    async def test_the_worker_manager_is_never_read_to_build_a_status(self) -> None:
        """Merging the manager's live view into the controller's status is how one generation's verdict lands on another."""
        handler, manager, _controller = _make_rollout_handler()

        await handler.get_cell(ENGINE_CELL_ID)

        assert manager.cell_info_calls == []

    @pytest.mark.asyncio
    async def test_a_cell_the_controller_does_not_track_yet_carries_no_health(self) -> None:
        """Reconcile only hands the manager's cell to the controller a poll period later."""
        handler, _manager, _controller = _make_rollout_handler(health=None)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Pending"
        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.TRUE)]

    @pytest.mark.asyncio
    async def test_the_controllers_status_is_passed_through_verbatim(self) -> None:
        """The handle renders what the controller computed; re-deriving phase here would let the two disagree."""
        handler, _manager, _controller = _make_rollout_handler(status=_PENDING_STATUS)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status == _PENDING_STATUS

    @pytest.mark.asyncio
    async def test_health_is_read_without_awaiting_the_controller(self) -> None:
        """The api server serves from its own event loop, so the controller is read synchronously."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        controller = MockInferenceController()
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=controller)

        await handler.get_cell(ENGINE_CELL_ID)

        assert controller.status_calls == 1

    @pytest.mark.asyncio
    async def test_suspend_stops_the_cell_in_the_worker_manager(self) -> None:
        """The manager owns the processes, so healing goes through it, not the controller."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=MockInferenceController())

        await handler.suspend(ENGINE_CELL_ID)

        assert manager.stopped_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_resume_starts_the_cell_in_the_worker_manager(self) -> None:
        """Resume relaunches the cell, which reconcile then observes as a new generation."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, suspended=True))
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=MockInferenceController())

        await handler.resume(ENGINE_CELL_ID)

        assert manager.started_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_suspended_afterwards(self) -> None:
        """The heal loop reads back the status it just asked for."""
        handler, _manager, _controller = _make_rollout_handler()

        await handler.suspend(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_resumed_cell_is_pending_without_health_until_reconcile_sees_it(self) -> None:
        """The new generation is still gated, so calling it Running would be the old process' verdict."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        await handler.resume(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Pending"
        assert cell.spec.suspend is False
        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.TRUE)]

    @pytest.mark.asyncio
    async def test_a_suspend_resume_pair_within_one_poll_interval_drops_the_old_verdict(self) -> None:
        """This is the window the mini ft heal loop lives in; a stale Healthy here means the new cell is never healed."""
        handler, _manager, controller = _make_rollout_handler(health=TriState.TRUE)

        await handler.suspend(ENGINE_CELL_ID)
        await handler.resume(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Pending"
        assert [c.type for c in cell.status.conditions] == ["Allocated"]
        assert controller.status_calls == 1

    @pytest.mark.asyncio
    async def test_the_status_comes_back_once_reconcile_observes_the_new_generation(self) -> None:
        """The published document must recover on its own, without another suspend or resume."""
        handler, _manager, controller = _make_rollout_handler(suspended=True)
        await handler.resume(ENGINE_CELL_ID)

        controller.observe_cell(ENGINE_CELL_ID, _running_status(TriState.TRUE))
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
        ]

    def test_cell_type_is_rollout(self) -> None:
        """The cell type is what lets a soak target engines without crashing trainer cells."""
        handler, _manager, _controller = _make_rollout_handler(cell_id="inference-engine-0-0-3")
        assert handler.cell_type == "rollout"

    async def test_only_the_engine_specs_are_listed(self) -> None:
        """Routers and session servers are cells of the manager too, but not rollout cells."""
        manager = MockWorkerManager(
            {**make_cell_summaries("inference-engine-0-0-0"), **make_cell_summaries("miles-router-0")}
        )
        controller = MockInferenceController({"inference-engine-0-0-0": _running_status(TriState.TRUE)})
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=controller)

        assert await handler.list_cell_ids() == ["inference-engine-0-0-0"]

    async def test_a_suspended_cell_is_still_listed(self) -> None:
        """A suspended cell that vanished from the listing could never be resumed."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        assert await handler.list_cell_ids() == [ENGINE_CELL_ID]

    async def test_listing_reads_its_sources_once_for_all_cells(self) -> None:
        """This listing is polled for the life of the run, so it must not scale in round trips."""
        controller = MockInferenceController(
            {name: _running_status(TriState.TRUE) for name in ["engine-a", "engine-b", "engine-c"]}
        )
        manager = MockWorkerManager(make_cell_summaries("engine-a", "engine-b", "engine-c"))
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=controller)

        cells = await handler.list_cells()

        assert len(cells) == 3
        assert controller.status_calls == 1


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
    group._cells_by_index = {0: cell}
    return group


class _ConcreteCellHandler(_CellHandler):
    @property
    def cell_type(self) -> str:
        return "fake"

    async def list_cell_ids(self) -> list[str]:
        return ["0"]

    async def get_cell(self, cell_id: str) -> object:
        raise NotImplementedError

    async def suspend(self, cell_id: str) -> None:
        raise NotImplementedError

    async def resume(self, cell_id: str) -> None:
        raise NotImplementedError


class TestActorCellHandlerInjectFault:
    @pytest.mark.asyncio
    async def test_inject_fault_calls_actor_with_mode_value(self) -> None:
        """inject_fault forwards mode.value to the selected actor's remote handle."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        await handler.inject_fault("actor-0", mode=FailureMode.SIGKILL, sub_index=1)

        assert cell._actor.inject_fault.remote_calls == [(("sigkill",), {})]

    @pytest.mark.asyncio
    async def test_inject_fault_raises_when_cell_not_alive(self) -> None:
        """inject_fault raises RuntimeError when the target cell is not alive."""
        cell = _FakeInjectCell(is_alive=False, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        with pytest.raises(RuntimeError, match="not alive"):
            await handler.inject_fault("actor-0", mode=FailureMode.SIGKILL, sub_index=0)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_out_of_range(self) -> None:
        """inject_fault raises IndexError when sub_index exceeds the actor count."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        with pytest.raises(IndexError, match="out of range"):
            await handler.inject_fault("actor-0", mode=FailureMode.SIGKILL, sub_index=2)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_negative(self) -> None:
        """inject_fault raises IndexError when sub_index is negative."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        with pytest.raises(IndexError, match="out of range"):
            await handler.inject_fault("actor-0", mode=FailureMode.SIGKILL, sub_index=-1)


class TestBaseCellHandlerInjectFault:
    @pytest.mark.asyncio
    async def test_base_inject_fault_raises_not_implemented(self) -> None:
        """The base handler names the subclass that cannot inject faults."""
        handler = _ConcreteCellHandler()

        with pytest.raises(NotImplementedError, match="_ConcreteCellHandler does not support fault injection"):
            await handler.inject_fault("actor-0", mode=FailureMode.SIGKILL, sub_index=0)


class TestRolloutCellHandlerInjectFault:
    @pytest.mark.asyncio
    async def test_injection_is_forwarded_to_the_worker_manager(self) -> None:
        """The manager owns the actors, so it is the one that can crash them."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        manager.inject_fault = MockRemoteCall(None)
        handler = _RolloutCellHandler(
            worker_manager=manager,
            inference_controller=MockInferenceController(),
        )

        await handler.inject_fault(ENGINE_CELL_ID, mode=FailureMode.SIGKILL, sub_index=1)

        assert manager.inject_fault.calls == [
            ((ENGINE_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 1}),
        ]
