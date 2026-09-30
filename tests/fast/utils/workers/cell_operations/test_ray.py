from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import miles.utils.workers.cell_operations.ray as cell_operations_ray_mod
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.cell_operations.ray import RayCellOperations

_TRAINER_CELL_ID = "trainer-engine-actor-00001"


class _RecordingRemoteMethod:
    def __init__(self, *, name: str, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls
        self.result: dict[str, Any] = {}

    async def remote(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._calls.append((self._name, args, kwargs))
        return self.result


class _RecordingWorkerManagerHandle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_cell_infos = _RecordingRemoteMethod(name="get_cell_infos", calls=self.calls)
        self.start_cells = _RecordingRemoteMethod(name="start_cells", calls=self.calls)
        self.stop_cells = _RecordingRemoteMethod(name="stop_cells", calls=self.calls)
        self.inject_fault = _RecordingRemoteMethod(name="inject_fault", calls=self.calls)


@dataclass(frozen=True)
class _Fixture:
    worker_manager: _RecordingWorkerManagerHandle
    operations: RayCellOperations


def _make_fixture() -> _Fixture:
    worker_manager = _RecordingWorkerManagerHandle()
    return _Fixture(
        worker_manager=worker_manager,
        operations=RayCellOperations(worker_manager_handle=worker_manager),
    )


class TestRayCellOperationsDisruptiveOperations:
    """Every cell kind is stopped and crashed through the worker manager, with no controller in the path."""

    async def test_a_rollout_cells_suspend_reaches_the_worker_manager(self) -> None:
        """Routing it through the inference controller would deadlock against the weight-update lock."""
        fixture = _make_fixture()

        await asyncio.wait_for(fixture.operations.suspend(cell_id="engine-0-2"), timeout=5.0)

        assert fixture.worker_manager.calls == [("stop_cells", (["engine-0-2"],), {})]

    async def test_a_trainer_cells_suspend_reaches_the_worker_manager(self) -> None:
        """A trainer cell was already stopped this way, and the two kinds now take the same path."""
        fixture = _make_fixture()

        await asyncio.wait_for(fixture.operations.suspend(cell_id=_TRAINER_CELL_ID), timeout=5.0)

        assert fixture.worker_manager.calls == [("stop_cells", ([_TRAINER_CELL_ID],), {})]

    async def test_a_rollout_cells_fault_reaches_the_worker_manager(self) -> None:
        """The fault has to land while a weight update is running, which the controller detour forbade."""
        fixture = _make_fixture()

        await asyncio.wait_for(
            fixture.operations.inject_fault(cell_id="engine-0-2", mode=FailureMode.SIGKILL, sub_index=0), timeout=5.0
        )

        assert fixture.worker_manager.calls == [
            ("inject_fault", ("engine-0-2",), {"mode": "sigkill", "worker_in_cell_index": 0})
        ]

    async def test_a_trainer_cells_fault_reaches_the_worker_manager(self) -> None:
        """Regression: routing a trainer cell through the rollout controller raised, so the actor never died."""
        fixture = _make_fixture()

        await asyncio.wait_for(
            fixture.operations.inject_fault(cell_id=_TRAINER_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0),
            timeout=5.0,
        )

        assert fixture.worker_manager.calls == [
            ("inject_fault", (_TRAINER_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 0})
        ]

    async def test_a_cell_the_controller_never_listed_is_still_crashed(self) -> None:
        """A cell being replaced is exactly the one a soak wants to crash, and no membership read gates it."""
        fixture = _make_fixture()

        await asyncio.wait_for(
            fixture.operations.inject_fault(cell_id="engine-0-7", mode=FailureMode.SIGKILL, sub_index=0), timeout=5.0
        )

        assert fixture.worker_manager.calls == [
            ("inject_fault", ("engine-0-7",), {"mode": "sigkill", "worker_in_cell_index": 0})
        ]


class TestRayCellOperationsProtocol:
    async def test_cell_infos_forwards_pool_ids_and_returns_the_actor_result(self) -> None:
        """Cell info reads forward every pool ID by keyword and preserve the actor result."""
        fixture = _make_fixture()
        pool_ids = ["engine-0", "rollout-1"]
        actor_result = {"engine-0-2": SimpleNamespace(), "rollout-1-3": SimpleNamespace()}
        fixture.worker_manager.get_cell_infos.result = actor_result

        result = await fixture.operations.cell_infos(pool_ids=pool_ids)

        assert fixture.worker_manager.calls == [("get_cell_infos", (), {"pool_ids": pool_ids})]
        assert result is actor_result

    async def test_resume_starts_exactly_the_requested_cell(self) -> None:
        """Resuming a cell sends exactly that cell ID in a one-element list."""
        fixture = _make_fixture()

        await fixture.operations.resume(cell_id="engine-0-2")

        assert fixture.worker_manager.calls == [("start_cells", (["engine-0-2"],), {})]


class TestRayCellOperationsHasNoInferenceControllerPath:
    def test_the_constructor_takes_the_worker_manager_handle_alone(self) -> None:
        """A second constructor argument is how the controller detour came back the last time."""
        parameters = inspect.signature(RayCellOperations.__init__).parameters

        assert list(parameters) == ["self", "worker_manager_handle"]
        assert parameters["worker_manager_handle"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_constructing_with_a_resolve_inference_controller_argument_is_rejected(self) -> None:
        """The removed keyword must fail loudly rather than be accepted and quietly ignored."""
        with pytest.raises(TypeError):
            RayCellOperations(
                worker_manager_handle=_RecordingWorkerManagerHandle(),
                resolve_inference_controller=lambda: None,
            )

    def test_an_instance_holds_nothing_but_the_worker_manager_handle(self) -> None:
        """A cached controller handle on the instance is the state the M27 guard needed."""
        fixture = _make_fixture()

        assert list(vars(fixture.operations)) == ["_worker_manager_handle"]
        assert fixture.operations._worker_manager_handle is fixture.worker_manager

    def test_the_removed_trainer_cell_id_prefix_helper_is_gone(self) -> None:
        """Routing by cell-id prefix only existed to keep trainer cells away from the controller."""
        assert not hasattr(cell_operations_ray_mod, "_is_trainer_cell_id")

    def test_no_removed_controller_entry_point_survives_on_the_class(self) -> None:
        """Either name back on this class would mean a suspend can block on the weight-update lock."""
        for name in ("_controller", "_resolve_inference_controller", "stop_cell_between_weight_updates"):
            assert not hasattr(RayCellOperations, name), name

    def test_the_module_does_not_reach_into_the_ray_application_layer(self) -> None:
        """This layer sat below miles.ray, and the guard was the only reason it imported upwards."""
        assert "miles.ray" not in inspect.getsource(cell_operations_ray_mod)

    def test_every_base_operation_is_implemented_here(self) -> None:
        """A dropped override would silently fall back to an abstract method at heal time."""
        for name in ("cell_infos", "suspend", "resume", "inject_fault"):
            assert getattr(RayCellOperations, name) is not getattr(BaseCellOperations, name), name


class TestRayCellOperationsSuspendReachesTheWorkerManagerUnconditionally:
    async def test_a_cell_the_controller_never_listed_is_still_suspended(self) -> None:
        """A cell mid-replacement is exactly the one a heal loop must be able to stop."""
        fixture = _make_fixture()

        await asyncio.wait_for(fixture.operations.suspend(cell_id="engine-0-7"), timeout=5.0)

        assert fixture.worker_manager.calls == [("stop_cells", (["engine-0-7"],), {})]

    async def test_concurrent_suspends_all_reach_the_worker_manager(self) -> None:
        """No lock serializes suspends any more, so a fleet-wide stop cannot deadlock on one cell."""
        fixture = _make_fixture()
        cell_ids = ["engine-0-0", "engine-0-1", _TRAINER_CELL_ID]

        await asyncio.wait_for(
            asyncio.gather(*(fixture.operations.suspend(cell_id=cell_id) for cell_id in cell_ids)), timeout=5.0
        )

        assert sorted(args[0][0] for _, args, _ in fixture.worker_manager.calls) == sorted(cell_ids)

    async def test_a_suspend_and_a_fault_on_the_same_cell_both_land_in_order(self) -> None:
        """Stopping a cell and crashing it are independent calls, neither gating the other."""
        fixture = _make_fixture()

        await fixture.operations.suspend(cell_id="engine-0-2")
        await fixture.operations.inject_fault(cell_id="engine-0-2", mode=FailureMode.SIGKILL, sub_index=0)

        assert [name for name, _, _ in fixture.worker_manager.calls] == ["stop_cells", "inject_fault"]


class TestRayCellOperationsInjectFaultPayload:
    @pytest.mark.parametrize("mode", list(FailureMode))
    async def test_the_failure_mode_crosses_as_its_string_value(self, mode: FailureMode) -> None:
        """The worker manager takes the wire value, and an enum would not survive the actor call."""
        fixture = _make_fixture()

        await fixture.operations.inject_fault(cell_id="engine-0-2", mode=mode, sub_index=0)

        assert fixture.worker_manager.calls == [
            ("inject_fault", ("engine-0-2",), {"mode": mode.value, "worker_in_cell_index": 0})
        ]

    async def test_the_sub_index_names_the_worker_inside_the_cell(self) -> None:
        """A multi-worker cell needs the rank picked, not the whole cell crashed."""
        fixture = _make_fixture()

        await fixture.operations.inject_fault(cell_id=_TRAINER_CELL_ID, mode=FailureMode.SEGFAULT, sub_index=3)

        assert fixture.worker_manager.calls == [
            ("inject_fault", (_TRAINER_CELL_ID,), {"mode": "segfault", "worker_in_cell_index": 3})
        ]
