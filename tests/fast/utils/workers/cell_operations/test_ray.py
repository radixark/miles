from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.test_utils.fault_hooks import FaultHookCommand, FaultHookRequest
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import (
    CellTerminationNotConfirmedError,
    CellTerminationOutcome,
    FaultTarget,
)
from miles.utils.workers.cell_operations.ray import RayCellOperations

_TRAINER_CELL_ID = "trainer-engine-actor-00001"


class _RecordingRemoteMethod:
    def __init__(self, *, name: str, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
        self._name = name
        self._calls = calls
        self.result: Any = {}
        self.hangs: bool = False

    async def remote(self, *args: Any, **kwargs: Any) -> Any:
        self._calls.append((self._name, args, kwargs))
        if self.hangs:
            await asyncio.Event().wait()
        return self.result


class _RecordingWorkerManagerHandle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_cell_infos = _RecordingRemoteMethod(name="get_cell_infos", calls=self.calls)
        self.start_cells = _RecordingRemoteMethod(name="start_cells", calls=self.calls)
        self.stop_cells = _RecordingRemoteMethod(name="stop_cells", calls=self.calls)
        self.inject_fault = _RecordingRemoteMethod(name="inject_fault", calls=self.calls)
        self.control_fault_hook = _RecordingRemoteMethod(name="control_fault_hook", calls=self.calls)
        self.stop_cell_incarnation = _RecordingRemoteMethod(name="stop_cell_incarnation", calls=self.calls)


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

    async def test_hook_control_preserves_incarnation_and_cancellation_request(self) -> None:
        """The Ray bridge forwards the complete cancellation rather than issuing a new arm."""
        fixture = _make_fixture()
        target = FaultTarget(cell_id=_TRAINER_CELL_ID, sub_index=1, workers_hash="old-generation")
        command = FaultHookCommand(
            operation="cancel",
            request=FaultHookRequest(
                request_id="cancel-me", instance_id="old-process", hook="trainer_before_all_gather", mode="exit"
            ),
        )
        fixture.worker_manager.control_fault_hook.result = "response"
        assert await fixture.operations.control_fault_hook(target=target, command=command) == "response"
        assert fixture.worker_manager.calls == [("control_fault_hook", (), {"target": target, "command": command})]

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


class TestRayCellOperationsTerminateIncarnation:
    async def test_the_manager_decides_whether_the_incarnation_still_matches(self) -> None:
        """The check and the kill must happen under the manager's lock, not be split across two calls from here."""
        fixture = _make_fixture()
        fixture.worker_manager.stop_cell_incarnation.result = "terminated"

        outcome = await fixture.operations.terminate_incarnation(
            cell_id=_TRAINER_CELL_ID, expected_workers_hash="hash-1"
        )

        assert outcome is CellTerminationOutcome.TERMINATED
        assert fixture.worker_manager.calls == [
            ("stop_cell_incarnation", (_TRAINER_CELL_ID,), {"expected_workers_hash": "hash-1"})
        ]

    async def test_a_stale_request_is_answered_as_stale(self) -> None:
        """The caller has to be able to tell a killed incarnation from one that was already replaced."""
        fixture = _make_fixture()
        fixture.worker_manager.stop_cell_incarnation.result = "stale"

        outcome = await fixture.operations.terminate_incarnation(
            cell_id=_TRAINER_CELL_ID, expected_workers_hash="hash-1"
        )

        assert outcome is CellTerminationOutcome.STALE

    async def test_a_manager_that_never_answers_is_not_a_confirmed_termination(self) -> None:
        """The manager itself can be wedged, and waiting on it forever is the deadlock this op removes."""
        fixture = _make_fixture()
        fixture.worker_manager.stop_cell_incarnation.hangs = True

        with pytest.raises(CellTerminationNotConfirmedError):
            await fixture.operations.terminate_incarnation(
                cell_id=_TRAINER_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_rollout_cell_is_terminated_through_the_same_manager_call(self) -> None:
        """The inference controller terminates a failed engine from inside its own lock, so nothing may re-enter it."""
        fixture = _make_fixture()
        fixture.worker_manager.stop_cell_incarnation.result = "terminated"

        outcome = await asyncio.wait_for(
            fixture.operations.terminate_incarnation(cell_id="engine-0-2", expected_workers_hash="hash-1"),
            timeout=5.0,
        )

        assert outcome is CellTerminationOutcome.TERMINATED
        assert fixture.worker_manager.calls == [
            ("stop_cell_incarnation", ("engine-0-2",), {"expected_workers_hash": "hash-1"})
        ]

    async def test_a_kill_the_manager_could_not_confirm_is_not_a_termination(self) -> None:
        """The manager killed the actors but they kept answering, which is not the confirmed death it promises."""
        fixture = _make_fixture()
        fixture.worker_manager.stop_cell_incarnation.result = "not_confirmed"

        with pytest.raises(CellTerminationNotConfirmedError):
            await fixture.operations.terminate_incarnation(cell_id=_TRAINER_CELL_ID, expected_workers_hash="hash-1")
