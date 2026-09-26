import asyncio
import sys
from collections.abc import Callable

import pytest
from tests.fast.utils.test_utils.fault_injector.fakes import _CellOperations, _Clock, _Controller, _Timer

from miles.utils.test_utils.fault_injector import controller as controller_module
from miles.utils.test_utils.fault_injector import request_executor
from miles.utils.test_utils.fault_injector.actions import cell
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.cell import StartCellAction, StopCellAction
from miles.utils.test_utils.fault_injector.actions.process import ExitProcessAction
from miles.utils.test_utils.fault_injector.controller import (
    FaultHookCommand,
    FaultHookConflictError,
    FaultHookOperation,
    _FaultHookController,
    reach_fault_hook,
    reach_fault_hook_async,
)
from miles.utils.test_utils.fault_injector.models import (
    DeclaredFaultHookTarget,
    FaultHookName,
    FaultHookOwner,
    FaultHookRecord,
    FaultHookRequest,
    FaultHookStatus,
)

_CELL_HOOK = FaultHookName.TRAINER_CONTROLLER_STEP_END
_ACTOR_HOOK = FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE
_CRASH = FaultHookRequest(
    request_id="crash",
    hook_name=_ACTOR_HOOK,
    action=ExitProcessAction(),
    rollout_id=4,
    attempt=0,
    target=DeclaredFaultHookTarget(cell_id="trainer-engine-actor-1", rank=0),
)
_SEND = FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND


def _stop(request_id: str = "stop", cell_id: str = "cell-0", **fields: object) -> FaultHookRequest:
    return FaultHookRequest(
        request_id=request_id,
        hook_name=fields.pop("hook_name", _CELL_HOOK),
        action=StopCellAction(cell_id=cell_id),
        **fields,
    )


def _set(hooks: _FaultHookController, request: FaultHookRequest) -> FaultHookRecord:
    return hooks.apply(FaultHookCommand(operation=FaultHookOperation.SET, request=request))


def _clear(hooks: _FaultHookController, request: FaultHookRequest) -> FaultHookRecord:
    return hooks.apply(FaultHookCommand(operation=FaultHookOperation.CLEAR, request=request))


def _statuses(records: list[FaultHookRecord], request_id: str) -> list[FaultHookStatus]:
    return [record.status for record in records if record.request.request_id == request_id]


class TestCellHooks:
    @pytest.mark.parametrize("rollout_id,expected", [(2, []), (3, ["trainer-engine-actor-0"])])
    async def test_stop_only_fires_at_the_declared_step(
        self,
        configure_hooks: Callable[..., _FaultHookController],
        operations: _CellOperations,
        rollout_id: int,
        expected: list[str],
    ) -> None:
        """Stopping must happen only after the configured trainer step finishes."""
        request = FaultHookRequest(
            request_id="stop",
            hook_name=_CELL_HOOK,
            rollout_id=3,
            action=StopCellAction(cell_id="trainer-engine-actor-0"),
        )
        hooks = configure_hooks([request], owner=FaultHookOwner.TRAINER_CONTROLLER, operations=operations)
        await hooks._reach_async(_CELL_HOOK, {"rollout_id": rollout_id})
        assert operations.stopped == expected
        assert operations.started == []

    @pytest.mark.parametrize("observed_after_reads", [0, 1, 2])
    async def test_start_waits_until_the_resumed_cell_is_observed(
        self,
        configure_hooks: Callable[..., _FaultHookController],
        operations: _CellOperations,
        observed_after_reads: int,
    ) -> None:
        """Resuming a dropped cell must wait for its return to the controller membership."""
        controller = _Controller(
            observed_after_reads=observed_after_reads,
            cell_ids=("trainer-engine-actor-0", "trainer-engine-actor-2"),
            initial_cell_ids=("trainer-engine-actor-0",),
        )
        request = FaultHookRequest(
            request_id="start",
            hook_name=_CELL_HOOK,
            rollout_id=3,
            action=StartCellAction(cell_id="trainer-engine-actor-2"),
        )
        hooks = configure_hooks(
            [request], owner=FaultHookOwner.TRAINER_CONTROLLER, operations=operations, controller=controller
        )
        await hooks._reach_async(_CELL_HOOK, {"rollout_id": 3})
        assert operations.started == ["trainer-engine-actor-2"]
        assert operations.stopped == []
        assert controller.reads == observed_after_reads + 1

    async def test_unobserved_start_fails_at_the_bounded_deadline(
        self,
        configure_hooks: Callable[..., _FaultHookController],
        operations: _CellOperations,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A resumed cell that never rejoins must fail instead of making the scenario pass."""
        monkeypatch.setattr(cell, "CELL_RESUME_OBSERVED_TIMEOUT_SECONDS", 0.0)
        request = FaultHookRequest(
            request_id="start", hook_name=_CELL_HOOK, action=StartCellAction(cell_id="trainer-engine-actor-0")
        )
        hooks = configure_hooks(
            [request],
            owner=FaultHookOwner.TRAINER_CONTROLLER,
            operations=operations,
            controller=_Controller(observed_after_reads=sys.maxsize),
        )
        with pytest.raises(TimeoutError, match="was resumed but is not observed yet"):
            await hooks._reach_async(_CELL_HOOK, {})
        assert operations.started == ["trainer-engine-actor-0"]

    @pytest.mark.parametrize("reject_stop", [False, True])
    async def test_multiple_actions_preserve_order_and_stop_after_failure(
        self, configure_hooks: Callable[..., _FaultHookController], reject_stop: bool
    ) -> None:
        """A rejected transition must propagate before later actions can alter membership."""
        operations = _CellOperations(reject_stop=reject_stop)
        requests = [
            FaultHookRequest(
                request_id="stop",
                hook_name=_CELL_HOOK,
                rollout_id=3,
                action=StopCellAction(cell_id="trainer-engine-actor-0"),
            ),
            FaultHookRequest(
                request_id="start",
                hook_name=_CELL_HOOK,
                rollout_id=3,
                action=StartCellAction(cell_id="trainer-engine-actor-2"),
            ),
        ]
        hooks = configure_hooks(
            requests,
            owner=FaultHookOwner.TRAINER_CONTROLLER,
            operations=operations,
            controller=_Controller(cell_ids=("trainer-engine-actor-0", "trainer-engine-actor-2")),
        )
        if reject_stop:
            with pytest.raises(RuntimeError, match="rejected the stop"):
                await hooks._reach_async(_CELL_HOOK, {"rollout_id": 3})
            assert operations.stopped == operations.started == []
        else:
            await hooks._reach_async(_CELL_HOOK, {"rollout_id": 3})
            assert operations.stopped == ["trainer-engine-actor-0"]
            assert operations.started == ["trainer-engine-actor-2"]
            await hooks._reach_async(_CELL_HOOK, {"rollout_id": 3})
            assert operations.stopped == ["trainer-engine-actor-0"]
            assert operations.started == ["trainer-engine-actor-2"]

    async def test_no_plan_has_no_cell_side_effects(
        self, configure_hooks: Callable[..., _FaultHookController], operations: _CellOperations
    ) -> None:
        """Unarmed controllers must leave cell membership alone."""
        hooks = configure_hooks([], owner=FaultHookOwner.TRAINER_CONTROLLER, operations=operations)
        await hooks._reach_async(_CELL_HOOK, {"rollout_id": 5})
        assert operations.stopped == operations.started == []


class TestActorHooks:
    @pytest.mark.parametrize(
        "cell_id,rank,rollout_id,attempt,expected",
        [
            ("trainer-engine-actor-1", 0, 4, 0, [1]),
            ("trainer-engine-actor-0", 0, 4, 0, []),
            ("rollout-engine-1", 0, 4, 0, []),
            ("trainer-engine-actor-1", 1, 4, 0, []),
            ("trainer-engine-actor-1", 0, 3, 0, []),
            ("trainer-engine-actor-1", 0, 4, 1, []),
        ],
    )
    async def test_only_the_declared_actor_and_attempt_exit(
        self,
        configure_hooks: Callable[..., _FaultHookController],
        recorded_exit_codes: list[int],
        cell_id: str,
        rank: int,
        rollout_id: int,
        attempt: int,
        expected: list[int],
    ) -> None:
        """An injected crash must spare other cells, ranks, rollouts and retries."""
        hooks = configure_hooks([_CRASH], owner=FaultHookOwner.TRAINER_ACTOR, cell_id=cell_id, rank=rank)
        await hooks._reach_async(_ACTOR_HOOK, {"rollout_id": rollout_id, "attempt": attempt})
        assert recorded_exit_codes == expected

    @pytest.mark.parametrize("owner", [FaultHookOwner.TRAINER_CONTROLLER, FaultHookOwner.ORCHESTRATOR])
    async def test_other_owners_ignore_actor_requests(
        self,
        configure_hooks: Callable[..., _FaultHookController],
        recorded_exit_codes: list[int],
        owner: FaultHookOwner,
    ) -> None:
        """The same launch plan must not execute an actor fault in another owner."""
        hooks = configure_hooks([_CRASH], owner=owner)
        await hooks._reach_async(_ACTOR_HOOK, {"rollout_id": 4, "attempt": 0})
        assert recorded_exit_codes == []

    async def test_no_actor_plan_never_exits(
        self, configure_hooks: Callable[..., _FaultHookController], recorded_exit_codes: list[int]
    ) -> None:
        """Normal actors must not exit when they reach an unarmed hook."""
        hooks = configure_hooks([], owner=FaultHookOwner.TRAINER_ACTOR, cell_id="trainer-engine-actor-1", rank=0)
        await hooks._reach_async(_ACTOR_HOOK, {"rollout_id": 4, "attempt": 0})
        assert recorded_exit_codes == []

    async def test_mixed_plan_routes_each_request_to_its_owner(
        self,
        configure_hooks: Callable[..., _FaultHookController],
        recorded_exit_codes: list[int],
        operations: _CellOperations,
    ) -> None:
        """Loading a shared plan must retain the owner-specific action and ignore the others."""
        stop = FaultHookRequest(
            request_id="stop", hook_name=_CELL_HOOK, action=StopCellAction(cell_id="trainer-engine-actor-0")
        )
        requests = [stop, _CRASH]
        trainer = configure_hooks(
            requests, owner=FaultHookOwner.TRAINER_ACTOR, cell_id="trainer-engine-actor-1", rank=0
        )
        controller = configure_hooks(requests, owner=FaultHookOwner.TRAINER_CONTROLLER, operations=operations)
        await trainer._reach_async(_CELL_HOOK, {})
        await controller._reach_async(_ACTOR_HOOK, {"rollout_id": 4, "attempt": 0})
        assert operations.stopped == recorded_exit_codes == []
        await trainer._reach_async(_ACTOR_HOOK, {"rollout_id": 4, "attempt": 0})
        await controller._reach_async(_CELL_HOOK, {})
        assert recorded_exit_codes == [1]
        assert operations.stopped == ["trainer-engine-actor-0"]

    async def test_other_actor_hook_does_not_consume_the_crash_request(
        self, configure_hooks: Callable[..., _FaultHookController], recorded_exit_codes: list[int]
    ) -> None:
        """A different trainer hook must leave the armed crash pending for its exact hook."""
        hooks = configure_hooks([_CRASH], owner=FaultHookOwner.TRAINER_ACTOR, cell_id="trainer-engine-actor-1", rank=0)
        context = {"rollout_id": 4, "attempt": 0}
        await hooks._reach_async(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND, context)
        assert recorded_exit_codes == []
        await hooks._reach_async(_ACTOR_HOOK, context)
        await hooks._reach_async(_ACTOR_HOOK, context)
        assert recorded_exit_codes == [1]


class TestRuntimeSetAndClear:
    def test_an_immediate_request_fires_on_set_and_frees_its_id(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """A request without a hook must fire once when set and leave its ID reusable."""
        record = _set(runtime_hooks, _stop(hook_name=None))
        assert record.status == FaultHookStatus.FIRED
        assert operations.stopped == ["cell-0"]
        _set(runtime_hooks, _stop(hook_name=None, cell_id="cell-1"))
        assert operations.stopped == ["cell-0", "cell-1"]
        assert _statuses(hook_records(), "stop") == [FaultHookStatus.PENDING, FaultHookStatus.FIRED] * 2

    def test_a_duplicate_id_is_rejected_and_leaves_the_original_armed(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """Re-setting a pending ID must fail without replacing what it guards."""
        _set(runtime_hooks, _stop(rollout_id=3))
        with pytest.raises(FaultHookConflictError, match="already set"):
            _set(runtime_hooks, _stop(rollout_id=4, cell_id="cell-1"))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 4})
        assert operations.stopped == []
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]

    def test_the_same_trigger_under_another_id_is_rejected(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """Two IDs firing the same action at one trigger must not both be armed."""
        _set(runtime_hooks, _stop("a", rollout_id=3))
        with pytest.raises(FaultHookConflictError, match="same trigger"):
            _set(runtime_hooks, _stop("b", rollout_id=3))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]
        assert _statuses(hook_records(), "b") == []

    def test_requests_for_different_rollouts_coexist(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """Requests differing only in rollout must each fire at their own step."""
        _set(runtime_hooks, _stop("a", rollout_id=3))
        _set(runtime_hooks, _stop("b", rollout_id=4, cell_id="cell-1"))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 4})
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-1", "cell-0"]

    def test_a_matching_clear_disarms_the_request(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """A cleared request must never fire and cannot be cleared twice."""
        request = _stop(rollout_id=3)
        _set(runtime_hooks, request)
        assert _clear(runtime_hooks, request).status == FaultHookStatus.CLEARED
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == []
        with pytest.raises(FaultHookConflictError, match="never set"):
            _clear(runtime_hooks, request)
        assert _statuses(hook_records(), "stop") == [FaultHookStatus.PENDING, FaultHookStatus.CLEARED]

    def test_a_clear_that_differs_from_the_set_request_is_rejected(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """Clearing with the right ID but another request must leave the original armed."""
        _set(runtime_hooks, _stop(rollout_id=3))
        with pytest.raises(FaultHookConflictError, match="does not match"):
            _clear(runtime_hooks, _stop(rollout_id=4))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]

    def test_a_fired_request_can_no_longer_be_cleared_but_can_be_rearmed(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """A one-shot request must leave the controller after firing and free its ID."""
        request = _stop(rollout_id=3)
        _set(runtime_hooks, request)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        with pytest.raises(FaultHookConflictError, match="never set"):
            _clear(runtime_hooks, request)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]
        _set(runtime_hooks, request)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0", "cell-0"]

    async def test_a_reach_while_the_action_runs_does_not_fire_it_again(self) -> None:
        """A concurrent reach of the same hook must not re-run an action that already fired."""
        operations = _CellOperations(stop_gate=asyncio.Event())
        hooks = _FaultHookController()
        hooks.configure(resources=FaultHookResources(cell_operations=operations))
        _set(hooks, _stop(rollout_id=3))
        first = asyncio.create_task(hooks._reach_async(_CELL_HOOK, {"rollout_id": 3}))
        while not operations.entered:
            await asyncio.sleep(0)
        await hooks._reach_async(_CELL_HOOK, {"rollout_id": 3})
        operations.stop_gate.set()
        await first
        assert operations.entered == operations.stopped == ["cell-0"]

    async def test_rearming_an_id_while_its_first_action_fails_keeps_the_new_request(
        self, hook_records: Callable[[], list[FaultHookRecord]]
    ) -> None:
        """A late failure of the fired request must not consume the request re-set under its ID."""
        operations = _CellOperations(reject_stop=True, stop_gate=asyncio.Event())
        hooks = _FaultHookController()
        hooks.configure(resources=FaultHookResources(cell_operations=operations))
        _set(hooks, _stop(rollout_id=3))
        first = asyncio.create_task(hooks._reach_async(_CELL_HOOK, {"rollout_id": 3}))
        while not operations.entered:
            await asyncio.sleep(0)
        assert _set(hooks, _stop(rollout_id=4, cell_id="cell-1")).status == FaultHookStatus.PENDING
        operations.stop_gate.set()
        with pytest.raises(RuntimeError, match="rejected the stop"):
            await first
        operations.reject_stop = False
        await hooks._reach_async(_CELL_HOOK, {"rollout_id": 4})
        assert operations.stopped == ["cell-1"]
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.FIRED,
            FaultHookStatus.PENDING,
            FaultHookStatus.FAILED,
            FaultHookStatus.FIRED,
        ]


class TestReachEntries:
    def test_the_sync_module_entry_reaches_the_process_controller(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The sync entry used by instrumented code must dispatch through the process controller."""
        monkeypatch.setattr(controller_module, "fault_hook_controller", runtime_hooks)
        _set(runtime_hooks, _stop(rollout_id=3))
        reach_fault_hook(_CELL_HOOK, rollout_id=2)
        assert operations.stopped == []
        reach_fault_hook(_CELL_HOOK, rollout_id=3)
        assert operations.stopped == ["cell-0"]

    async def test_the_async_module_entry_reaches_the_process_controller(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The async entry used by instrumented code must dispatch through the process controller."""
        monkeypatch.setattr(controller_module, "fault_hook_controller", runtime_hooks)
        _set(runtime_hooks, _stop(rollout_id=3))
        await reach_fault_hook_async(_CELL_HOOK, rollout_id=3)
        assert operations.stopped == ["cell-0"]

    def test_a_sync_failure_without_a_loop_is_raised_and_recorded_once(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """A failing action on the sync entry must reach the caller, end FAILED and never retry."""
        operations.reject_stop = True
        _set(runtime_hooks, _stop(rollout_id=3))
        with pytest.raises(RuntimeError, match="rejected the stop"):
            runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.entered == ["cell-0"]
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.FIRED,
            FaultHookStatus.FAILED,
        ]

    async def test_a_sync_failure_inside_a_running_loop_is_raised(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """A sync reach from code already on an event loop must still surface the action failure."""
        operations.reject_stop = True
        _set(runtime_hooks, _stop(rollout_id=3))
        with pytest.raises(RuntimeError, match="rejected the stop"):
            runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})

    async def test_a_sync_reach_inside_a_running_loop_completes_before_returning(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """A sync reach on a busy loop must run the action to completion before the caller continues."""
        _set(runtime_hooks, _stop(rollout_id=3))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]

    def test_the_fired_record_carries_the_reached_context_and_time(
        self,
        runtime_hooks: _FaultHookController,
        clock: _Clock,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """The FIRED event must say where and when the hook was reached."""
        _set(runtime_hooks, _stop(rollout_id=3))
        clock.advance(2.0)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3, "attempt": 1})
        [pending, fired] = hook_records()
        assert pending.status == FaultHookStatus.PENDING and pending.context is None
        assert pending.set_at == pending.changed_at == 100.0
        assert fired.status == FaultHookStatus.FIRED
        assert fired.context == FaultHookContext(rollout_id=3, attempt=1)
        assert fired.set_at == 100.0
        assert fired.reached_at == fired.due_at == fired.changed_at == 102.0

    def test_an_event_log_failure_does_not_block_the_fault(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken event logger must not stop a fault from being injected."""

        def broken() -> None:
            raise OSError("disk full")

        monkeypatch.setattr(request_executor, "is_event_logger_initialized", lambda: True)
        monkeypatch.setattr(request_executor, "get_event_logger", broken)
        _set(runtime_hooks, _stop(hook_name=None))
        assert operations.stopped == ["cell-0"]


class TestWithContext:
    @pytest.mark.parametrize("weight_version,expected", [(7, ["cell-0"]), (8, [])])
    async def test_a_reach_inherits_the_weight_version_of_the_update(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        weight_version: int,
        expected: list[str],
    ) -> None:
        """A weight update hook must match on the version its enclosing update publishes."""
        _set(runtime_hooks, _stop(hook_name=_SEND, weight_version=7, rollout_id=3))
        with runtime_hooks.with_context(FaultHookContext(weight_version=weight_version, rollout_id=3)):
            await runtime_hooks._reach_async(_SEND, {})
        assert operations.stopped == expected

    async def test_reach_arguments_override_the_inherited_context(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """Values passed at the hook must win over the enclosing update context."""
        _set(runtime_hooks, _stop("a", hook_name=_SEND, rollout_id=3))
        _set(runtime_hooks, _stop("b", hook_name=_SEND, rollout_id=4, cell_id="cell-1"))
        with runtime_hooks.with_context(FaultHookContext(rollout_id=3)):
            await runtime_hooks._reach_async(_SEND, {"rollout_id": 4})
        assert operations.stopped == ["cell-1"]

    async def test_the_context_is_cleared_even_when_the_update_raises(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """A failed update must not leak its version into later hooks."""
        _set(runtime_hooks, _stop(hook_name=_SEND, weight_version=7))
        with pytest.raises(ValueError, match="update failed"):
            with runtime_hooks.with_context(FaultHookContext(weight_version=7)):
                raise ValueError("update failed")
        await runtime_hooks._reach_async(_SEND, {})
        assert operations.stopped == []

    def test_an_immediate_request_set_inside_an_update_records_its_context(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations
    ) -> None:
        """An immediate request must be recorded with the context of the update it interrupted."""
        context = FaultHookContext(weight_version=7, rollout_id=3, debug_weight_update_id="u-1")
        with runtime_hooks.with_context(context):
            record = _set(runtime_hooks, _stop(hook_name=None))
        assert record.context == context
        assert operations.stopped == ["cell-0"]


class TestDelayedRequests:
    def test_a_delayed_request_schedules_one_timer_for_its_exact_delay(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """Reaching a delayed hook must arm one daemon timer and act only when it is due."""
        _set(runtime_hooks, _stop(rollout_id=3, delay_ms=250))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        [timer] = timers
        assert timer.started and timer.daemon
        assert timer.interval == 0.25
        assert operations.stopped == []
        [_, scheduled] = hook_records()
        assert scheduled.status == FaultHookStatus.SCHEDULED
        assert scheduled.reached_at == 100.0 and scheduled.due_at == 100.25

    def test_the_due_callback_fires_exactly_once(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """A due timer must run the action once even if its callback runs twice."""
        _set(runtime_hooks, _stop(rollout_id=3, delay_ms=250))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        clock.advance(0.25)
        timers[0].fire()
        timers[0].fire()
        assert operations.stopped == ["cell-0"]
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.SCHEDULED,
            FaultHookStatus.FIRED,
        ]

    def test_a_cleared_request_ignores_its_late_callback(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """Clearing a scheduled request must cancel it even if the timer already started running."""
        request = _stop(rollout_id=3, delay_ms=250)
        _set(runtime_hooks, request)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        _clear(runtime_hooks, request)
        assert timers[0].cancelled
        timers[0].fire()
        assert operations.stopped == []
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.SCHEDULED,
            FaultHookStatus.CLEARED,
        ]

    def test_a_stale_callback_cannot_fire_the_request_rearmed_under_its_id(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
    ) -> None:
        """A timer of a cleared request must never fire its successor with the same ID."""
        first = _stop(rollout_id=3, delay_ms=250)
        _set(runtime_hooks, first)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        _clear(runtime_hooks, first)
        _set(runtime_hooks, _stop(rollout_id=4, delay_ms=250, cell_id="cell-1"))
        timers[0].fire()
        assert operations.stopped == []
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 4})
        timers[0].fire()
        assert operations.stopped == []
        timers[1].fire()
        assert operations.stopped == ["cell-1"]

    def test_a_failing_due_action_is_recorded_failed(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """A delayed action that fails must leave FAILED evidence and not be retried."""
        operations.reject_stop = True
        _set(runtime_hooks, _stop(rollout_id=3, delay_ms=250))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        with pytest.raises(RuntimeError, match="rejected the stop"):
            timers[0].fire()
        timers[0].fire()
        assert operations.entered == ["cell-0"]
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.SCHEDULED,
            FaultHookStatus.FIRED,
            FaultHookStatus.FAILED,
        ]

    def test_an_immediate_request_with_a_delay_is_scheduled_on_set(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
    ) -> None:
        """A request without a hook but with a delay must wait for its timer after being set."""
        assert _set(runtime_hooks, _stop(hook_name=None, delay_ms=100)).status == FaultHookStatus.SCHEDULED
        assert operations.stopped == []
        assert timers[0].interval == 0.1
        timers[0].fire()
        assert operations.stopped == ["cell-0"]

    def test_a_scheduled_request_still_holds_its_trigger(
        self,
        runtime_hooks: _FaultHookController,
        clock: _Clock,
        timers: list[_Timer],
    ) -> None:
        """A request waiting on its timer must keep refusing a duplicate trigger until it fires."""
        _set(runtime_hooks, _stop("a", rollout_id=3, delay_ms=250))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        with pytest.raises(FaultHookConflictError):
            _set(runtime_hooks, _stop("b", rollout_id=3, delay_ms=250))
        timers[0].fire()
        assert _set(runtime_hooks, _stop("b", rollout_id=3, delay_ms=250)).status == FaultHookStatus.PENDING


class TestLifetime:
    def test_a_request_fires_just_before_its_deadline(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations, clock: _Clock
    ) -> None:
        """A request reached before set time plus lifetime must still fire."""
        _set(runtime_hooks, _stop(rollout_id=3, lifetime_seconds=5))
        clock.advance(4.999)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]

    def test_a_request_reached_at_its_exact_deadline_expires(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """Reaching the hook exactly at the deadline must expire the request instead of firing it."""
        request = _stop(rollout_id=3, lifetime_seconds=5)
        _set(runtime_hooks, request)
        clock.advance(5.0)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == []
        with pytest.raises(FaultHookConflictError, match="never set"):
            _clear(runtime_hooks, request)
        [_, expired] = hook_records()
        assert expired.status == FaultHookStatus.EXPIRED
        assert expired.changed_at == 105.0

    def test_an_expired_id_and_trigger_can_be_set_again(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """Expiry must be noticed by the next SET so the same ID and trigger are free again."""
        _set(runtime_hooks, _stop(rollout_id=3, lifetime_seconds=5))
        clock.advance(5.0)
        _set(runtime_hooks, _stop(rollout_id=3))
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.EXPIRED,
            FaultHookStatus.PENDING,
            FaultHookStatus.FIRED,
        ]

    def test_a_scheduled_request_that_outlives_its_lifetime_never_fires(
        self,
        runtime_hooks: _FaultHookController,
        operations: _CellOperations,
        clock: _Clock,
        timers: list[_Timer],
        hook_records: Callable[[], list[FaultHookRecord]],
    ) -> None:
        """Lifetime must count from SET, so a timer due after the deadline must expire instead."""
        _set(runtime_hooks, _stop(rollout_id=3, lifetime_seconds=5, delay_ms=1000))
        clock.advance(4.5)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        clock.advance(1.0)
        timers[0].fire()
        assert operations.stopped == []
        assert timers[0].cancelled
        assert _statuses(hook_records(), "stop") == [
            FaultHookStatus.PENDING,
            FaultHookStatus.SCHEDULED,
            FaultHookStatus.EXPIRED,
        ]

    def test_a_request_without_lifetime_never_expires(
        self, runtime_hooks: _FaultHookController, operations: _CellOperations, clock: _Clock
    ) -> None:
        """Omitting the lifetime must keep the request armed indefinitely."""
        _set(runtime_hooks, _stop(rollout_id=3))
        clock.advance(10_000.0)
        runtime_hooks._reach(_CELL_HOOK, {"rollout_id": 3})
        assert operations.stopped == ["cell-0"]
