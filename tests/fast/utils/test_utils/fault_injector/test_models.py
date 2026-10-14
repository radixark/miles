import math

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils.audit_utils.event_logger.models import Event, FaultHookEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.actions.cell import StopCellAction
from miles.utils.test_utils.fault_injector.actions.process import DeadlockThreadAction, ObserveAction
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation
from miles.utils.test_utils.fault_injector.models import (
    DeclaredFaultHookTarget,
    FaultHookName,
    FaultHookOwner,
    FaultHookRecord,
    FaultHookRequest,
    FaultHookStatus,
    ObservedFaultHookTarget,
)

_SEND = FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND
_OBSERVED = ObservedFaultHookTarget(
    cell_id="rollout-engine-0", rank=1, workers_hash="hash-a", boot_uuid="boot-1", pod_uid="pod-1"
)


def _request(**change: object) -> FaultHookRequest:
    return FaultHookRequest.model_validate(
        {"request_id": "r", "hook_name": _SEND, "action": {"kind": "kill_process"}, "rollout_id": 3} | change
    )


class TestFaultHookName:
    @pytest.mark.parametrize(
        "name,owner",
        [
            (FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER, FaultHookOwner.TRAINER_ACTOR),
            (FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND, FaultHookOwner.TRAINER_ACTOR),
            (FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE, FaultHookOwner.TRAINER_ACTOR),
            (FaultHookName.TRAINER_CONTROLLER_STEP_END, FaultHookOwner.TRAINER_CONTROLLER),
            (FaultHookName.ORCHESTRATOR_STEP_END, FaultHookOwner.ORCHESTRATOR),
        ],
    )
    def test_every_hook_belongs_to_the_process_that_reaches_it(
        self, name: FaultHookName, owner: FaultHookOwner
    ) -> None:
        """Each hook name must route to the owner whose code reaches it."""
        assert name.owner is owner

    def test_the_owner_table_covers_every_hook(self) -> None:
        """A new hook name must not be added without an owner."""
        assert {name.owner for name in FaultHookName} == set(FaultHookOwner)


class TestFaultHookTargets:
    @pytest.mark.parametrize(
        "target,cell_id,rank,expected",
        [
            (DeclaredFaultHookTarget(), "trainer-0", 3, True),
            (DeclaredFaultHookTarget(), None, None, True),
            (DeclaredFaultHookTarget(cell_id="trainer-0"), "trainer-0", 5, True),
            (DeclaredFaultHookTarget(cell_id="trainer-0"), "trainer-1", 5, False),
            (DeclaredFaultHookTarget(cell_id="trainer-0"), None, 5, False),
            (DeclaredFaultHookTarget(rank=0), "trainer-0", 0, True),
            (DeclaredFaultHookTarget(rank=0), "trainer-0", 1, False),
            (DeclaredFaultHookTarget(cell_id="trainer-0", rank=0), "trainer-0", 0, True),
            (DeclaredFaultHookTarget(cell_id="trainer-0", rank=0), "trainer-0", None, False),
        ],
    )
    def test_an_omitted_field_is_a_wildcard_and_a_named_one_must_match(
        self, target: DeclaredFaultHookTarget, cell_id: str | None, rank: int | None, expected: bool
    ) -> None:
        """Only omitted target fields may match any process."""
        assert target.covers(cell_id=cell_id, rank=rank) is expected

    @pytest.mark.parametrize(
        "raw",
        [
            {"kind": "observed", "rank": 0, "workers_hash": "h"},
            {"kind": "observed", "cell_id": "c", "workers_hash": "h"},
            {"kind": "observed", "cell_id": "c", "rank": 0},
            {"kind": "observed", "cell_id": "c", "rank": 0, "workers_hash": ""},
            {"kind": "observed", "cell_id": "c", "rank": -1, "workers_hash": "h"},
            {"kind": "declared", "rank": -1},
            {"kind": "observed", "cell_id": "c", "rank": 0, "workers_hash": "h", "bogus": 1},
            {"kind": "unknown"},
        ],
    )
    def test_an_observed_target_must_name_one_incarnation(self, raw: dict[str, object]) -> None:
        """An observed target without its cell, rank or worker identity must be rejected."""
        with pytest.raises(ValidationError):
            _request(target=raw)

    def test_the_observed_identity_survives_a_json_round_trip(self) -> None:
        """Serialising an observed target must keep every identity field and its kind."""
        request = _request(target=_OBSERVED.model_dump(mode="json"))
        restored = FaultHookRequest.model_validate_json(request.model_dump_json())
        assert restored.target == _OBSERVED
        assert isinstance(restored.target, ObservedFaultHookTarget)

    def test_an_omitted_target_is_the_declared_wildcard(self) -> None:
        """A request without a target must be a declared wildcard, not an observed one."""
        assert _request().target == DeclaredFaultHookTarget()


class TestFaultHookRequestValidation:
    @pytest.mark.parametrize(
        "change",
        [
            {"delay_ms": -1},
            {"delay_ms": 300_001},
            {"delay_ms": math.inf},
            {"delay_ms": math.nan},
            {"lifetime_seconds": 0},
            {"lifetime_seconds": -1},
            {"lifetime_seconds": 300.001},
            {"lifetime_seconds": math.inf},
            {"lifetime_seconds": math.nan},
            {"attempt": -1},
            {"weight_version": -1},
            {"hook_name": "not_a_hook"},
        ],
    )
    def test_out_of_range_timing_and_filters_are_rejected(self, change: dict[str, object]) -> None:
        """Delays, lifetimes and filters outside their bounds must fail validation."""
        with pytest.raises(ValidationError):
            _request(**change)

    @pytest.mark.parametrize(
        "change", [{"delay_ms": 0}, {"delay_ms": 300_000}, {"lifetime_seconds": 300}, {"lifetime_seconds": 0.001}]
    )
    def test_the_inclusive_timing_bounds_are_accepted(self, change: dict[str, object]) -> None:
        """The largest delay and lifetime and the smallest positive lifetime must be valid."""
        _request(**change)

    def test_a_delayed_deadlock_is_rejected(self) -> None:
        """A thread deadlock must fire immediately or not be accepted at all."""
        with pytest.raises(ValidationError, match="requires immediate hook execution"):
            _request(action={"kind": "deadlock_thread"}, delay_ms=1)

    def test_an_immediate_deadlock_and_a_delayed_kill_are_accepted(self) -> None:
        """The deadlock delay restriction must not spill over to other actions."""
        assert isinstance(_request(action={"kind": "deadlock_thread"}).action, DeadlockThreadAction)
        assert _request(delay_ms=50).delay_ms == 50

    @pytest.mark.parametrize(
        "hook_name", [FaultHookName.TRAINER_CONTROLLER_STEP_END, FaultHookName.ORCHESTRATOR_STEP_END]
    )
    @pytest.mark.parametrize("target", [{"kind": "declared", "rank": 0}, _OBSERVED.model_dump(mode="json")])
    def test_a_hook_reached_outside_cells_cannot_name_a_target(
        self, hook_name: FaultHookName, target: dict[str, object]
    ) -> None:
        """Controller and orchestrator hooks must reject any non-wildcard target."""
        with pytest.raises(ValidationError, match="outside any cell"):
            _request(hook_name=hook_name, target=target)

    @pytest.mark.parametrize("hook_name", [None, FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE])
    def test_actor_and_immediate_requests_may_name_an_observed_target(self, hook_name: FaultHookName | None) -> None:
        """Requests reached inside a cell must keep their observed target."""
        assert _request(hook_name=hook_name, target=_OBSERVED.model_dump(mode="json")).target == _OBSERVED

    @pytest.mark.parametrize(
        "action",
        [
            {"kind": "observe"},
            {"kind": "stop_process"},
            {"kind": "freeze_process"},
            {"kind": "segfault_process"},
            {"kind": "exit_process"},
            {"kind": "start_cell", "cell_id": "c"},
            {"kind": "sleep_forever"},
        ],
    )
    def test_every_action_kind_round_trips_through_json(self, action: dict[str, object]) -> None:
        """Each member of the action union must deserialise back to the same action."""
        request = _request(action=action)
        assert FaultHookRequest.model_validate_json(request.model_dump_json()) == request
        assert request.action.kind == action["kind"]


class TestFaultHookRequestMatches:
    @pytest.mark.parametrize(
        "filters,context,expected",
        [
            ({}, FaultHookContext(), True),
            ({"rollout_id": 3}, FaultHookContext(rollout_id=3), True),
            ({"rollout_id": 3}, FaultHookContext(rollout_id=4), False),
            ({"rollout_id": 3}, FaultHookContext(), False),
            ({"attempt": 0}, FaultHookContext(rollout_id=3, attempt=1), False),
            ({"weight_version": 7}, FaultHookContext(weight_version=7), True),
            ({"weight_version": 7}, FaultHookContext(weight_version=8), False),
            ({"weight_version": 7}, FaultHookContext(), False),
            ({"rollout_id": 0}, FaultHookContext(rollout_id=0), True),
            ({"rollout_id": 0}, FaultHookContext(), False),
        ],
    )
    def test_each_named_filter_must_equal_the_reached_context(
        self, filters: dict[str, int], context: FaultHookContext, expected: bool
    ) -> None:
        """Rollout, attempt and weight version filters must match exactly, including zero."""
        request = FaultHookRequest(request_id="r", hook_name=_SEND, action=ObserveAction(), **filters)
        assert request.matches(context) is expected


class TestFaultHookRequestConflicts:
    def test_the_same_trigger_with_another_id_conflicts(self) -> None:
        """Two requests firing the same action at the same trigger must conflict."""
        assert _request(request_id="a").conflicts_with(_request(request_id="b"))

    @pytest.mark.parametrize(
        "change",
        [
            {"hook_name": FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE},
            {"action": {"kind": "stop_process"}},
            {"rollout_id": 4},
            {"rollout_id": None},
            {"attempt": 0},
            {"weight_version": 1},
        ],
    )
    def test_any_differing_trigger_field_does_not_conflict(self, change: dict[str, object]) -> None:
        """Requests that differ in hook, action, rollout, attempt or weight version must coexist."""
        assert not _request(request_id="a").conflicts_with(_request(request_id="b", **change))

    def test_immediate_requests_never_conflict(self) -> None:
        """Requests without a hook must never be refused as a duplicate trigger."""
        immediate = _request(hook_name=None, rollout_id=None)
        assert not immediate.conflicts_with(immediate.model_copy(update={"request_id": "b"}))


class TestFaultHookWireFormats:
    def test_a_command_round_trips_with_an_observed_target(self) -> None:
        """The HTTP command body must carry operation, action and observed identity intact."""
        command = FaultHookCommand(
            operation=FaultHookOperation.CLEAR,
            request=_request(hook_name=None, rollout_id=None, delay_ms=5, target=_OBSERVED.model_dump(mode="json")),
        )
        assert FaultHookCommand.model_validate_json(command.model_dump_json()) == command

    @pytest.mark.parametrize("status", list(FaultHookStatus))
    def test_a_fault_hook_event_round_trips_through_the_event_union(self, status: FaultHookStatus) -> None:
        """A logged record must parse back as a fault hook event with every field."""
        record = FaultHookRecord(
            request=_request(action=StopCellAction(cell_id="c").model_dump(), weight_version=2, delay_ms=10),
            status=status,
            set_at=1.0,
            changed_at=2.0,
            reached_at=1.5,
            due_at=1.51,
            context=FaultHookContext(rollout_id=3, weight_version=2, debug_weight_update_id="u"),
        )
        event = FaultHookEvent(
            timestamp="2026-09-26T00:00:00Z", source=SimpleProcessIdentity(component="main"), record=record
        )
        restored = TypeAdapter(Event).validate_json(event.model_dump_json())
        assert isinstance(restored, FaultHookEvent)
        assert restored.record == record

    def test_a_record_rejects_unknown_fields(self) -> None:
        """Records must stay strict so a renamed field cannot vanish silently."""
        with pytest.raises(ValidationError):
            FaultHookRecord.model_validate(
                {"request": _request().model_dump(), "status": "fired", "set_at": 0, "changed_at": 0, "extra": 1}
            )
