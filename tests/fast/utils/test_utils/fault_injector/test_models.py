import pytest
from pydantic import ValidationError

from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.actions.process import ObserveAction
from miles.utils.test_utils.fault_injector.models import (
    DeclaredFaultHookTarget,
    FaultHookName,
    FaultHookOwner,
    FaultHookRecord,
    FaultHookRequest,
)

_SEND = FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND


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

    def test_an_omitted_target_is_the_declared_wildcard(self) -> None:
        """A request without a target must be a declared wildcard, not an observed one."""
        assert _request().target == DeclaredFaultHookTarget()


class TestFaultHookRequestValidation:
    @pytest.mark.parametrize(
        "change",
        [
            {"attempt": -1},
            {"weight_version": -1},
            {"hook_name": "not_a_hook"},
        ],
    )
    def test_out_of_range_timing_and_filters_are_rejected(self, change: dict[str, object]) -> None:
        """Filters and timing fields outside their bounds must fail validation."""
        with pytest.raises(ValidationError):
            _request(**change)

    @pytest.mark.parametrize(
        "action",
        [
            {"kind": "observe"},
            {"kind": "segfault_process"},
            {"kind": "exit_process"},
            {"kind": "start_cell", "cell_id": "c"},
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


class TestFaultHookWireFormats:
    def test_a_record_rejects_unknown_fields(self) -> None:
        """Records must stay strict so a renamed field cannot vanish silently."""
        with pytest.raises(ValidationError):
            FaultHookRecord.model_validate(
                {"request": _request().model_dump(), "status": "fired", "set_at": 0, "changed_at": 0, "extra": 1}
            )
