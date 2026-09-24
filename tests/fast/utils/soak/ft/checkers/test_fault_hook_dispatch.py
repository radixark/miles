import pytest
from tests.fast.utils.soak.soak_fakes import (
    _at,
    _cell_target,
    _fault_target,
    _hook_fault_applied,
    _hook_fault_request,
    _hook_record_event,
    _requested,
    _update_context,
    _weight_update_result,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakActionRequest
from tests.utils.soak.ft.checkers.fault_hook_dispatch import assert_hook_dispatches, assert_p2p_receiver_failures

from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookStatus

_TRAINER = _cell_target(kind="actor", cell_index=1, incarnation="trainer-inc")
_ROLLOUT = _cell_target(kind="rollout", cell_index=0, incarnation="rollout-inc")
_TRAINER_HOOK = _fault_target(_TRAINER.identity, workers_hash=_TRAINER.incarnation)


def _direct(request_id: str = "req-1") -> SoakActionRequest:
    return _hook_fault_request(_TRAINER, request_id=request_id)


def _through_trainer(request_id: str = "req-r") -> SoakActionRequest:
    return _hook_fault_request(_ROLLOUT, request_id=request_id, hook_target=_TRAINER_HOOK)


def _applied(request: SoakActionRequest) -> list[SoakEvent]:
    return [_requested(request, at=_at(0)), _hook_fault_applied(request, at=_at(20))]


def _fired(request: SoakActionRequest, **overrides: object) -> Event:
    return _hook_record_event(request, FaultHookStatus.FIRED, context=_update_context(), **overrides)


class TestAssertHookDispatches:
    def test_one_fired_dispatch_after_its_deadline_passes(self) -> None:
        """A SCHEDULED then FIRED record carrying the update is the healthy trace."""
        request = _direct()
        training = [_hook_record_event(request, FaultHookStatus.SCHEDULED, reached_at=10.0), _fired(request)]

        assert_hook_dispatches(_applied(request), training_events=training)

    def test_an_applied_hook_without_worker_evidence_fails(self) -> None:
        """An effect with no hook record may have come from anything but the hook."""
        with pytest.raises(AssertionError, match="no worker-side evidence"):
            assert_hook_dispatches(_applied(_direct()), training_events=[])

    @pytest.mark.parametrize("count", [0, 2])
    def test_anything_but_exactly_one_fired_dispatch_fails(self, count: int) -> None:
        """Zero dispatches means the effect was not the hook; two means the one-shot fault ran twice."""
        request = _direct()
        training = [_hook_record_event(request, FaultHookStatus.SCHEDULED), *[_fired(request)] * count]

        with pytest.raises(AssertionError, match="exactly one dispatch"):
            assert_hook_dispatches(_applied(request), training_events=training)

    @pytest.mark.parametrize(
        "overrides",
        [dict(hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER), dict(delay_ms=999.0)],
        ids=["hook", "delay"],
    )
    def test_evidence_with_other_fault_parameters_fails(self, overrides: dict[str, object]) -> None:
        """A record for another hook or delay is not the fault that was drawn."""
        request = _direct()

        with pytest.raises(AssertionError, match="mixes fault parameters"):
            assert_hook_dispatches(_applied(request), training_events=[_fired(request, **overrides)])

    def test_a_record_of_another_request_does_not_count(self) -> None:
        """Evidence is joined by request id, so another fault's dispatch proves nothing here."""
        request = _direct("req-1")

        with pytest.raises(AssertionError, match="no worker-side evidence"):
            assert_hook_dispatches(_applied(request), training_events=[_fired(_direct("req-2"))])

    @pytest.mark.parametrize("status", [FaultHookStatus.CLEARED, FaultHookStatus.EXPIRED, FaultHookStatus.FAILED])
    def test_a_fired_hook_that_also_ended_unfired_fails(self, status: FaultHookStatus) -> None:
        """A record that both fired and cleared, expired or failed is contradictory evidence."""
        request = _direct()
        training = [_fired(request), _hook_record_event(request, status, changed_at=11.0)]

        with pytest.raises(AssertionError, match="claims clearing, expiration or dispatch failure"):
            assert_hook_dispatches(_applied(request), training_events=training)

    def test_a_dispatch_without_its_weight_update_fails(self) -> None:
        """Without the update id the fault cannot be tied to the update it interrupted."""
        request = _direct()
        fired = _hook_record_event(request, FaultHookStatus.FIRED, context=None)

        with pytest.raises(AssertionError, match="lacks its exact update"):
            assert_hook_dispatches(_applied(request), training_events=[fired])

    @pytest.mark.parametrize("missing", ["reached_at", "due_at"])
    def test_a_dispatch_without_target_local_timing_fails(self, missing: str) -> None:
        """Without the worker's own clock readings the delay cannot be verified."""
        request = _direct()

        with pytest.raises(AssertionError, match="target-local timing"):
            assert_hook_dispatches(_applied(request), training_events=[_fired(request, **{missing: None})])

    def test_a_dispatch_before_its_deadline_fails(self) -> None:
        """Firing before due_at means the drawn delay was not honoured."""
        request = _direct()

        with pytest.raises(AssertionError, match="before its target-local deadline"):
            assert_hook_dispatches(_applied(request), training_events=[_fired(request, due_at=10.5, changed_at=10.4)])

    def test_a_dispatch_exactly_at_its_deadline_passes(self) -> None:
        """The deadline itself is a legal firing time."""
        request = _direct()

        assert_hook_dispatches(_applied(request), training_events=[_fired(request, due_at=10.5, changed_at=10.5)])

    def test_an_effect_on_another_target_fails(self) -> None:
        """An effect observed on a different worker cannot confirm this hook."""
        request = _direct()
        events = [
            _requested(request, at=_at(0)),
            _hook_fault_applied(request, at=_at(20), target=_TRAINER_HOOK.model_copy(update={"rank": 1})),
        ]

        with pytest.raises(AssertionError, match="another target"):
            assert_hook_dispatches(events, training_events=[_fired(request)])

    def test_unapplied_timer_and_hookless_requests_are_not_checked(self) -> None:
        """Only applied hook faults need dispatch evidence, but at least one must be confirmed."""
        unapplied = _direct("req-unapplied")
        timer = _hook_fault_request(_TRAINER, request_id="req-timer", hook_name=None)
        events = [_requested(unapplied, at=_at(0)), *_applied(timer)]

        with pytest.raises(AssertionError, match="No hook-triggered fault was confirmed"):
            assert_hook_dispatches(events, training_events=[])


class TestAssertP2PReceiverFailures:
    @staticmethod
    def _result(**overrides: object) -> Event:
        values: dict[str, object] = dict(
            cell_hashes={_ROLLOUT.identity: _ROLLOUT.incarnation, "rollout-other": "other-inc"},
            updated=["rollout-other"],
            failed=[_ROLLOUT.identity],
            candidate_version=7,
        )
        values.update(overrides)
        return _weight_update_result("update-7", at=_at(15), **values)

    def test_the_receiver_failed_in_the_exact_update_its_hook_fired_in(self) -> None:
        """The rollout target is in the failed cells of the update named by the dispatch."""
        request = _through_trainer()

        assert_p2p_receiver_failures(_applied(request), training_events=[_fired(request), self._result()])

    def test_a_receiver_that_survived_the_update_fails(self) -> None:
        """If the receiver took the update the fault did not hit it during the transfer."""
        request = _through_trainer()
        result = self._result(updated=[_ROLLOUT.identity, "rollout-other"], failed=[])

        with pytest.raises(AssertionError, match="Receiver survived"):
            assert_p2p_receiver_failures(_applied(request), training_events=[_fired(request), result])

    def test_an_update_that_targeted_another_incarnation_fails(self) -> None:
        """A failure of the cell's previous incarnation is not this fault's effect."""
        request = _through_trainer()
        result = self._result(cell_hashes={_ROLLOUT.identity: "older-inc", "rollout-other": "other-inc"})

        with pytest.raises(AssertionError, match="another incarnation"):
            assert_p2p_receiver_failures(_applied(request), training_events=[_fired(request), result])

    def test_an_update_whose_candidate_differs_from_the_dispatch_fails(self) -> None:
        """The dispatch and the result must describe the same candidate version."""
        request = _through_trainer()

        with pytest.raises(AssertionError, match="changed its candidate version"):
            assert_p2p_receiver_failures(
                _applied(request),
                training_events=[
                    _fired(request),
                    self._result(candidate_version=8, updated=[], failed=[_ROLLOUT.identity, "rollout-other"]),
                ],
            )

    def test_a_dispatch_without_its_update_fails(self) -> None:
        """Without the update id there is no update to check the receiver against."""
        request = _through_trainer()
        fired = _hook_record_event(request, FaultHookStatus.FIRED, context=None)

        with pytest.raises(AssertionError, match="lacks its exact update"):
            assert_p2p_receiver_failures(_applied(request), training_events=[fired, self._result()])

    def test_a_dispatch_naming_an_unlogged_update_fails(self) -> None:
        """A dispatch in an update that never logged a result has no evidence to match."""
        request = _through_trainer()

        with pytest.raises(KeyError):
            assert_p2p_receiver_failures(
                _applied(request), training_events=[_fired(request, context=_update_context("update-missing"))]
            )

    def test_direct_trainer_faults_are_not_receiver_faults(self) -> None:
        """A fault on the hook's own worker says nothing about a receiver, and one receiver must be checked."""
        request = _direct()

        with pytest.raises(AssertionError, match="No receiver fault was checked"):
            assert_p2p_receiver_failures(_applied(request), training_events=[_fired(request), self._result()])
