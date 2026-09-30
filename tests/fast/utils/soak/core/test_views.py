from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import (
    _applied,
    _at,
    _cell_target,
    _observation,
    _request,
    _requested,
    _result,
    _step_end,
    _sut_main_source,
)
from tests.utils.soak.core.events import SoakAdmissionClosedEvent, SoakEvent, SoakEvidenceArchivedEvent
from tests.utils.soak.core.views import (
    compute_num_injections,
    compute_successful_form_names,
    is_normal_step,
    latest_observation,
    project_actions,
    quiescent_polls_of_type,
    tail_started_at,
    trainer_step_ends,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import MetricEvent


def _archived(sources: dict[str, Path], *, missing: list[str], at: float) -> SoakEvidenceArchivedEvent:
    return SoakEvidenceArchivedEvent(timestamp=_at(at), sources=sources, missing_sources=missing, sha256_of_file={})


class TestProjectActions:
    def test_requested_applied_and_result_are_joined_by_request_id(self) -> None:
        """Each request collects its own effect and result, never another request's."""
        first = _request(_cell_target(), request_id="first")
        second = _request(_cell_target(cell_index=1), request_id="second")
        events = [
            _requested(first, at=_at(0)),
            _requested(second, at=_at(1)),
            _applied(second, at=_at(2)),
            _result(first, at=_at(3), returned=False),
        ]

        actions = project_actions(events)

        assert list(actions) == ["first", "second"]
        assert (actions["first"].applied, actions["first"].result.returned) == (None, False)
        assert (actions["second"].applied.request_id, actions["second"].result) == ("second", None)

    def test_an_effect_without_a_request_is_not_an_action(self) -> None:
        """Applied or result events of unknown requests do not create actions."""
        stray = _request(_cell_target(), request_id="stray")

        assert project_actions([_applied(stray, at=_at(0)), _result(stray, at=_at(1))]) == {}


class TestInjectionCounts:
    def _events(self) -> list[SoakEvent]:
        landed = _request(_cell_target(), form_name="kill", request_id="landed")
        refused = _request(_cell_target(), form_name="delete", request_id="refused")
        rollout = _request(_cell_target(kind="rollout"), form_name="kill", request_id="rollout")
        return [
            _requested(landed, at=_at(0)),
            _applied(landed, at=_at(1)),
            _requested(refused, at=_at(2)),
            _result(refused, at=_at(3)),
            _requested(rollout, at=_at(4)),
            _applied(rollout, at=_at(5)),
        ]

    def test_only_applied_actions_of_the_kind_are_counted(self) -> None:
        """Requests that never landed and actions of other kinds are not injections."""
        events = self._events()

        assert compute_num_injections(events, kind="actor") == 1
        assert compute_num_injections(events) == 2

    def test_only_applied_forms_have_worked(self) -> None:
        """A form proves itself only through an applied effect of its own kind."""
        assert compute_successful_form_names(self._events(), kind="actor") == {"kill"}


class TestQuiescentPollsOfType:
    def test_consecutive_complete_live_polls_are_counted_per_kind(self) -> None:
        """Each kind counts its own run of polls with every expected target alive."""
        full = [_cell_target(), _cell_target(kind="rollout")]
        events = [
            _observation(full, at=_at(0)),
            _observation([_cell_target()], at=_at(1)),
            _observation(full, at=_at(2)),
        ]

        assert quiescent_polls_of_type(events, expected_count_of_kind={"actor": 1, "rollout": 1}) == {
            "actor": 3,
            "rollout": 1,
        }

    def test_a_request_resets_only_its_own_kind(self) -> None:
        """Requesting a fault on one kind leaves another kind's quiescence intact."""
        full = [_cell_target(), _cell_target(kind="rollout")]
        request = _request(_cell_target(kind="rollout"))
        events = [_observation(full, at=_at(0)), _requested(request, at=_at(1)), _observation(full, at=_at(2))]

        assert quiescent_polls_of_type(events, expected_count_of_kind={"actor": 1, "rollout": 1}) == {
            "actor": 2,
            "rollout": 1,
        }

    def test_extra_targets_are_not_quiescent(self) -> None:
        """More targets than expected means a replacement is still around."""
        events = [_observation([_cell_target(), _cell_target(cell_index=1)], at=_at(0))]

        assert quiescent_polls_of_type(events, expected_count_of_kind={"actor": 1}) == {"actor": 0}


class TestTrainerStepEnds:
    def test_only_actor_trainer_controller_steps_are_progress(self) -> None:
        """Steps of other trainers or processes do not count as actor training progress."""
        actor = _step_end(1, at=_at(0))
        critic = _step_end(2, at=_at(0), trainer_id="critic")
        metric = MetricEvent(timestamp=_at(0), source=_sut_main_source(), metrics={})
        events = [_observation(None, at=_at(0), new_sut_events=[metric, critic, actor])]

        assert trainer_step_ends(events) == [actor]

    @pytest.mark.parametrize(
        ("outcomes", "normal"),
        [
            ({0: [TrainStepOutcome.NORMAL]}, True),
            ({0: "error", 1: [TrainStepOutcome.NORMAL]}, True),
            ({0: "error"}, False),
            ({0: [TrainStepOutcome.DISCARDED_SHOULD_RETRY]}, False),
            ({}, False),
        ],
    )
    def test_a_step_is_normal_when_any_cell_reports_a_normal_outcome(self, outcomes: dict, normal: bool) -> None:
        """Errors and retries alone do not make a normal step."""
        step = _step_end(0, at=_at(0)).model_copy(update={"cell_outcomes": outcomes})

        assert is_normal_step(step) is normal


class TestObservationViews:
    def test_the_latest_observation_is_the_last_one_recorded(self) -> None:
        """Later non-observation events do not hide the latest observation."""
        last = _observation([], at=_at(1))
        events = [_observation(None, at=_at(0)), last, SoakAdmissionClosedEvent(timestamp=_at(2))]

        assert latest_observation(events) is last
        assert latest_observation([]) is None


class TestTailStartedAt:
    def test_the_tail_starts_at_the_later_of_closure_and_the_last_effect(self) -> None:
        """A fault landing after closure delays the start of the tail."""
        request = _request(_cell_target())
        events = [
            _requested(request, at=_at(0)),
            SoakAdmissionClosedEvent(timestamp=_at(1)),
            _applied(request, at=_at(5)),
        ]

        assert tail_started_at(events) == _at(5)

    def test_without_closure_there_is_no_tail(self) -> None:
        """Asking for the tail of an open run is an error."""
        with pytest.raises(AssertionError, match="never closed"):
            tail_started_at([])
