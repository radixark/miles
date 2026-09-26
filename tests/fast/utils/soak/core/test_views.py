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
    _weight_update_result,
)
from tests.utils.soak.core.events import SoakAdmissionClosedEvent, SoakEvent, SoakEvidenceArchivedEvent
from tests.utils.soak.core.views import (
    alive_targets_of_kind,
    compute_injection_times,
    compute_num_injections,
    compute_successful_form_names,
    event_source,
    is_normal_step,
    latest_observation,
    project_actions,
    quiescent_polls_of_type,
    tail_started_at,
    trainer_step_ends,
    weight_update_results,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import MetricEvent, WeightUpdateResultEvent


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
        assert compute_injection_times(events, kind="actor") == [_at(1)]

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

    def test_alive_targets_of_kind_skip_dead_and_other_kinds(self) -> None:
        """Only live targets of the asked kind are candidates."""
        alive = _cell_target()
        observation = _observation(
            [alive, _cell_target(cell_index=1, alive=False), _cell_target(kind="rollout")], at=_at(0)
        )

        assert alive_targets_of_kind(observation, "actor") == [alive]
        assert alive_targets_of_kind(_observation(None, at=_at(0)), "actor") == []


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


class TestEventSource:
    def test_an_archived_source_replaces_the_live_fallback(self, tmp_path: Path) -> None:
        """Checkers read the archived copy once one exists."""
        events = [_archived({"training_events": tmp_path / "archived"}, missing=[], at=0)]

        assert event_source(events, name="training_events", fallback=tmp_path / "live") == tmp_path / "archived"

    def test_the_latest_archive_wins(self, tmp_path: Path) -> None:
        """A later archive supersedes an earlier one."""
        events = [
            _archived({"training_events": tmp_path / "old"}, missing=[], at=0),
            _archived({"training_events": tmp_path / "new"}, missing=[], at=1),
        ]

        assert event_source(events, name="training_events", fallback=tmp_path) == tmp_path / "new"

    def test_a_source_archived_as_missing_is_refused(self, tmp_path: Path) -> None:
        """Checkers must not silently fall back to live data the archive says was missing."""
        events = [_archived({}, missing=["training_events"], at=0)]

        with pytest.raises(AssertionError, match="Missing archived soak evidence"):
            event_source(events, name="training_events", fallback=tmp_path)

    def test_without_an_archive_the_fallback_is_used(self, tmp_path: Path) -> None:
        """Before archiving checkers read the live source."""
        assert event_source([], name="training_events", fallback=tmp_path) == tmp_path


class TestWeightUpdateResults:
    _HASHES: dict[str, str] = {"rollout-0": "inc-0", "rollout-1": "inc-1"}

    def _result(self, **overrides: object) -> WeightUpdateResultEvent:
        values: dict[str, object] = dict(cell_hashes=self._HASHES, updated=["rollout-0"], failed=["rollout-1"])
        values.update(overrides)
        return _weight_update_result("update-1", at=_at(0), **values)

    def test_a_consistent_partial_result_is_returned_and_other_events_are_dropped(self) -> None:
        """Only result events come back, and a partition of the snapshot into updated and failed is valid."""
        result = self._result()

        assert weight_update_results([result, _step_end(1, at=_at(1))]) == [result]

    @pytest.mark.parametrize(
        "overrides,message",
        [
            (dict(updated=["rollout-0", "rollout-0"], failed=["rollout-1"]), "Repeated updated engine"),
            (dict(updated=["rollout-0", "rollout-1"], failed=["rollout-1"]), "also reported failed"),
            (dict(updated=["rollout-0"], failed=[]), "omits assigned targets"),
            (dict(updated=["rollout-0", "rollout-9"], failed=["rollout-1"]), "omits assigned targets"),
            (dict(cell_hashes={"rollout-0": "", "rollout-1": "inc-1"}), "lacks its incarnation"),
        ],
        ids=["repeated", "overlap", "omitted", "unassigned", "empty-hash"],
    )
    def test_an_inconsistent_result_is_rejected(self, overrides: dict[str, object], message: str) -> None:
        """A result whose cell sets contradict the snapshot cannot be used as evidence."""
        with pytest.raises(AssertionError, match=message):
            weight_update_results([self._result(**overrides)])

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(published_version=None),
            dict(updated=[], failed=["rollout-0", "rollout-1"], published_version=1),
            dict(published_version=2),
        ],
        ids=["updated-unpublished", "nothing-updated-but-published", "published-other-version"],
    )
    def test_a_published_version_disagreeing_with_the_updated_cells_is_rejected(
        self, overrides: dict[str, object]
    ) -> None:
        """Publication is exactly the candidate when some cell took it and nothing otherwise."""
        published = overrides.pop("published_version")
        result = self._result(**overrides).model_copy(update={"published_version": published})

        with pytest.raises(AssertionError, match="Published version is inconsistent"):
            weight_update_results([result])
