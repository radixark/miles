from datetime import datetime, timedelta, timezone

from tests.e2e.ft.conftest_ft.fault_injection import state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    PENDING,
    RUNNING_NOT_SERVING,
    SERVING,
    SUSPENDED,
    log_of,
    note_injected,
    staged,
)


def test_observed_states_record_only_transitions() -> None:
    """Polling runs for the life of the training run, so repeats must not accumulate."""
    log = log_of([SERVING, SERVING, SUSPENDED, SUSPENDED, SERVING])

    assert views.compute_states_of_cell_name(log.events) == {"rollout-engine-0": [SERVING, SUSPENDED, SERVING]}


def test_a_serve_after_the_last_injection_clears_the_cell() -> None:
    """This is the recovery the soak asserts: the injected engine ends up serving again."""
    log = log_of([SERVING, PENDING, SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_a_completely_missed_down_window_is_not_an_offence() -> None:
    """A replacement can finish between two polls, and the witness must not demand the down it never saw."""
    log = log_of([SERVING, SERVING, SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_a_replacement_that_never_reaches_the_router_is_an_offence() -> None:
    """Regression: a relaunched engine stuck at PendingWeights also reads Running, and must not pass."""
    log = log_of([SERVING, PENDING, RUNNING_NOT_SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {
        "rollout-engine-0": [SERVING.value, PENDING.value, RUNNING_NOT_SERVING.value]
    }


def test_a_cell_that_was_never_injected_owes_no_serve() -> None:
    """Otherwise a run that injected nothing would still fail the witness on ordinary churn."""
    log = log_of([PENDING, SERVING])

    assert views.compute_num_injections(log.events, cell_type="rollout") == 0
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_a_serve_that_predates_the_last_injection_does_not_discharge_it() -> None:
    """Otherwise the last crash of a soak is paid for by the recovery of the crash before it."""
    log = log_of([SERVING, PENDING, SERVING, PENDING], inject_before={1: 1, 3: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {
        "rollout-engine-0": [SERVING.value, PENDING.value, SERVING.value, PENDING.value]
    }


def test_only_the_last_injection_of_a_cell_needs_a_serve_after_it() -> None:
    """Injections are serialized by the quiescence gate, so one final fresh serve settles the whole cell."""
    log = log_of([SERVING, PENDING, SERVING, SERVING], inject_before={1: 1, 3: 1})

    assert views.compute_num_injections(log.events, cell_type="rollout") == 2
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_offences_of_another_cell_kind_do_not_count() -> None:
    """A mixed soak injects both kinds, and the rollout view must only see rollout cells."""
    log = state.EventLog()
    log.observe([staged("actor-0", SERVING, cell_type="actor")])
    note_injected(log, "actor-0")
    log.observe([staged("actor-0", PENDING, cell_type="actor")])

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="actor", grace_seconds=0.0) == {
        "actor-0": [SERVING.value, PENDING.value]
    }


def test_a_siblings_serve_cannot_clear_the_injected_cells_debt() -> None:
    """Only the victim's own fresh Serving reading proves its recovery."""
    log = state.EventLog()
    log.observe([staged("rollout-engine-0", SERVING), staged("rollout-engine-1", SERVING)])
    note_injected(log, "rollout-engine-0")
    log.observe([staged("rollout-engine-0", PENDING), staged("rollout-engine-1", SERVING)])

    offenders = views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0)
    assert set(offenders) == {"rollout-engine-0"}


class TestStaleServingGrace:
    def test_a_serve_inside_the_stale_window_does_not_clear_the_cell(self) -> None:
        """The api server reports a just-killed cell Serving for ~95s, so an early serve proves nothing."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=30)),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }

    def test_a_serve_exactly_at_the_grace_bound_clears_the_cell(self) -> None:
        """The witness reads >= 120s as fresh, so a regression to a strict comparison must show here."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=120)),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {}

    def test_a_serve_reported_by_a_cell_that_is_not_healthy_does_not_clear_it(self) -> None:
        """A cell that died without being deregistered keeps reporting Serving, and heals nothing."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=130), alive=False),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }

    def test_a_serve_past_the_stale_window_clears_the_cell(self) -> None:
        """A reading older than the staleness bound is fresh, and it is the one that proves recovery."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=130)),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {}


def _observation(
    name: str, cell_state: state.ObservedCellState, *, at: datetime, alive: bool = True
) -> state.ObservationsEvent:
    return state.ObservationsEvent(
        timestamp=at, cell_infos={name: state.CellInfo(cell_type="rollout", state=cell_state, alive=alive)}
    )


def _injection(name: str, *, at: datetime) -> state.InjectionEvent:
    return state.InjectionEvent(
        timestamp=at, cell_name=name, form_name="inject_fault:sigkill", succeeded=True, harmed=True
    )


class TestWhichInjectionsCount:
    def test_a_form_that_left_its_cell_running_is_not_a_crash_anything_has_to_heal(self) -> None:
        """A hot restart replaces the orchestration script and harms no cell, so no cell owes a recovery."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=True, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout") == 0
        assert (
            views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}
        )

    def test_a_form_that_left_its_cell_running_is_still_a_draw_that_fired(self) -> None:
        """A soak counting what it actually did to the run has to see it, and asks for it by name."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=True, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 1

    def test_a_form_that_crashed_its_cell_counts_as_both(self) -> None:
        """The crash forms every floor assertion was written for must go on counting as they did."""
        log = _log_of_one_injection(form_name="crash_pod", succeeded=True, harmed=True)

        assert views.compute_num_injections(log.events, cell_type="rollout") == 1
        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 1

    def test_a_draw_that_never_landed_counts_as_neither(self) -> None:
        """An attempt the cluster refused did nothing to the run, whatever the form would have done."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=False, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 0

    def test_the_successes_of_one_form_are_counted_apart_from_another(self) -> None:
        """A mixed soak draws several forms, and each one's own assertions count only its own draws."""
        log = state.EventLog()
        log.observe([staged("rollout-engine-0", SERVING)])
        log.note_injection_attempt(cell_name="rollout-engine-0", form_name="hot_restart", succeeded=True, harmed=False)
        log.note_injection_attempt(cell_name="rollout-engine-0", form_name="crash_pod", succeeded=True, harmed=True)
        log.note_injection_attempt(
            cell_name="rollout-engine-0", form_name="hot_restart", succeeded=False, harmed=False
        )

        assert views.compute_num_successful_injections_of_form(log.events, form_name="hot_restart") == 1
        assert views.compute_num_successful_injections_of_form(log.events, form_name="crash_pod") == 1


def _log_of_one_injection(*, form_name: str, succeeded: bool, harmed: bool) -> state.EventLog:
    log = state.EventLog()
    log.observe([staged("rollout-engine-0", SERVING)])
    log.note_injection_attempt(cell_name="rollout-engine-0", form_name=form_name, succeeded=succeeded, harmed=harmed)
    log.observe([staged("rollout-engine-0", SERVING)])
    return log
