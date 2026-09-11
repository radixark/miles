import random

import pytest
from tests.fast.utils.soak.utils import AsyncStubFaultForm, typed_cell
from tests.utils.soak.config import SoakCellPolicy, SoakPolicy
from tests.utils.soak.core import SoakActionScheduler
from tests.utils.soak.state import (
    EventLog,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionResultEvent,
    SoakObservation,
    SoakScheduleEvent,
)


class TestScheduler:
    @pytest.mark.parametrize(
        "scheduled, actor_count, rollout_count, expected",
        [
            (("rollout",), 2, 2, "rollout"),
            (("rollout",), 2, 1, None),
            (("actor", "rollout"), 2, 1, "actor"),
            (("actor", "rollout"), 1, 2, "rollout"),
            (("actor", "rollout"), 1, 1, None),
        ],
    )
    def test_target_selection_keeps_survivors_separate_for_each_cell_kind(
        self, scheduled: tuple[str, ...], actor_count: int, rollout_count: int, expected: str | None
    ) -> None:
        """Another kind's replicas cannot authorize crashing a kind's last healthy cell."""
        forms = {kind: [AsyncStubFaultForm(name="fault", execute=_effect)] for kind in scheduled}
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals=dict.fromkeys(scheduled, 1), forms=forms)
        cells = [typed_cell(f"actor-{i}", "actor") for i in range(actor_count)] + [
            typed_cell(f"rollout-{i}", "rollout") for i in range(rollout_count)
        ]
        events = [SoakScheduleEvent(due_of_type=dict.fromkeys(scheduled, 0)), SoakObservation(cells=cells)]

        request = scheduler.choose(events=events, now=10)

        if expected is None:
            assert request is None
        else:
            assert request is not None
            assert request.target["metadata"]["labels"]["miles.io/cell-type"] == expected

    def test_a_mixed_run_eventually_selects_both_cell_kinds(self) -> None:
        """Each configured kind remains reachable through the shared scheduler."""
        forms = {kind: [AsyncStubFaultForm(name="fault", execute=_effect)] for kind in ("actor", "rollout")}
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1, "rollout": 1}, forms=forms)
        log = EventLog()
        log.note_schedule(SoakScheduleEvent(due_of_type={"actor": 0, "rollout": 0}))
        log.note_observation(
            SoakObservation(cells=[typed_cell(f"{kind}-{i}", kind) for kind in forms for i in range(3)])
        )
        selected = set()
        for now in range(1, 21):
            request = scheduler.choose(events=log.events, now=now * 100)
            if request is None:
                continue
            selected.add(request.target["metadata"]["labels"]["miles.io/cell-type"])
            log.note_action_requested(request)
            log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))
        assert selected == {"actor", "rollout"}

    def test_recorded_deadlines_survive_rebuilding_the_scheduler(self) -> None:
        """Reconstruction cannot redraw a recorded deadline or mutate its observation."""
        forms = {"actor": [AsyncStubFaultForm(name="fault", execute=_effect)]}
        log = EventLog()
        cells = [typed_cell(f"actor-{i}", "actor") for i in range(3)]
        log.note_observation(SoakObservation(cells=cells))
        cells.clear()
        log.note_schedule(SoakScheduleEvent(due_of_type={"actor": 10}))
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms)
        assert scheduler.choose(events=log.events, now=9) is None
        request = scheduler.choose(events=log.events, now=10)
        assert request is not None and request.next_due_at > 10
        log.note_action_requested(request)
        log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=False, error="refused"))
        rebuilt = SoakActionScheduler(rng=random.Random(99), mean_intervals={"actor": 1}, forms=forms)
        assert rebuilt.choose(events=log.events, now=10) is None
        assert rebuilt.choose(events=log.events, now=request.next_due_at) is not None

    @pytest.mark.parametrize("returned", [False, True])
    def test_each_attempt_reschedules_even_when_the_response_fails(self, returned: bool) -> None:
        """A failed request must not become an overdue retry on every observation."""
        forms = {"actor": [AsyncStubFaultForm(name="fault", execute=_effect)]}
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 100}, forms=forms)
        log = EventLog()
        log.note_schedule(SoakScheduleEvent(due_of_type={"actor": 0}))
        log.note_observation(SoakObservation(cells=[typed_cell(f"actor-{i}", "actor") for i in range(3)]))
        request = scheduler.choose(events=log.events, now=10)
        assert request is not None and request.next_due_at > 10
        log.note_action_requested(request)
        log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=returned))
        assert scheduler.choose(events=log.events, now=10) is None
        assert scheduler.choose(events=log.events, now=request.next_due_at) is not None

    def test_unproven_forms_are_selected_before_a_proven_form_repeats(self) -> None:
        """Every configured form must get an opportunity before successful forms consume more coverage."""
        forms = {"actor": [AsyncStubFaultForm(name=name, execute=_effect) for name in ("a", "b", "c")]}
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms)
        log = EventLog()
        log.note_schedule(SoakScheduleEvent(due_of_type={"actor": 0}))
        log.note_observation(SoakObservation(cells=[typed_cell(f"actor-{i}", "actor") for i in range(4)]))
        drawn = []
        for now in (100, 200, 300):
            request = scheduler.choose(events=log.events, now=now)
            assert request is not None
            drawn.append(request.form_name)
            log.note_action_requested(request)
            log.note_action_applied(SoakActionAppliedEvent(request_id=request.request_id, evidence={"applied": True}))
            log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))
        assert set(drawn) == {"a", "b", "c"}

    def test_a_return_without_effect_does_not_satisfy_form_coverage(self) -> None:
        """A command returning successfully cannot displace an unproven fault from the coverage queue."""
        forms = {"actor": [AsyncStubFaultForm(name=name, execute=_effect) for name in ("works", "refused")]}
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms)
        log = EventLog()
        log.note_schedule(SoakScheduleEvent(due_of_type={"actor": 0}))
        cells = [typed_cell(f"actor-{i}", "actor") for i in range(4)]
        log.note_observation(SoakObservation(cells=cells))
        successful = SoakActionRequest(target=cells[0], form_name="works", harms_cell=True)
        log.note_action_requested(successful)
        log.note_action_applied(SoakActionAppliedEvent(request_id=successful.request_id, evidence={"applied": True}))
        log.note_action_result(SoakActionResultEvent(request_id=successful.request_id, returned=True))
        for now in (100, 200):
            request = scheduler.choose(events=log.events, now=now)
            assert request is not None and request.form_name == "refused"
            log.note_action_requested(request)
            log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))

    @pytest.mark.parametrize("healthy, serving", [(True, True), (False, True), (True, False), (False, False)])
    def test_explicit_zero_survivors_allows_faults_during_updates_or_recovery(
        self, healthy: bool, serving: bool
    ) -> None:
        """Readiness does not veto a fault when the configured policy permits zero healthy survivors."""
        forms = {"rollout": [AsyncStubFaultForm(name="fault", execute=_effect)]}
        scheduler = SoakActionScheduler(
            rng=random.Random(0),
            mean_intervals={"rollout": 1},
            forms=forms,
            policy=SoakPolicy(cell_policies={"rollout": SoakCellPolicy(min_survivors=0)}),
        )
        events = [
            SoakScheduleEvent(due_of_type={"rollout": 0}),
            SoakObservation(cells=[typed_cell("rollout-0", "rollout", healthy=healthy, serving=serving)]),
        ]
        assert scheduler.choose(events=events, now=10) is not None

    def test_closed_admission_cannot_be_reopened_by_another_observation(self) -> None:
        """A recovery tail keeps collecting observations without admitting another fault."""
        forms = {"actor": [AsyncStubFaultForm(name="fault", execute=_effect)]}
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"actor": 1}, forms=forms)
        log = EventLog()
        log.note_schedule(SoakScheduleEvent(due_of_type={"actor": 0}))
        log.close_admission()
        log.note_observation(SoakObservation(cells=[typed_cell(f"actor-{i}", "actor") for i in range(2)]))
        assert scheduler.choose(events=log.events, now=100) is None


async def _effect(request: SoakActionRequest) -> dict:
    return {"request_id": request.request_id}
