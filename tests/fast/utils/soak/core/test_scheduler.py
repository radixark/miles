import pytest
from tests.fast.utils.soak.soak_fakes import (
    _applied,
    _at,
    _cell_target,
    _FakeForm,
    _observation,
    _request,
    _requested,
    _result,
    _step_end,
)
from tests.utils.soak.core import scheduler as scheduler_module
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTargetConfig
from tests.utils.soak.core.events import SoakAdmissionClosedEvent, SoakEvent
from tests.utils.soak.core.scheduler import SoakActionScheduler
from tests.utils.soak.core.types import SoakForms, SoakTarget

from miles.backends.megatron_utils.ft.types import TrainStepOutcome

_LATER = 1e12


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: 0.0)


def _scheduler(
    forms: SoakForms,
    *,
    seed: int = 0,
    expected_count: int = 2,
    quiescent_polls_required: int = 1,
    start_after_rollout_id: int | None = None,
) -> SoakActionScheduler:
    return SoakActionScheduler(
        forms=forms,
        config=SoakRunnerConfig(
            seed=seed,
            start_after_rollout_id=start_after_rollout_id,
            quiescent_polls_required=quiescent_polls_required,
            target_configs={
                kind: SoakTargetConfig(expected_count=expected_count, mean_interval_seconds=10.0) for kind in forms
            },
        ),
    )


def _actors(count: int = 2, **kwargs: object) -> list[SoakTarget]:
    return [_cell_target(cell_index=index, **kwargs) for index in range(count)]


def _polls(targets: list[SoakTarget], *, count: int = 1, start: float = 0) -> list[SoakEvent]:
    return [_observation(targets, at=_at(start + index)) for index in range(count)]


class TestSchedulerDraws:
    def test_the_same_seed_draws_the_same_due_times_and_requests(self) -> None:
        """A soak is reproducible from its seed."""
        forms_a = {"actor": [_FakeForm(name="a"), _FakeForm(name="b")]}
        forms_b = {"actor": [_FakeForm(name="a"), _FakeForm(name="b")]}
        events = _polls(_actors(3), count=1)
        first, second = _scheduler(forms_a, seed=5, expected_count=3), _scheduler(forms_b, seed=5, expected_count=3)

        assert first.due_of_type == second.due_of_type
        one, other = first.choose(events=events, now=_LATER), second.choose(events=events, now=_LATER)
        assert (one.form_name, one.target) == (other.form_name, other.target)

    def test_different_seeds_draw_different_due_times(self) -> None:
        """The seed actually feeds the exponential draw."""
        forms = {"actor": [_FakeForm()]}

        assert _scheduler(forms, seed=1).due_of_type != _scheduler(forms, seed=2).due_of_type

    def test_a_kind_is_not_chosen_before_its_due_time(self) -> None:
        """Nothing is injected until the drawn interval elapses."""
        scheduler = _scheduler({"actor": [_FakeForm()]})
        due = scheduler.due_of_type["actor"]
        events = _polls(_actors())

        assert due > 0
        assert scheduler.choose(events=events, now=due * 0.999) is None
        assert scheduler.choose(events=events, now=due) is not None

    def test_each_kind_follows_its_own_clock(self) -> None:
        """Only the kind whose own due time passed is chosen."""
        scheduler = _scheduler({"actor": [_FakeForm()], "rollout": [_FakeForm()]})
        earlier = min(scheduler.due_of_type, key=scheduler.due_of_type.__getitem__)
        events = _polls([*_actors(), *(_cell_target(kind="rollout", cell_index=index) for index in range(2))])

        request = scheduler.choose(events=events, now=scheduler.due_of_type[earlier])

        assert request.target.kind == earlier

    def test_a_request_does_not_redraw_until_its_effect_is_applied(self) -> None:
        """Requesting alone keeps the due time; the applied event redraws it exactly once."""
        form = _FakeForm(recovered=False)
        scheduler = _scheduler({"actor": [form]})
        events = _polls(_actors())
        due = scheduler.due_of_type["actor"]
        request = scheduler.choose(events=events, now=_LATER)
        events += [_requested(request, at=_at(10)), _result(request, at=_at(11))]

        scheduler.choose(events=events, now=_LATER)
        assert scheduler.due_of_type["actor"] == due

        events.append(_applied(request, at=_at(12)))
        scheduler.choose(events=events, now=_LATER)
        redrawn = scheduler.due_of_type["actor"]
        scheduler.choose(events=events, now=2 * _LATER)

        assert redrawn > _LATER
        assert scheduler.due_of_type["actor"] == redrawn
        assert scheduler.awaiting_applied == {}

    def test_an_applied_action_of_one_kind_does_not_redraw_another(self) -> None:
        """Redrawing is per kind, so one kind's effect cannot reset another kind's clock."""
        scheduler = _scheduler({"actor": [_FakeForm(recovered=False)], "rollout": [_FakeForm()]})
        rollout_due = scheduler.due_of_type["rollout"]
        events = _polls(_actors())
        request = scheduler.choose(events=events, now=_LATER)
        events += [_requested(request, at=_at(10)), _applied(request, at=_at(11))]

        scheduler.choose(events=events, now=_LATER)

        assert request.target.kind == "actor"
        assert scheduler.due_of_type["rollout"] == rollout_due

    def test_a_form_that_declines_leaves_nothing_awaiting(self) -> None:
        """No request means nothing to wait for and no chosen action."""
        scheduler = _scheduler({"actor": [_FakeForm(creates_request=False)]})

        assert scheduler.choose(events=_polls(_actors()), now=_LATER) is None
        assert scheduler.awaiting_applied == {}


class TestSchedulerGates:
    def test_no_request_after_admission_closed(self) -> None:
        """Closing admission stops injection even when a kind is due and quiescent."""
        scheduler = _scheduler({"actor": [_FakeForm()]})
        events = [*_polls(_actors()), SoakAdmissionClosedEvent(timestamp=_at(5))]

        assert scheduler.choose(events=events, now=_LATER) is None

    def test_injection_waits_for_a_normal_step_at_the_start_rollout(self) -> None:
        """Faults begin only after training normally finished the configured rollout."""
        scheduler = _scheduler({"actor": [_FakeForm()]}, start_after_rollout_id=3)
        early = _observation(None, at=_at(0), new_sut_events=[_step_end(2, at=_at(0))])
        retried = _observation(
            None,
            at=_at(1),
            new_sut_events=[_step_end(3, at=_at(1), outcomes=[TrainStepOutcome.DISCARDED_SHOULD_RETRY])],
        )
        normal = _observation(None, at=_at(2), new_sut_events=[_step_end(3, at=_at(2))])

        assert scheduler.choose(events=[early, retried, *_polls(_actors(), start=3)], now=_LATER) is None
        assert scheduler.choose(events=[early, retried, normal, *_polls(_actors(), start=3)], now=_LATER) is not None

    def test_an_unrecovered_action_blocks_the_next_injection(self) -> None:
        """Only one outstanding fault at a time: a new one waits for the last to recover."""
        form = _FakeForm(recovered=False)
        scheduler = _scheduler({"actor": [form]})
        earlier = _request(_actors()[0], form_name="fake", request_id="earlier")
        events = [
            _requested(earlier, at=_at(0)),
            _applied(earlier, at=_at(1)),
            _result(earlier, at=_at(2)),
            *_polls(_actors(), start=3),
        ]

        assert scheduler.choose(events=events, now=_LATER) is None
        form.recovered = True
        assert scheduler.choose(events=events, now=_LATER) is not None

    def test_a_failed_action_stops_the_soak(self) -> None:
        """A form that could not execute its fault fails loudly instead of being retried silently."""
        scheduler = _scheduler({"actor": [_FakeForm()]})
        earlier = _request(_actors()[0], form_name="fake", request_id="earlier")
        events = [_requested(earlier, at=_at(0)), _result(earlier, at=_at(1), returned=False, error="refused")]

        with pytest.raises(RuntimeError, match="refused"):
            scheduler.choose(events=events, now=_LATER)

    @pytest.mark.parametrize(
        "latest",
        [
            _observation(_actors(), at=_at(9), errors={"pods": "down"}),
            _observation(None, at=_at(9), errors={"observation": "timeout"}),
        ],
    )
    def test_an_observation_with_errors_blocks_injection(self, latest: SoakEvent) -> None:
        """A partial view of the cluster is no basis for choosing a target."""
        scheduler = _scheduler({"actor": [_FakeForm()]})

        assert scheduler.choose(events=[*_polls(_actors(), count=3), latest], now=_LATER) is None

    def test_no_observation_blocks_injection(self) -> None:
        """Nothing is chosen before the first observation."""
        assert _scheduler({"actor": [_FakeForm()]}).choose(events=[], now=_LATER) is None

    def test_injection_waits_for_the_required_consecutive_quiescent_polls(self) -> None:
        """A kind must look settled for the configured number of consecutive polls."""
        scheduler = _scheduler({"actor": [_FakeForm()]}, quiescent_polls_required=3)

        assert scheduler.choose(events=_polls(_actors(), count=2), now=_LATER) is None
        assert scheduler.choose(events=_polls(_actors(), count=3), now=_LATER) is not None

    @pytest.mark.parametrize(
        "unsettled",
        [_actors(1), [_cell_target(cell_index=0), _cell_target(cell_index=1, alive=False)]],
    )
    def test_a_missing_or_dead_target_restarts_the_quiescence_count(self, unsettled: list[SoakTarget]) -> None:
        """A disturbed poll resets the count, so earlier settled polls no longer qualify."""
        scheduler = _scheduler({"actor": [_FakeForm()]}, quiescent_polls_required=2)
        events = [
            *_polls(_actors(), count=3),
            _observation(unsettled, at=_at(3)),
            *_polls(_actors(), count=1, start=4),
        ]

        assert scheduler.choose(events=events, now=_LATER) is None

    def test_a_failed_poll_restarts_the_quiescence_count(self) -> None:
        """A poll that saw nothing cannot extend a run of settled polls across it."""
        scheduler = _scheduler({"actor": [_FakeForm()]}, quiescent_polls_required=2)
        events = [
            *_polls(_actors(), count=3),
            _observation(None, at=_at(3), errors={"observation": "timeout"}),
            *_polls(_actors(), count=1, start=4),
        ]

        assert scheduler.choose(events=events, now=_LATER) is None

    def test_a_request_restarts_the_quiescence_count_of_its_kind(self) -> None:
        """Polls from before the last request do not count toward settling after it."""
        scheduler = _scheduler({"actor": [_FakeForm()]}, quiescent_polls_required=2)
        earlier = _request(_actors()[0], form_name="fake", request_id="earlier")
        events = [
            *_polls(_actors(), count=3),
            _requested(earlier, at=_at(3)),
            _applied(earlier, at=_at(4)),
            _result(earlier, at=_at(5)),
            *_polls(_actors(), count=1, start=6),
        ]

        assert scheduler.choose(events=events, now=_LATER) is None
        assert scheduler.choose(events=[*events, *_polls(_actors(), count=1, start=7)], now=_LATER) is not None

    def test_an_alive_but_unready_target_blocks_injection(self) -> None:
        """Every expected target must be ready, not just alive."""
        scheduler = _scheduler({"actor": [_FakeForm()]})
        targets = [_cell_target(cell_index=0), _cell_target(cell_index=1, ready=False)]

        assert scheduler.choose(events=_polls(targets, count=2), now=_LATER) is None

    def test_a_harmful_form_needs_a_spare_replica(self) -> None:
        """Harming the only replica of a kind would stop training, so it is never chosen."""
        harmful = _scheduler({"actor": [_FakeForm(harms_target=True)]}, expected_count=1)
        harmless = _scheduler({"actor": [_FakeForm(harms_target=False)]}, expected_count=1)
        events = _polls(_actors(1))

        assert harmful.choose(events=events, now=_LATER) is None
        assert harmless.choose(events=events, now=_LATER) is not None

    def test_a_harmful_form_targets_one_of_the_ready_replicas(self) -> None:
        """With spares available the chosen target is one of the observed ready targets."""
        scheduler = _scheduler({"actor": [_FakeForm(harms_target=True)]}, expected_count=3)
        targets = _actors(3)

        request = scheduler.choose(events=_polls(targets), now=_LATER)

        assert request.target in targets


class TestSchedulerFormPreference:
    def _proven(self, form_name: str, *, applied: bool) -> list[SoakEvent]:
        earlier = _request(_actors()[0], form_name=form_name, request_id=f"earlier-{form_name}")
        events: list[SoakEvent] = [_requested(earlier, at=_at(0))]
        if applied:
            events.append(_applied(earlier, at=_at(1)))
        return [*events, _result(earlier, at=_at(2)), *_polls(_actors(), start=3)]

    def test_a_form_that_never_worked_is_preferred(self) -> None:
        """Once a form has landed, the others are drawn until they land too."""
        chosen = {
            _scheduler({"actor": [_FakeForm(name="a"), _FakeForm(name="b")]}, seed=seed)
            .choose(events=self._proven("a", applied=True), now=_LATER)
            .form_name
            for seed in range(20)
        }

        assert chosen == {"b"}

    def test_a_request_that_never_applied_does_not_prove_its_form(self) -> None:
        """A draw that never landed leaves its form in the unproven pool."""
        chosen = {
            _scheduler({"actor": [_FakeForm(name="a"), _FakeForm(name="b")]}, seed=seed)
            .choose(events=self._proven("a", applied=False), now=_LATER)
            .form_name
            for seed in range(20)
        }

        assert chosen == {"a", "b"}

    def test_every_form_proven_falls_back_to_all_forms(self) -> None:
        """When all forms landed the draw is over every form again."""
        events = [*self._proven("a", applied=True), *self._proven("b", applied=True)]
        chosen = {
            _scheduler({"actor": [_FakeForm(name="a"), _FakeForm(name="b")]}, seed=seed)
            .choose(events=events, now=_LATER)
            .form_name
            for seed in range(20)
        }

        assert chosen == {"a", "b"}
