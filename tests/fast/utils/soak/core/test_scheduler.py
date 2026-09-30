import pytest
from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _FakeForm, _observation, _requested, _result
from tests.utils.soak.core import scheduler as scheduler_module
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTargetConfig
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.scheduler import SoakActionScheduler
from tests.utils.soak.core.types import SoakForms, SoakTarget

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
