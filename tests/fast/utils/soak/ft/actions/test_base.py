from tests.fast.utils.soak.soak_fakes import _at, _healed_injection, _observation, _step_end
from tests.utils.soak.core.events import SoakActionResultEvent, SoakEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction

_FORM = InjectFaultForm(base_url="http://api:18080", action=KillProcessAction())


def _is_recovered(events: list[SoakEvent]) -> bool:
    [action] = project_actions(events).values()
    return _FORM.is_recovered(action=action, events=events)


def _rollout_injection(**kwargs: object) -> list[SoakEvent]:
    return _healed_injection(_FORM.name, kind="rollout", cell_index=0, start=0, request_id="r", **kwargs)


class TestBaseCellFaultFormIsRecovered:
    def test_a_healed_cell_followed_by_a_normal_step_is_recovered(self) -> None:
        """Recovery needs the fresh ready incarnation and a later normal training step."""
        assert _is_recovered(_rollout_injection())

    def test_a_cell_fault_form_harms_its_target(self) -> None:
        """Cell faults always take a replica down, so the scheduler keeps a spare."""
        assert _FORM.harms_target

    def test_without_a_later_normal_step_it_is_not_recovered(self) -> None:
        """A healed cell with training still stalled has not recovered the run."""
        assert not _is_recovered(_rollout_injection(step_after=False))

    def test_a_step_before_the_recovery_does_not_count(self) -> None:
        """Progress from before the replacement was ready proves nothing."""
        events = _rollout_injection(step_after=False)
        events.insert(3, _observation(None, at=_at(2.5), new_sut_events=[_step_end(9, at=_at(2.5))]))

        assert not _is_recovered(events)

    def test_a_step_at_the_recovery_instant_does_not_count(self) -> None:
        """The step must strictly follow the recovery observation."""
        events = [
            *_rollout_injection(step_after=False),
            _observation(None, at=_at(4), new_sut_events=[_step_end(9, at=_at(3))]),
        ]

        assert not _is_recovered(events)

    def test_a_retried_step_after_recovery_does_not_count(self) -> None:
        """Only a normal step shows training moved on."""
        events = [
            *_rollout_injection(step_after=False),
            _observation(
                None,
                at=_at(4),
                new_sut_events=[_step_end(9, at=_at(4), outcomes=[TrainStepOutcome.DISCARDED_SHOULD_RETRY])],
            ),
        ]

        assert not _is_recovered(events)

    def test_an_action_without_a_returned_result_is_not_recovered(self) -> None:
        """A failed or still running action never counts as recovered."""
        failed = [
            event.model_copy(update={"returned": False}) if isinstance(event, SoakActionResultEvent) else event
            for event in _rollout_injection()
        ]
        running = [event for event in _rollout_injection() if not isinstance(event, SoakActionResultEvent)]

        assert not _is_recovered(failed)
        assert not _is_recovered(running)

    def test_an_action_without_an_effect_is_not_recovered(self) -> None:
        """A request that never landed has nothing to recover from."""
        events = [event for event in _rollout_injection() if event.kind != "action_applied"]

        assert not _is_recovered(events)
