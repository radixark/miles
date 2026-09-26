import pytest
from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _observation, _request, _requested, _step_end
from tests.utils.soak.core.checkers.tail_completeness import assert_tail_complete
from tests.utils.soak.core.events import SoakAdmissionClosedEvent, SoakEvent

from miles.backends.megatron_utils.ft.types import TrainStepOutcome


def _progress(rollout_id: int, *, at: float, **kwargs: object) -> SoakEvent:
    return _observation(None, at=_at(at), new_sut_events=[_step_end(rollout_id, at=_at(at), **kwargs)])


class TestAssertTailComplete:
    def test_a_normal_step_past_the_closing_rollout_after_closure_completes_the_tail(self) -> None:
        """A later rollout finishing normally after admission closed proves the tail."""
        assert_tail_complete([_progress(3, at=0), SoakAdmissionClosedEvent(timestamp=_at(1)), _progress(4, at=2)])

    def test_a_run_whose_admission_never_closed_is_rejected(self) -> None:
        """Without a closure there is no tail to judge."""
        with pytest.raises(AssertionError, match="never closed"):
            assert_tail_complete([_progress(3, at=0)])

    def test_no_progress_after_closure_is_rejected(self) -> None:
        """Steps from before the closure do not count toward the tail."""
        with pytest.raises(AssertionError, match="no successful training progress"):
            assert_tail_complete([_progress(3, at=0), SoakAdmissionClosedEvent(timestamp=_at(1))])

    def test_a_late_repeat_of_an_already_finished_rollout_is_rejected(self) -> None:
        """A step after closure must advance past every rollout seen before it."""
        with pytest.raises(AssertionError, match="no successful training progress"):
            assert_tail_complete([_progress(3, at=0), SoakAdmissionClosedEvent(timestamp=_at(1)), _progress(3, at=2)])

    def test_progress_before_the_last_applied_fault_is_rejected(self) -> None:
        """A fault landing after closure moves the tail start past earlier steps."""
        request = _request(_cell_target())
        with pytest.raises(AssertionError, match="no successful training progress"):
            assert_tail_complete(
                [
                    _requested(request, at=_at(0)),
                    SoakAdmissionClosedEvent(timestamp=_at(1)),
                    _progress(4, at=2),
                    _applied(request, at=_at(3)),
                ]
            )

    def test_progress_after_the_last_applied_fault_completes_the_tail(self) -> None:
        """A normal step after both the closure and the last effect proves the tail."""
        request = _request(_cell_target())
        assert_tail_complete(
            [
                _requested(request, at=_at(0)),
                SoakAdmissionClosedEvent(timestamp=_at(1)),
                _applied(request, at=_at(2)),
                _progress(4, at=3),
            ]
        )

    def test_a_step_without_a_normal_outcome_is_rejected(self) -> None:
        """A discarded retry after closure is not successful progress."""
        with pytest.raises(AssertionError, match="no successful training progress"):
            assert_tail_complete(
                [
                    SoakAdmissionClosedEvent(timestamp=_at(1)),
                    _progress(4, at=2, outcomes=[TrainStepOutcome.DISCARDED_SHOULD_RETRY]),
                ]
            )

    def test_a_step_of_another_trainer_is_ignored(self) -> None:
        """Only the actor trainer's steps prove the tail."""
        with pytest.raises(AssertionError, match="no successful training progress"):
            assert_tail_complete([SoakAdmissionClosedEvent(timestamp=_at(1)), _progress(4, at=2, trainer_id="critic")])
