import pytest

from tests.utils.soak.ft.checkers.resize import assert_counts_follow_schedule, assert_schedule_fits
from tests.utils.soak.ft.types import ResizeStep, compute_sizes


class TestResizeSchedule:
    def test_expansion_and_contraction_preserve_every_requested_size(self) -> None:
        """A valid schedule retains both growth and shrink transitions."""
        schedule = (ResizeStep(at_rollout=2, replicas=4), ResizeStep(at_rollout=6, replicas=1))
        assert_schedule_fits(schedule, initial_replicas=2, num_rollouts=11)
        assert compute_sizes(initial_replicas=2, schedule=schedule) == [2, 4, 1]

    @pytest.mark.parametrize(
        "steps,num_rollouts,message",
        [
            ([], 10, "empty schedule"),
            ([(2, 2)], 10, "scales nothing"),
            ([(2, 4), (5, 1)], 12, "still be landing"),
            ([(2, 4)], 6, "leaving no rollout"),
        ],
    )
    def test_invalid_schedules_fail_before_launch(
        self, steps: list[tuple[int, int]], num_rollouts: int, message: str
    ) -> None:
        """Schedules must change size and leave distinct observable training windows."""
        schedule = tuple(ResizeStep(at_rollout=at, replicas=size) for at, size in steps)
        with pytest.raises(AssertionError, match=message):
            assert_schedule_fits(schedule, initial_replicas=2, num_rollouts=num_rollouts)


class TestObservedResizeCounts:
    @pytest.mark.parametrize("landing", [2, 5])
    def test_immediate_and_last_allowed_landing_are_accepted(self, landing: int) -> None:
        """A resize may land immediately or at the configured lag boundary."""
        assert_counts_follow_schedule(
            {i: 2 if i < landing else 4 for i in range(8)},
            initial=2,
            schedule=(ResizeStep(at_rollout=2, replicas=4),),
            num_rollouts=8,
            subject="cells",
        )

    @pytest.mark.parametrize(
        "counts,message",
        [
            ({i: 4 for i in range(8)}, "was 4 at rollout 0"),
            ({i: 2 for i in range(8)}, "only 0"),
            ({i: 2 if i < 6 else 4 for i in range(8)}, "more than 3"),
            ({i: 2 if i < 2 or i == 4 else 4 for i in range(8)}, "was 2 at rollout 4"),
            ({i: 2 if i < 2 else 4 for i in range(7)}, "exactly once"),
        ],
    )
    def test_missing_early_late_or_reverted_transitions_are_rejected(
        self, counts: dict[int, int], message: str
    ) -> None:
        """Observed counts must cover all rollouts and follow the requested transition window."""
        with pytest.raises(AssertionError, match=message):
            assert_counts_follow_schedule(
                counts,
                initial=2,
                schedule=(ResizeStep(at_rollout=2, replicas=4),),
                num_rollouts=8,
                subject="cells",
            )


class TestAppliedResizeEvidence:
    @pytest.mark.parametrize("failure", [None, "missing", "step", "unapplied", "previous_size"])
    def test_each_scheduled_transition_requires_matching_applied_evidence(self, failure: str | None) -> None:
        """A resize schedule is proven only by matching requests, application evidence, and prior sizes."""
        from tests.utils.soak.core.events import SoakActionAppliedEvent, SoakActionRequestedEvent
        from tests.utils.soak.core.types import SoakActionRequest
        from tests.utils.soak.ft.checkers.resize import assert_resizes_follow_schedule
        from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails

        step = ResizeStep(at_rollout=2, replicas=4)
        request = SoakActionRequest(
            target=PoolTarget(
                identity="train", alive=True, ready=True, replicas=3 if failure == "previous_size" else 2
            ),
            form_name="resize:actor",
            details=ResizeDetails(step=step.model_copy(update={"replicas": 5}) if failure == "step" else step),
        )
        events = [SoakActionRequestedEvent(request=request)]
        if failure != "unapplied":
            events.append(SoakActionAppliedEvent(request_id=request.request_id, evidence=PoolResizedEvidence()))
        if failure == "missing":
            events = []

        if failure is None:
            assert_resizes_follow_schedule(events, schedule=(step,), initial_replicas=2)
        else:
            with pytest.raises(AssertionError):
                assert_resizes_follow_schedule(events, schedule=(step,), initial_replicas=2)
