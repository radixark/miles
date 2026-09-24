from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails, ResizeStep, compute_sizes

LANDING_LAG_ROLLOUTS: int = 3


def assert_schedule_fits(schedule: tuple[ResizeStep, ...], *, initial_replicas: int, num_rollouts: int) -> None:
    assert schedule, "an empty schedule resizes nothing, and this scenario is about a pool that changes size"
    replicas = initial_replicas
    previous_at = -LANDING_LAG_ROLLOUTS - 1
    for step in schedule:
        assert step.replicas != replicas, f"{step} keeps the pool at {replicas} replica(s), so it scales nothing"
        assert step.at_rollout > previous_at + LANDING_LAG_ROLLOUTS, (
            f"{step} fires while the step before it may still be landing (up to {LANDING_LAG_ROLLOUTS} rollouts "
            f"after rollout {previous_at}), so the two sizes could not be told apart"
        )
        replicas = step.replicas
        previous_at = step.at_rollout
    assert previous_at + LANDING_LAG_ROLLOUTS < num_rollouts - 1, (
        f"the last step fires at rollout {previous_at} and may land up to {LANDING_LAG_ROLLOUTS} rollouts later, "
        f"leaving no rollout of the {num_rollouts} to train at the final size"
    )


def assert_resizes_follow_schedule(
    events: list[SoakEvent], *, schedule: tuple[ResizeStep, ...], initial_replicas: int
) -> None:
    actions = [
        action
        for action in project_actions(events).values()
        if isinstance(action.requested.request.details, ResizeDetails)
    ]
    assert len(actions) == len(schedule), (
        f"the run ended after {len(actions)} of {len(schedule)} resize(s), so the pool never went through every "
        f"size the schedule names"
    )

    sizes = compute_sizes(initial_replicas=initial_replicas, schedule=schedule)
    for index, (step, action) in enumerate(zip(schedule, actions, strict=True)):
        request = action.requested.request
        assert request.details == ResizeDetails(
            step=step
        ), f"resize {index} asked for {request.details}, and the schedule has {step} there"
        assert action.applied is not None and isinstance(
            action.applied.evidence, PoolResizedEvidence
        ), f"resize {index} ({request.request_id}) never reported the pool it resized"
        assert isinstance(request.target, PoolTarget) and request.target.replicas == sizes[index], (
            f"resize {index} found {request.target.identity} at {request.target.replicas} replica(s), and the "
            f"schedule has it at {sizes[index]} there ({sizes})"
        )

    print(f"the pool was resized as scheduled: {sizes}")


def assert_counts_follow_schedule(
    counts_of_rollout_id: dict[int, int],
    *,
    initial: int,
    schedule: tuple[ResizeStep, ...],
    num_rollouts: int,
    subject: str,
) -> None:
    assert sorted(counts_of_rollout_id) == list(range(num_rollouts)), (
        f"{subject} is known for rollouts {sorted(counts_of_rollout_id)}, not for each of the {num_rollouts} rollouts "
        f"exactly once"
    )

    sizes = compute_sizes(initial_replicas=initial, schedule=schedule)
    num_landed = 0
    landed_at: list[int] = []
    for rollout_id in range(num_rollouts):
        count = counts_of_rollout_id[rollout_id]
        if (
            num_landed < len(schedule)
            and rollout_id >= schedule[num_landed].at_rollout
            and count == sizes[num_landed + 1]
        ):
            num_landed += 1
            landed_at.append(rollout_id)
        assert count == sizes[num_landed], (
            f"{subject} was {count} at rollout {rollout_id}, and the schedule {schedule} (from {initial}) has it at "
            f"{sizes[num_landed]} there; the whole series is {counts_of_rollout_id}"
        )

    assert num_landed == len(schedule), (
        f"only {num_landed} of the {len(schedule)} resize(s) ever showed in {subject}: the series "
        f"{counts_of_rollout_id} never reached {sizes[num_landed + 1]}"
    )
    for step, rollout_id in zip(schedule, landed_at, strict=True):
        assert rollout_id <= step.at_rollout + LANDING_LAG_ROLLOUTS, (
            f"the resize fired during rollout {step.at_rollout} showed in {subject} only at rollout {rollout_id}, "
            f"more than {LANDING_LAG_ROLLOUTS} rollout(s) later"
        )

    print(f"{subject} followed the schedule: {sizes} landing at rollouts {landed_at}")
