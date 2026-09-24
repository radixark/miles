from collections.abc import Callable, Sequence
from pathlib import Path

from tests.utils.soak.core.config import RunMoment
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.ft.types import CellTarget, PoolResizedEvidence, ResizeDetails

from miles.utils.test_utils.comparisons.inference_engine_checksums import read_inference_engine_checksum_events

LANDING_LAG_ROLLOUTS: int = 3


def assert_schedule_leaves_room(
    moments: tuple[RunMoment, ...], sizes: tuple[int, ...], *, initial_replicas: int, num_rollouts: int
) -> None:
    assert sizes, "an empty schedule resizes nothing, and this scenario is about a pool that changes size"
    replicas = initial_replicas
    previous_at = -LANDING_LAG_ROLLOUTS - 1
    for moment, size in zip(moments, sizes, strict=True):
        assert size != replicas, f"resizing to {size} at {moment} keeps the pool at {replicas}, so it scales nothing"
        assert moment.at_rollout > previous_at + LANDING_LAG_ROLLOUTS, (
            f"{moment} fires while the resize before it may still be landing (up to {LANDING_LAG_ROLLOUTS} rollouts "
            f"after rollout {previous_at}), so the two sizes could not be told apart"
        )
        replicas = size
        previous_at = moment.at_rollout
    assert previous_at + LANDING_LAG_ROLLOUTS < num_rollouts - 1, (
        f"the last resize fires at rollout {previous_at} and may land up to {LANDING_LAG_ROLLOUTS} rollouts later, "
        f"leaving no rollout of the {num_rollouts} to train at the final size"
    )


def assert_resizes_follow_schedule(events: list[SoakEvent], *, sizes: tuple[int, ...], initial_replicas: int) -> None:
    actions = [
        action
        for action in project_actions(events).values()
        if isinstance(action.requested.request.details, ResizeDetails)
    ]
    assert len(actions) == len(sizes), (
        f"the run ended after {len(actions)} of {len(sizes)} resize(s), so the pool never went through every "
        f"size the schedule names"
    )

    replicas = initial_replicas
    for index, (size, action) in enumerate(zip(sizes, actions, strict=True)):
        request = action.requested.request
        assert request.details == ResizeDetails(
            replicas=size
        ), f"resize {index} asked for {request.details}, and the schedule has {size} there"
        assert (
            action.result is not None and action.result.returned
        ), f"resize {index} ({request.request_id}) did not return: {action.result}"
        assert action.applied is not None and isinstance(
            evidence := action.applied.evidence, PoolResizedEvidence
        ), f"resize {index} ({request.request_id}) never reported the pool it resized"
        assert (evidence.replicas_before, evidence.replicas_after) == (replicas, size), (
            f"resize {index} moved {request.target.identity} {evidence.replicas_before} -> "
            f"{evidence.replicas_after}, and the schedule (from {initial_replicas}) has it {replicas} -> {size}"
        )
        replicas = size

    print(f"the pool was resized as scheduled: {[initial_replicas, *sizes]}")


def read_engine_counts_of_rollout(dump_dir: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for event in read_inference_engine_checksum_events(Path(dump_dir)):
        assert event.rollout_id not in counts, f"{dump_dir} holds two weight updates for rollout {event.rollout_id}"
        counts[event.rollout_id] = len(event.engine_snapshots)
    return counts


def assert_counts_follow_schedule(
    counts_of_rollout: dict[int, int],
    *,
    initial: int,
    moments: tuple[RunMoment, ...],
    sizes: tuple[int, ...],
    num_rollouts: int,
    what: str,
) -> None:
    assert sorted(counts_of_rollout) == list(range(num_rollouts)), (
        f"{what} is known for rollouts {sorted(counts_of_rollout)}, not for each of the {num_rollouts} rollouts "
        f"exactly once"
    )

    schedule = list(zip(moments, sizes, strict=True))
    levels = [initial, *sizes]
    phase = 0
    landed_at: list[int] = []
    for rollout_id in range(num_rollouts):
        count = counts_of_rollout[rollout_id]
        if phase < len(schedule) and rollout_id >= moments[phase].at_rollout and count == levels[phase + 1]:
            phase += 1
            landed_at.append(rollout_id)
        assert count == levels[phase], (
            f"{what} was {count} at rollout {rollout_id}, and the schedule {schedule} (from {initial}) has it at "
            f"{levels[phase]} there; the whole series is {counts_of_rollout}"
        )

    assert phase == len(schedule), (
        f"only {phase} of the {len(schedule)} resize(s) ever showed in {what}: the series {counts_of_rollout} "
        f"never reached {levels[phase + 1]}"
    )
    for moment, rollout_id in zip(moments, landed_at, strict=True):
        assert rollout_id <= moment.at_rollout + LANDING_LAG_ROLLOUTS, (
            f"the resize fired at rollout {moment.at_rollout} showed in {what} only at rollout {rollout_id}, more "
            f"than {LANDING_LAG_ROLLOUTS} rollout(s) later"
        )

    print(f"{what} followed the schedule: {levels} landing at rollouts {landed_at}")


def assert_observed_cells(
    events: Sequence[SoakEvent],
    *,
    cell_type: str,
    initial: int,
    sizes: tuple[int, ...],
    counts: Callable[[CellTarget], bool],
) -> None:
    peak = max(initial, *sizes)
    final = sizes[-1]
    observed = [
        sum(1 for target in event.targets if target.kind == cell_type and counts(target))
        for event in events
        if isinstance(event, SoakObservationEvent) and event.targets is not None
    ]
    assert observed, f"the api server was never read, so nothing here says how many {cell_type} cells the run saw"
    assert max(observed) == peak, (
        f"the api server listed at most {max(observed)} {cell_type} cell(s) at once, not the {peak} the pool "
        f"was resized to: {observed}"
    )
    assert observed[-1] == final, (
        f"the api server listed {observed[-1]} {cell_type} cell(s) when the run ended, not the {final} the "
        f"pool was resized back to: {observed}"
    )
    print(f"the api server listed up to {peak} and finally {final} {cell_type} cell(s) across {len(observed)} reads")
