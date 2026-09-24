from collections.abc import Callable, Sequence
from pathlib import Path

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.ft.actions.resize import LANDING_LAG_ROLLOUTS, ScalingStep
from tests.utils.soak.ft.types import CellTarget, PoolResizedEvidence, ResizeDetails

from miles.utils.test_utils.comparisons.inference_engine_checksums import read_inference_engine_checksum_events


def assert_resizes_follow_schedule(
    events: list[SoakEvent], *, schedule: tuple[ScalingStep, ...], initial_replicas: int
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

    replicas = initial_replicas
    for index, (step, action) in enumerate(zip(schedule, actions, strict=True)):
        request = action.requested.request
        assert request.details == ResizeDetails(
            replicas=step.replicas, at_rollout=step.at_rollout, moment=step.moment
        ), f"resize {index} asked for {request.details}, and the schedule has {step} there"
        assert (
            action.result is not None and action.result.returned
        ), f"resize {index} ({request.request_id}) did not return: {action.result}"
        assert action.applied is not None and isinstance(
            evidence := action.applied.evidence, PoolResizedEvidence
        ), f"resize {index} ({request.request_id}) never reported the pool it resized"
        assert (evidence.replicas_before, evidence.replicas_after) == (replicas, step.replicas), (
            f"resize {index} moved {request.target.identity} {evidence.replicas_before} -> "
            f"{evidence.replicas_after}, and the schedule (from {initial_replicas}) has it {replicas} -> "
            f"{step.replicas}"
        )
        replicas = step.replicas

    print(f"the pool was resized as scheduled: {[initial_replicas, *(step.replicas for step in schedule)]}")


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
    schedule: tuple[ScalingStep, ...],
    num_rollouts: int,
    what: str,
) -> None:
    assert sorted(counts_of_rollout) == list(range(num_rollouts)), (
        f"{what} is known for rollouts {sorted(counts_of_rollout)}, not for each of the {num_rollouts} rollouts "
        f"exactly once"
    )

    sizes = [initial, *(step.replicas for step in schedule)]
    phase = 0
    landed_at: list[int] = []
    for rollout_id in range(num_rollouts):
        count = counts_of_rollout[rollout_id]
        if phase < len(schedule) and rollout_id >= schedule[phase].at_rollout and count == sizes[phase + 1]:
            phase += 1
            landed_at.append(rollout_id)
        assert count == sizes[phase], (
            f"{what} was {count} at rollout {rollout_id}, and the schedule {schedule} (from {initial}) has it at "
            f"{sizes[phase]} there; the whole series is {counts_of_rollout}"
        )

    assert phase == len(schedule), (
        f"only {phase} of the {len(schedule)} resize(s) ever showed in {what}: the series {counts_of_rollout} "
        f"never reached {sizes[phase + 1]}"
    )
    for step, rollout_id in zip(schedule, landed_at, strict=True):
        assert rollout_id <= step.at_rollout + LANDING_LAG_ROLLOUTS, (
            f"the resize fired at rollout {step.at_rollout} showed in {what} only at rollout {rollout_id}, more "
            f"than {LANDING_LAG_ROLLOUTS} rollout(s) later"
        )

    print(f"{what} followed the schedule: {sizes} landing at rollouts {landed_at}")


def assert_observed_cells(
    events: Sequence[SoakEvent],
    *,
    cell_type: str,
    initial: int,
    schedule: tuple[ScalingStep, ...],
    counts: Callable[[CellTarget], bool],
) -> None:
    peak = max(initial, *(step.replicas for step in schedule))
    final = schedule[-1].replicas
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
