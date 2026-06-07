import logging
from collections import defaultdict
from collections.abc import Iterator

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.utils.event_logger.models import (
    Event,
    TrainAdvantageComputationEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class WitnessDataMismatchIssue(FrozenStrictBaseModel):
    rollout_id: int
    cell_index: int
    description: str
    expected_witness_ids: list[int]
    actual_witness_ids: list[int]


class WitnessMissingSnapshotIssue(FrozenStrictBaseModel):
    rollout_id: int
    cell_index: int
    description: str


WitnessIssue = WitnessDataMismatchIssue | WitnessMissingSnapshotIssue


def check(events: list[Event]) -> list[WitnessIssue]:
    """
    Related events:
    * WitnessAllocateIdEvent: when allocating `witness_id` to `sample_index`
    * WitnessSnapshotParamEvent: near the end of each train() step in MegatronTrainRayActor
        * If a witness_id appears in the weight, it means the corresponding data is consumed at least once.
    * TrainGroupStepEndEvent: after each train() step in RayTrainGroup

    Check:
    1. For each (rollout_id, cell_index),
       if TrainGroupStepEndEvent claims the cell ends with TrainStepOutcome.NORMAL,
       then its WitnessSnapshotParamEvent should observe *EXACTLY* the training data in rollout_id=0~curr.

    Remarks:
    * To correlate witness_id vs sample_index utilize WitnessAllocateIdEvent.
    * Witness' ring buffer will remove old data, thus we need to ignore the appearance/disappearance of
      all values in `WitnessSnapshotParamEvent.stale_ids`
    """

    return list(
        _find_mismatches(
            all_step_events=_filter_by_type(events, TrainGroupStepEndEvent),
            all_witness_events=_filter_by_type(events, WitnessSnapshotParamEvent),
            expected_witness_ids_of_step=_compute_expected_witness_ids_of_step(
                _filter_by_type(events, WitnessAllocateIdEvent)
            ),
            zero_adv_witness_ids_by_rollout=_compute_zero_advantage_witness_ids(
                _filter_by_type(events, TrainAdvantageComputationEvent)
            ),
        )
    )


def _filter_by_type(arr: list, ty: type) -> list:
    return [x for x in arr if isinstance(x, ty)]


def _compute_zero_advantage_witness_ids(
    events: list[TrainAdvantageComputationEvent],
) -> dict[int, set[int]]:
    """Return witness_ids where all per-token advantages == 0.0, keyed by rollout_id.

    Unioned across cells: under indep_dp the per-cell weight snapshot reflects the
    GLOBAL (allreduced) gradient, so a zero-advantage sample contributes nothing
    and its witness is absent from EVERY cell — even cells that never owned it.
    Keying per (rollout_id, cell_index) would let a cell excuse only its own shard,
    falsely flagging peers' zero-advantage witnesses as missing.
    """
    result: dict[int, set[int]] = defaultdict(set)

    for event in events:
        for adv_tokens, wid_tokens in zip(event.advantages, event.witness_ids, strict=True):
            if all(v == 0.0 for v in adv_tokens):
                result[event.rollout_id].add(wid_tokens[0])

    return dict(result)


def _compute_expected_witness_ids_of_step(events: list[WitnessAllocateIdEvent]) -> dict[int, set[int]]:
    latest_by_rollout: dict[int, WitnessAllocateIdEvent] = {}
    for e in events:
        if e.rollout_id not in latest_by_rollout or e.attempt > latest_by_rollout[e.rollout_id].attempt:
            latest_by_rollout[e.rollout_id] = e

    allocated_witness_ids_of_rollout_id = {
        rid: list(e.witness_id_to_sample_index.keys()) for rid, e in latest_by_rollout.items()
    }

    ans: dict[int, set[int]] = {}
    running: set[int] = set()
    for rollout_id in sorted(allocated_witness_ids_of_rollout_id.keys()):
        running = running | set(allocated_witness_ids_of_rollout_id[rollout_id])
        ans[rollout_id] = set(running)
    return ans


def _find_mismatches(
    *,
    all_step_events: list[TrainGroupStepEndEvent],
    all_witness_events: list[WitnessSnapshotParamEvent],
    expected_witness_ids_of_step: dict[int, set[int]],
    zero_adv_witness_ids_by_rollout: dict[int, set[int]],
) -> Iterator[WitnessIssue]:
    for step_event in all_step_events:
        rollout_id = step_event.rollout_id

        for cell_index, cell_outcome in step_event.cell_outcomes.items():
            if cell_outcome == "error":
                continue
            if not all(r == TrainStepOutcome.NORMAL for r in cell_outcome):
                continue

            matching_events = [
                e for e in all_witness_events if e.rollout_id == rollout_id and e.source.cell_index == cell_index
            ]
            if matching_events:
                latest_attempt = max(e.attempt for e in matching_events)
                witness_events_of_cell = [e for e in matching_events if e.attempt == latest_attempt]
            else:
                witness_events_of_cell = []

            if not witness_events_of_cell:
                yield WitnessMissingSnapshotIssue(
                    rollout_id=rollout_id,
                    cell_index=cell_index,
                    description=f"Cell {cell_index} reported NORMAL for rollout {rollout_id} but no WitnessSnapshotParamEvent was found",
                )
                continue

            zero_adv_ids = zero_adv_witness_ids_by_rollout.get(rollout_id, set())

            for event in witness_events_of_cell:
                issue = _compare_snapshot(
                    event=event,
                    expected=expected_witness_ids_of_step.get(rollout_id, set()),
                    rollout_id=rollout_id,
                    cell_index=cell_index,
                    zero_adv_witness_ids=zero_adv_ids,
                )
                if issue is not None:
                    yield issue


def _compare_snapshot(
    *,
    event: WitnessSnapshotParamEvent,
    expected: set[int],
    rollout_id: int,
    cell_index: int,
    zero_adv_witness_ids: set[int],
) -> WitnessDataMismatchIssue | None:
    stale_set = set(event.stale_ids)
    filtered_expected = expected - stale_set - zero_adv_witness_ids
    filtered_actual = set(event.nonzero_witness_ids) - stale_set

    if filtered_expected == filtered_actual:
        return None

    return WitnessDataMismatchIssue(
        rollout_id=rollout_id,
        cell_index=cell_index,
        description=(
            f"Witness data mismatch for instance {event.instance_id}: "
            f"missing={sorted(filtered_expected - filtered_actual)}, "
            f"extra={sorted(filtered_actual - filtered_expected)}"
        ),
        expected_witness_ids=sorted(filtered_expected),
        actual_witness_ids=sorted(filtered_actual),
    )
