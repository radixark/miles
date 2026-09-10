import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, NamedTuple

from miles.utils.audit_utils.event_analyzer.utils import filter_by_type
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    Event,
    RolloutGroupRoutedEvent,
    RolloutHoldingsSnapshotEvent,
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
    TrainerCheckpointEvent,
    TrainerCpuWitnessEvent,
    TrainerGroupMappingEvent,
    TrainerTrainedSamplesEvent,
    TrainGroupStepEndEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity, TrainProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel

RUNTIME_TOLERANCE_SNAPSHOTS = 3
logger = logging.getLogger(__name__)


class SampleLostIssue(FrozenStrictBaseModel):
    rollout_id: int
    description: str
    sample_indices: list[int]


class SampleConsumedTwiceIssue(FrozenStrictBaseModel):
    description: str
    sample_indices: list[int]


class TrainerWeightHistoryIssue(FrozenStrictBaseModel):
    rollout_id: int
    description: str
    sources: list[str]
    sample_indices: list[int] = []


SampleOwnershipIssue = SampleLostIssue | SampleConsumedTwiceIssue | TrainerWeightHistoryIssue


@dataclass
class _WeightEvidence:
    trained: list[TrainerTrainedSamplesEvent] = field(default_factory=list)
    pending: set[int] = field(default_factory=set)
    defects: dict[str, str] = field(default_factory=dict)


def check(events: list[Event], *, latest_only: bool = False) -> list[SampleOwnershipIssue]:
    lineage_id = current_lineage_id(events)
    restores = {event.lineage_id: event for event in filter_by_type(events, RolloutStateRestoreEvent)}
    transitions = [e for e in filter_by_type(events, SampleOwnerTransitionEvent) if e.lineage_id == lineage_id]
    routed = [e for e in filter_by_type(events, RolloutGroupRoutedEvent) if e.lineage_id == lineage_id]
    snapshots = [e for e in filter_by_type(events, RolloutHoldingsSnapshotEvent) if e.lineage_id == lineage_id]
    if not snapshots:
        return []

    snapshots.sort(key=lambda snapshot: snapshot.timestamp)
    if latest_only:
        snapshots = snapshots[-RUNTIME_TOLERANCE_SNAPSHOTS:]
    evidence = []
    for snapshot in snapshots:
        if snapshot.rank_weight_witness_supported:
            evidence.append(_weight_evidence(events=events, snapshot=snapshot, restores=restores))
            continue
        _warn_unverified_weight_lineage()
        evidence.append(
            _WeightEvidence(
                trained=[
                    event
                    for event in filter_by_type(events, TrainerTrainedSamplesEvent)
                    if event.rollout_id <= snapshot.rollout_id
                    and (
                        event.lineage_id == lineage_id
                        or _ancestor_trained(event=event, lineage_id=lineage_id, restores=restores)
                    )
                ]
            )
        )

    rank_issues: list[SampleOwnershipIssue] = []
    recent: list[set[str]] = []
    for snapshot, state in zip(snapshots, evidence, strict=True):
        recent = [*recent, set(state.defects)][-RUNTIME_TOLERANCE_SNAPSHOTS:]
        failures = (
            set(state.defects)
            if snapshot.reason in {"save", "final"}
            else (set.intersection(*recent) if len(recent) == RUNTIME_TOLERANCE_SNAPSHOTS else set())
        )
        for source in sorted(failures):
            rank_issues.append(
                TrainerWeightHistoryIssue(
                    rollout_id=snapshot.rollout_id, sources=[source], description=state.defects[source]
                )
            )
    return [
        *rank_issues,
        *_check_nothing_lost(
            transitions=transitions,
            routed=routed,
            snapshots=snapshots,
            evidence=evidence,
        ),
        *_check_nothing_consumed_twice(
            transitions=transitions,
            snapshots=snapshots,
            trained=evidence[-1].trained if snapshots[-1].rank_weight_witness_supported else [],
        ),
    ]


@lru_cache(maxsize=1)
def _warn_unverified_weight_lineage() -> None:
    logger.warning(
        "The training backend does not provide a rank CPU witness; weight lineage and repeated optimizer consumption are not verified"
    )


class _RecordKey(NamedTuple):
    lineage_id: str | None
    trainer_model_id: str | None
    rollout_id: int
    step_id: int
    attempt: int
    slot: int | None


def _record_key(record: dict[str, Any]) -> _RecordKey:
    return _RecordKey(
        lineage_id=record["lineage_id"],
        trainer_model_id=record["trainer_model_id"],
        rollout_id=record["rollout_id"],
        step_id=record["step_id"],
        attempt=record["attempt"],
        slot=record.get("slot"),
    )


def _weight_evidence(
    *,
    events: list[Event],
    snapshot: RolloutHoldingsSnapshotEvent,
    restores: dict[str | None, RolloutStateRestoreEvent],
) -> _WeightEvidence:
    result = _WeightEvidence()
    witnesses, expected, boundaries = _snapshot_witnesses(events=events, snapshot=snapshot)
    by_role: dict[str, list[TrainerCpuWitnessEvent]] = defaultdict(list)
    for source, role in expected.items():
        witness = witnesses.get(source)
        boundary = boundaries.get(role, snapshot.rollout_id)
        if witness is None or (
            role in boundaries
            and (witness.rollout_id < boundary or (witness.reason == "step" and witness.rollout_id == boundary))
        ):
            last_rollout = witness.rollout_id if witness is not None else "never"
            result.defects[f"{source}:missing:last-witness:{last_rollout}"] = (
                f"Active rank {source} has no CPU weight evidence at the completed training boundary"
            )
            witnesses.pop(source, None)
    for source, witness in witnesses.items():
        if source in expected:
            by_role[witness.source.component].append(witness)

    mappings: dict[tuple[str, _RecordKey], list[TrainerGroupMappingEvent]] = defaultdict(list)
    for mapping in filter_by_type(events, TrainerGroupMappingEvent):
        if snapshot.reason == "step" and mapping.timestamp > snapshot.timestamp:
            continue
        mappings[mapping.source.component, _record_key(mapping.model_dump())].append(mapping)

    actor_start = min(
        (record["rollout_id"] for witness in by_role.get("actor", []) for record in witness.records),
        default=float("inf"),
    )
    for role, ranks in by_role.items():
        counts: list[Counter[tuple[bool, _RecordKey, tuple[int, ...]]]] = []
        for witness in ranks:
            source = witness.source.to_name()
            receipts: Counter[tuple[bool, _RecordKey, tuple[int, ...]]] = Counter()
            for pending, records in ((False, witness.records), (True, witness.pending_records)):
                for record in records:
                    key = _record_key(record)
                    if key.trainer_model_id != snapshot.trainer_model_id or key.rollout_id > boundaries.get(
                        role, snapshot.rollout_id
                    ):
                        continue
                    receipts[pending, key, tuple(sorted(record["group_indices"]))] += 1
            counts.append(receipts)

        common = counts[0].copy()
        combined = counts[0].copy()
        complete = len(ranks) == sum(value == role for value in expected.values())
        for receipts in counts[1:]:
            common &= receipts
            combined |= receipts
        for witness in ranks:
            for receipt in combined - common:
                result.defects[f"{witness.source.to_name()}:record:{receipt}"] = (
                    f"Active {role} ranks disagree on CPU weight record {receipt} through rollout {boundaries.get(role, snapshot.rollout_id)}"
                )
        if not complete:
            continue
        witness = ranks[0]
        for (pending, key, groups), count in common.items():
            receipt = TrainerTrainedSamplesEvent(
                timestamp=witness.timestamp,
                source=witness.source,
                lineage_id=key.lineage_id,
                rollout_id=key.rollout_id,
                trainer_model_id=key.trainer_model_id,
                sample_indices=[],
            )
            if key.lineage_id != snapshot.lineage_id and not _ancestor_trained(
                event=receipt, lineage_id=snapshot.lineage_id, restores=restores
            ):
                continue
            candidates = mappings.get((role, key), [])
            contents = {
                tuple(sorted((group, tuple(sorted(indices))) for group, indices in mapping.groups.items()))
                for mapping in candidates
            }
            mapping = dict(next(iter(contents))) if len(contents) == 1 else {}
            if len(contents) != 1 or not set(groups).issubset(mapping):
                for rank in ranks:
                    result.defects[f"{rank.source.to_name()}:mapping:{key}"] = (
                        f"CPU weight record {key} has missing or conflicting group mappings"
                    )
                continue
            indices = [index for group in groups for index in mapping[group]]
            if role == "critic" and key.rollout_id >= actor_start:
                continue
            if pending:
                result.pending.update(indices)
            else:
                result.trained.extend(
                    TrainerTrainedSamplesEvent(
                        timestamp=witness.timestamp,
                        source=witness.source,
                        lineage_id=key.lineage_id,
                        rollout_id=key.rollout_id,
                        trainer_model_id=key.trainer_model_id,
                        sample_indices=indices,
                    )
                    for _ in range(count)
                )
    return result


class _SnapshotWitnesses(NamedTuple):
    witnesses: dict[str, TrainerCpuWitnessEvent]
    expected: dict[str, str]
    boundaries: dict[str, int]


def _snapshot_witnesses(*, events: list[Event], snapshot: RolloutHoldingsSnapshotEvent) -> _SnapshotWitnesses:
    completions: dict[str, TrainGroupStepEndEvent] = {}
    active: dict[str, set[int]] = {}
    memberships: dict[str, CellReconfigureEvent] = {}
    for event in sorted(events, key=lambda event: event.timestamp):
        if event.timestamp > snapshot.timestamp:
            continue
        if (
            isinstance(event.source, (TrainProcessIdentity, TrainerControllerProcessIdentity))
            and event.source.model_id != snapshot.trainer_model_id
        ):
            continue
        if (
            isinstance(event, TrainGroupStepEndEvent)
            and event.role is not None
            and event.rollout_id <= snapshot.rollout_id
        ):
            completions[event.role] = event
        if isinstance(event, CellReconfigureEvent) and event.role is not None:
            active[event.role] = set(event.alive_cell_indices_after)
            memberships[event.role] = event

    boundaries = {role: event.rollout_id for role, event in completions.items()}
    expected: dict[str, str] = {}
    for role, event in completions.items():
        rank_counts = {cell: len(outcomes) for cell, outcomes in event.cell_outcomes.items() if outcomes != "error"}
        cells = set(rank_counts)
        if role in active:
            cells &= active[role]
            if memberships[role].timestamp >= event.timestamp:
                cells |= active[role]
        for cell in cells:
            for rank in range(rank_counts.get(cell, max(rank_counts.values(), default=0))):
                source = TrainProcessIdentity(
                    component=role, model_id=snapshot.trainer_model_id, cell_index=cell, rank_within_cell=rank
                )
                expected[source.to_name()] = role

    visible = [
        event
        for event in events
        if event.timestamp <= snapshot.timestamp
        and (not isinstance(event, TrainerCpuWitnessEvent) or event.lineage_id in (snapshot.lineage_id, None))
    ]
    witnesses = {
        event.source.to_name(): event
        for event in _latest_cpu_witnesses(visible)
        if event.trainer_model_id == snapshot.trainer_model_id
    }
    for source, witness in list(witnesses.items()):
        role = witness.source.component
        if (
            isinstance(witness.source, TrainProcessIdentity)
            and role in active
            and witness.source.cell_index not in active[role]
        ):
            del witnesses[source]
        elif role not in completions:
            expected[source] = role

    if snapshot.reason == "save":
        for role in (set(expected.values()) & {"actor", "critic"}) - snapshot.checkpoint_ids.keys():
            expected[f"{role}:checkpoint"] = role
        checkpoints = {event.checkpoint_id: event for event in filter_by_type(events, TrainerCheckpointEvent)}
        for role, checkpoint_id in snapshot.checkpoint_ids.items():
            checkpoint = checkpoints.get(checkpoint_id)
            if (
                checkpoint is None
                or checkpoint.role != role
                or checkpoint.rollout_id != snapshot.rollout_id
                or checkpoint.rank_count <= 0
                or checkpoint.cell_index not in checkpoint.alive_cell_indices
                or (
                    isinstance(checkpoint.source, TrainerControllerProcessIdentity)
                    and checkpoint.source.model_id != snapshot.trainer_model_id
                )
            ):
                expected[f"{role}:checkpoint:{checkpoint_id}"] = role
                continue
            boundaries[role] = snapshot.rollout_id
            expected = {source: expected_role for source, expected_role in expected.items() if expected_role != role}
            saved = {
                event.source.to_name(): event
                for event in sorted(filter_by_type(events, TrainerCpuWitnessEvent), key=lambda event: event.timestamp)
                if event.checkpoint_id == checkpoint_id
                and event.reason == "save"
                and event.reset
                and event.trainer_model_id == snapshot.trainer_model_id
                and event.rollout_id == snapshot.rollout_id
            }
            for cell in checkpoint.alive_cell_indices:
                for rank in range(checkpoint.rank_count):
                    source = TrainProcessIdentity(
                        component=role, model_id=snapshot.trainer_model_id, cell_index=cell, rank_within_cell=rank
                    ).to_name()
                    expected[source] = role
                    witnesses.pop(source, None)
                    if source in saved:
                        witnesses[source] = saved[source]
    return _SnapshotWitnesses(witnesses=witnesses, expected=expected, boundaries=boundaries)


def _latest_cpu_witnesses(events: list[Event]) -> list[TrainerCpuWitnessEvent]:
    latest: dict[str, TrainerCpuWitnessEvent] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for event in sorted(filter_by_type(events, TrainerCpuWitnessEvent), key=lambda event: event.timestamp):
        source = event.source.to_name()
        if event.reset:
            records[source] = list(event.records)
        else:
            if source not in records:
                continue
            records[source].extend(event.records)
        latest[source] = event
    return [event.model_copy(update={"records": records[source]}) for source, event in latest.items()]


def current_lineage_id(events: Sequence[Event]) -> str | None:
    restores = [event for event in filter_by_type(events, RolloutStateRestoreEvent) if event.lineage_id is not None]
    snapshots = filter_by_type(events, RolloutHoldingsSnapshotEvent)
    if not restores:
        return max(snapshots, key=lambda event: event.timestamp).lineage_id if snapshots else None
    parents = {event.parent_lineage_id for event in restores}
    leaves = [event for event in restores if event.lineage_id not in parents]
    by_id = {event.lineage_id: event for event in restores}

    def _depth(event: RolloutStateRestoreEvent) -> int:
        seen: set[str | None] = set()
        while event.lineage_id not in seen:
            seen.add(event.lineage_id)
            if event.parent_lineage_id not in by_id:
                break
            event = by_id[event.parent_lineage_id]
        return len(seen)

    return max(leaves, key=lambda event: (_depth(event), event.timestamp)).lineage_id


def _ancestor_trained(
    *, event: TrainerTrainedSamplesEvent, lineage_id: str | None, restores: dict[str | None, RolloutStateRestoreEvent]
) -> bool:
    seen: set[str | None] = set()
    cutoff: int | None = None
    while lineage_id in restores and lineage_id not in seen:
        seen.add(lineage_id)
        restore = restores[lineage_id]
        point = (restore.rollout_ids or {}).get(event.trainer_model_id, restore.rollout_id)
        if point is None:
            return False
        cutoff = point if cutoff is None else min(cutoff, point)
        lineage_id = restore.parent_lineage_id
        if event.lineage_id == lineage_id:
            return event.rollout_id <= cutoff
    return False


# ================================ nothing lost ================================


def _check_nothing_lost(
    *,
    transitions: list[SampleOwnerTransitionEvent],
    routed: list[RolloutGroupRoutedEvent],
    snapshots: list[RolloutHoldingsSnapshotEvent],
    evidence: list[_WeightEvidence],
) -> list[SampleOwnershipIssue]:
    ordered_transitions = sorted([*transitions, *routed], key=lambda event: event.timestamp)
    ordered_snapshots = sorted(snapshots, key=lambda event: event.timestamp)
    entered: set[int] = set()
    shared_prompts: set[int] = set()
    dropped: set[int] = set()
    transition_pos = 0

    issues: list[SampleOwnershipIssue] = []
    recent: list[set[int]] = []
    for snapshot, state in zip(ordered_snapshots, evidence, strict=True):
        trained_at: dict[int, int] = {}
        for event in state.trained:
            for index in event.sample_indices:
                trained_at[index] = min(trained_at.get(index, event.rollout_id), event.rollout_id)
        at = snapshot.timestamp
        while transition_pos < len(ordered_transitions) and ordered_transitions[transition_pos].timestamp <= at:
            event = ordered_transitions[transition_pos]
            if isinstance(event, RolloutGroupRoutedEvent):
                shared_prompts.difference_update(event.prompt_indices)
            elif event.to_owner in {SampleOwner.IN_FLIGHT, SampleOwner.RETRY_BUFFER}:
                shared_prompts.update(event.sample_indices)
            elif event.to_owner in {
                SampleOwner.OUTPUT_BUFFER,
                SampleOwner.HANDED_TO_TRAINER,
            }:
                entered |= set(event.sample_indices)
            elif event.to_owner == SampleOwner.DROPPED:
                dropped |= set(event.sample_indices)
            transition_pos += 1
        held = {index for indices in snapshot.holdings.values() for index in indices} | state.pending
        orphans = {
            index
            for index in (entered | shared_prompts) - dropped - held
            if index not in trained_at or trained_at[index] > snapshot.rollout_id
        }

        if snapshot.reason in {"save", "final"} and orphans:
            issues.append(
                SampleLostIssue(
                    rollout_id=snapshot.rollout_id,
                    description=(
                        f"the {snapshot.reason} snapshot of rollout {snapshot.rollout_id} accounts for no owner of "
                        f"{len(orphans)} samples that were generated; resuming from it would train on fewer "
                        f"prompts than the run consumed"
                    ),
                    sample_indices=sorted(orphans),
                )
            )

        recent = [*recent, orphans][-RUNTIME_TOLERANCE_SNAPSHOTS:]
        if len(recent) == RUNTIME_TOLERANCE_SNAPSHOTS and (persistent := set.intersection(*recent)):
            issues.append(
                SampleLostIssue(
                    rollout_id=snapshot.rollout_id,
                    description=(
                        f"{len(persistent)} samples had no owner at {RUNTIME_TOLERANCE_SNAPSHOTS} consecutive "
                        f"snapshots up to rollout {snapshot.rollout_id}, so this is not the lag between processes"
                    ),
                    sample_indices=sorted(persistent),
                )
            )
    return issues


# =========================== nothing consumed twice ===========================


def _check_nothing_consumed_twice(
    *,
    transitions: list[SampleOwnerTransitionEvent],
    snapshots: list[RolloutHoldingsSnapshotEvent],
    trained: list[TrainerTrainedSamplesEvent],
) -> list[SampleOwnershipIssue]:
    if any(snapshot.replays_samples for snapshot in snapshots):
        return []

    return [
        *_repeated_issue((event.sample_indices for event in trained), what="trained"),
        *_repeated_issue(
            (
                event.sample_indices
                for event in transitions
                if event.to_owner == SampleOwner.HANDED_TO_TRAINER and event.reason != "restored"
            ),
            what="handed",
        ),
    ]


def _repeated_issue(indices_lists: Iterable[Iterable[int]], *, what: str) -> list[SampleOwnershipIssue]:
    if not (repeated := _repeated(indices_lists)):
        return []
    description = (
        f"{len(repeated)} samples reached the weights of one lineage more than once; the same "
        f"gradient was applied twice and the effective batch is not what the run reports"
        if what == "trained"
        else f"{len(repeated)} samples were handed to training twice within one lineage; a replay that "
        f"the buffer does not declare would train them twice"
    )
    return [SampleConsumedTwiceIssue(description=description, sample_indices=sorted(repeated))]


# ==================================== misc ====================================


def _repeated(event_indices: Iterable[Iterable[int]]) -> set[int]:
    seen: set[int] = set()
    ans: set[int] = set()
    for indices in event_indices:
        unique = set(indices)
        ans |= seen & unique
        seen |= unique
    return ans
