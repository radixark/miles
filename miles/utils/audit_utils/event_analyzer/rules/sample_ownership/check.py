import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import (
    IssuedSampleIdentityIssue,
    SampleOwnershipIssue,
    SampleResolutionIssue,
)
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    Event,
    ExplicitlyDroppedSamplesEvent,
    OutputConsumption,
    TrainerModelCompanionInfoEvent,
    TrainGroupStepEndEvent,
)

logger = logging.getLogger(__name__)


class _CellAndSourceSampleIndex(NamedTuple):
    cell_index: int
    source_sample_index: int


class _GroupSlot(NamedTuple):
    group_index: int
    slot: int


class _GroupSlotAndSourceSampleIndex(NamedTuple):
    group_index: int
    slot: int
    source_sample_index: int


@dataclass(frozen=True)
class _IssuedSource:
    group_index: int | None
    slot: int | None
    source_sample_index: int
    issued_rollout_id: int | None


@dataclass(frozen=True)
class _Facts:
    issued_sources: list[_IssuedSource]
    drop_counts: Counter
    trained_consumptions: dict[_CellAndSourceSampleIndex, list[OutputConsumption]]
    skipped_nonfinite_consumptions: dict[_CellAndSourceSampleIndex, list[OutputConsumption]]
    cell_indices: list[int]
    latest_completed_rollout_id: int
    identity_issues: list[IssuedSampleIdentityIssue]


def check(events: list[Event], *, grace_steps: int) -> list[SampleOwnershipIssue]:
    """Sample ownership invariant.

    - Eligible: a source that is mature, or already consumed on some cell.
    - Mature: at least ``grace_steps`` actor steps completed after the step that issued it.
    - Pass, per cell
        - Case 1: exactly one complete output set -- consumptions share one ``output_count = N``, output indices
          are exactly ``0..N-1``, every count is ``1``, no sibling is both trained and skipped as nonfinite.
        - Case 2: exactly one explicit drop and no trained or skipped consumption.
    - Else is an issue: no outcome, both kinds of outcome, repeated drops, an incomplete set.
    - Every call rechecks every eligible source; a source consumed more than once is an issue (replay buffers are
      unsupported).

    Terms

    - ``source``: one sample issued by the data source, identified by ``source_sample_index``.
    - ``output``: one training input derived from a source, identified by ``(source_sample_index, output_index,
      output_count)``; a fully asynchronous source may yield several.
    - ``sibling``: another output of the same source.
    - ``consumption``: what a cell reports for one output: its lineage and how many times the cell trained on it
      (``trained``) or skipped it as nonfinite (``skipped``).
    - ``cell``: one trainer data-parallel cell, identified by ``cell_index``; every cell trains every batch.
    - ``trained`` vs ``skipped``: outputs the optimizer consumed vs outputs skipped as nonfinite.
    - ``explicit drop``: an ``ExplicitlyDroppedSamplesEvent`` naming the source.
    - ``actor step``: an all-NORMAL ``TrainGroupStepEndEvent`` of the actor role.
    - ``issued``, ``issuing step``: the ``DataSourceIssuedSamplesEvent`` carrying the source, and its ``rollout_id``.
    """
    if grace_steps < 0:
        raise ValueError("grace_steps must be non-negative")

    if (facts := _collect(events)) is None:
        logger.info(
            "Skipping the sample ownership check: no completed actor step has full model companion records yet"
        )
        return []

    return [
        *facts.identity_issues,
        *(
            issue
            for source in _eligible_sources(facts, grace_steps=grace_steps)
            for issue in _check_one_source(source, facts)
        ),
    ]


def _collect(events: list[Event]) -> _Facts | None:
    """Read every fact the rule needs out of one pass over the event log, or nothing when the state is unknown."""
    if (records := _read_latest_model_companion_info(events)) is None:
        return None

    issued_sources, identity_issues = _issued_sources(
        [event for event in events if isinstance(event, DataSourceIssuedSamplesEvent)]
    )

    trained: defaultdict[_CellAndSourceSampleIndex, list[OutputConsumption]] = defaultdict(list)
    skipped_nonfinite: defaultdict[_CellAndSourceSampleIndex, list[OutputConsumption]] = defaultdict(list)
    for record in records:
        for one in record.sample_counts:
            trained[_CellAndSourceSampleIndex(record.cell_index, one.sample.source_sample_index)].append(one)
        for one in record.skipped_nonfinite_sample_counts:
            skipped_nonfinite[_CellAndSourceSampleIndex(record.cell_index, one.sample.source_sample_index)].append(one)

    return _Facts(
        issued_sources=issued_sources,
        drop_counts=Counter(
            index
            for event in events
            if isinstance(event, ExplicitlyDroppedSamplesEvent)
            for index in event.source_sample_indices
        ),
        trained_consumptions=dict(trained),
        skipped_nonfinite_consumptions=dict(skipped_nonfinite),
        cell_indices=[record.cell_index for record in records],
        latest_completed_rollout_id=max(step.rollout_id for step in completed_actor_steps(events)),
        identity_issues=identity_issues,
    )


def _read_latest_model_companion_info(events: list[Event]) -> list[TrainerModelCompanionInfoEvent] | None:
    if not (steps := completed_actor_steps(events)):
        return None

    step = steps[-1]
    records = [
        event
        for event in events
        if isinstance(event, TrainerModelCompanionInfoEvent)
        and (event.rollout_id, event.attempt) == (step.rollout_id, step.attempt)
        and event.cell_index in step.cell_outcomes
    ]

    duplicated = {
        cell_index: count
        for cell_index, count in Counter(record.cell_index for record in records).items()
        if count > 1
    }
    assert not duplicated, (
        f"cells published more than one model companion record for rollout {step.rollout_id} attempt {step.attempt}: "
        f"{duplicated}"
    )
    if {record.cell_index for record in records} != set(step.cell_outcomes):
        return None
    return sorted(records, key=lambda record: record.cell_index)


def completed_actor_steps(events: list[Event]) -> list[TrainGroupStepEndEvent]:
    """Return the actor step ends whose every cell trained normally, oldest first."""
    steps = [
        event
        for event in events
        if isinstance(event, TrainGroupStepEndEvent) and event.role == "actor" and _every_cell_is_normal(event)
    ]
    return sorted(steps, key=lambda event: event.timestamp)


def _every_cell_is_normal(event: TrainGroupStepEndEvent) -> bool:
    """Say whether the step recorded at least one cell and no cell reported anything but a normal outcome."""
    return bool(event.cell_outcomes) and all(
        outcomes != "error" and outcomes and all(outcome == TrainStepOutcome.NORMAL for outcome in outcomes)
        for outcomes in event.cell_outcomes.values()
    )


def _issued_sources(
    events: list[DataSourceIssuedSamplesEvent],
) -> tuple[list[_IssuedSource], list[IssuedSampleIdentityIssue]]:
    """Collect the earliest issuance of every group slot, and report indices issued for more than one slot."""
    by_identity: dict[_GroupSlotAndSourceSampleIndex, _IssuedSource] = {}
    group_slots_by_source_sample_index: defaultdict[int, set[_GroupSlot]] = defaultdict(set)

    for event in events:
        for group in event.groups:
            for slot, sample_index in enumerate(group.sample_indices):
                key = _GroupSlotAndSourceSampleIndex(
                    group_index=group.group_index, slot=slot, source_sample_index=sample_index
                )
                if key not in by_identity or event.rollout_id < by_identity[key].issued_rollout_id:
                    by_identity[key] = _IssuedSource(
                        group_index=group.group_index,
                        slot=slot,
                        source_sample_index=sample_index,
                        issued_rollout_id=event.rollout_id,
                    )
                group_slots_by_source_sample_index[sample_index].add(
                    _GroupSlot(group_index=group.group_index, slot=slot)
                )

    conflicted = {
        sample_index
        for sample_index, group_slots in group_slots_by_source_sample_index.items()
        if len(group_slots) > 1
    }
    issues = [
        IssuedSampleIdentityIssue(
            description="the same sample index was issued for multiple GRPO slots",
            sample_index=sample_index,
            identities=[f"group {one.group_index} slot {one.slot}" for one in sorted(group_slots)],
        )
        for sample_index, group_slots in sorted(group_slots_by_source_sample_index.items())
        if sample_index in conflicted
    ]
    sources = [source for _, source in sorted(by_identity.items()) if source.source_sample_index not in conflicted]
    return sources, issues


def _eligible_sources(facts: _Facts, *, grace_steps: int) -> list[_IssuedSource]:
    """List what this round must resolve: every mature source plus every source already consumed somewhere."""
    consumed = {
        key.source_sample_index for key in (*facts.trained_consumptions, *facts.skipped_nonfinite_consumptions)
    }
    known = {source.source_sample_index for source in facts.issued_sources} | {
        issue.sample_index for issue in facts.identity_issues
    }
    return [
        *(
            source
            for source in facts.issued_sources
            if source.source_sample_index in consumed
            or _is_mature(
                source, latest_completed_rollout_id=facts.latest_completed_rollout_id, grace_steps=grace_steps
            )
        ),
        *(
            _IssuedSource(group_index=None, slot=None, source_sample_index=source_sample_index, issued_rollout_id=None)
            for source_sample_index in sorted(consumed - known)
        ),
    ]


def _is_mature(source: _IssuedSource, *, latest_completed_rollout_id: int, grace_steps: int) -> bool:
    """Say whether at least ``grace_steps`` actor steps completed after the step that issued this source."""
    return (
        source.issued_rollout_id is not None and latest_completed_rollout_id - source.issued_rollout_id >= grace_steps
    )


def _check_one_source(source: _IssuedSource, facts: _Facts) -> list[SampleResolutionIssue]:
    """Report every cell on which one eligible source does not end in exactly one outcome."""
    drop_count = facts.drop_counts[source.source_sample_index]
    if _has_repeated_drops(drop_count):
        return [
            _issue(
                source=source, cell_index=None, trained_consumptions=[], skipped_consumptions=[], drop_count=drop_count
            )
        ]

    issues = []
    for cell_index in facts.cell_indices:
        key = _CellAndSourceSampleIndex(cell_index, source.source_sample_index)
        trained = facts.trained_consumptions.get(key, [])
        skipped = facts.skipped_nonfinite_consumptions.get(key, [])
        resolved = (
            not trained and not skipped if _has_single_drop(drop_count) else _has_complete_output_set(trained, skipped)
        )
        if not resolved:
            issues.append(
                _issue(
                    source=source,
                    cell_index=cell_index,
                    trained_consumptions=_describe_outputs(trained),
                    skipped_consumptions=_describe_outputs(skipped),
                    drop_count=drop_count,
                )
            )
    return issues


def _has_repeated_drops(drop_count: int) -> bool:
    """Say whether one source was explicitly dropped more than once."""
    return drop_count > 1


def _has_single_drop(drop_count: int) -> bool:
    """Say whether one source carries exactly one explicit drop."""
    return drop_count == 1


def _has_complete_output_set(
    trained_consumptions: list[OutputConsumption],
    skipped_consumptions: list[OutputConsumption],
) -> bool:
    """Say whether the consumptions form exactly one complete output set, each output consumed exactly once."""
    consumptions = [*trained_consumptions, *skipped_consumptions]
    if not consumptions:
        return False

    output_counts = {one.sample.output_count for one in consumptions}
    if len(output_counts) != 1 or (output_count := next(iter(output_counts))) <= 0:
        return False
    return (
        len(consumptions) == output_count
        and {one.sample.output_index for one in consumptions} == set(range(output_count))
        and all(one.count == 1 for one in consumptions)
    )


def _describe_outputs(consumptions: list[OutputConsumption]) -> list[str]:
    """Render the training consumptions of one source sample for an issue report."""
    return [
        f"output {consumption.sample.output_index}/{consumption.sample.output_count}: count {consumption.count}"
        for consumption in consumptions
    ]


def _issue(
    *,
    source: _IssuedSource,
    cell_index: int | None,
    trained_consumptions: list[str],
    skipped_consumptions: list[str],
    drop_count: int,
) -> SampleResolutionIssue:
    """Build the issue that names the source sample, the cell, and the evidence behind the verdict."""
    return SampleResolutionIssue(
        description=_resolution_description(
            trained_consumptions=trained_consumptions,
            skipped_consumptions=skipped_consumptions,
            drop_count=drop_count,
        ),
        group_index=source.group_index,
        slot=source.slot,
        sample_index=source.source_sample_index,
        cell_index=cell_index,
        trained_consumptions=trained_consumptions,
        skipped_consumptions=skipped_consumptions,
        drop_count=drop_count,
    )


def _resolution_description(
    *, trained_consumptions: list[str], skipped_consumptions: list[str], drop_count: int
) -> str:
    """Name which of the mutually exclusive resolution failures this evidence shows."""
    if drop_count > 1:
        return "source sample was explicitly dropped more than once"
    if drop_count == 1:
        return "source sample had an output outcome and was explicitly dropped"
    if not trained_consumptions and not skipped_consumptions:
        return "source sample had no training outcome"
    return "source sample does not have one complete set of output outcomes"
