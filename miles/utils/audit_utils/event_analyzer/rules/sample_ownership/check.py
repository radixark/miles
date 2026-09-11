from dataclasses import dataclass
from datetime import datetime, timedelta

from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.issued import _issued_samples, _IssuedSample
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import (
    SampleOwnershipIssue,
    SampleResolutionIssue,
)
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.witness import (
    _describe_outputs,
    _group_outputs_by_source_sample,
    _latest_completed_cohort_snapshots,
    _outputs_have_exactly_one_outcome,
)
from miles.utils.audit_utils.event_analyzer.utils import filter_by_type
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    Event,
    ExplicitlyDroppedSamplesEvent,
    TrainingSampleCount,
)


@dataclass(frozen=True)
class _ReplicaOutcomes:
    replica_id: str
    trained_by_sample: dict[int, list[TrainingSampleCount]]
    skipped_by_sample: dict[int, list[TrainingSampleCount]]


def check(
    events: list[Event],
    *,
    grace_period: timedelta,
    now: datetime | None,
) -> list[SampleOwnershipIssue]:
    if grace_period < timedelta():
        raise ValueError("grace_period must be non-negative")

    issued, identity_issues = _issued_samples(filter_by_type(events, DataSourceIssuedSamplesEvent))
    drops: dict[int, int] = {}
    for event in filter_by_type(events, ExplicitlyDroppedSamplesEvent):
        for sample_index in event.sample_indices:
            drops[sample_index] = drops.get(sample_index, 0) + 1

    replica_snapshots, witness_issues = _latest_completed_cohort_snapshots(events)
    if witness_issues:
        return [*identity_issues, *witness_issues]
    replica_outcomes = [
        _ReplicaOutcomes(
            replica_id=snapshot.replica_id,
            trained_by_sample=_group_outputs_by_source_sample(snapshot.sample_counts),
            skipped_by_sample=_group_outputs_by_source_sample(snapshot.skipped_nonfinite_sample_counts),
        )
        for snapshot in replica_snapshots
    ]
    observed_source_sample_indices = {
        sample_index
        for replica in replica_outcomes
        for sample_index in replica.trained_by_sample.keys() | replica.skipped_by_sample.keys()
    }
    samples_to_check = [
        sample
        for sample in issued
        if sample.sample_index in observed_source_sample_indices
        or (now is not None and sample.issued_at is not None and now - sample.issued_at >= grace_period)
    ]
    known_indices = {sample.sample_index for sample in issued} | {issue.sample_index for issue in identity_issues}
    samples_to_check.extend(
        _IssuedSample(group_index=None, slot=None, sample_index=sample_index, issued_at=None)
        for sample_index in sorted(observed_source_sample_indices - known_indices)
    )
    return [
        *identity_issues,
        *(
            issue
            for sample in samples_to_check
            for issue in _resolution_issues(sample=sample, replica_outcomes=replica_outcomes, drops=drops)
        ),
    ]


def _resolution_issues(
    *,
    sample: _IssuedSample,
    replica_outcomes: list[_ReplicaOutcomes],
    drops: dict[int, int],
) -> list[SampleResolutionIssue]:
    drop_count = drops.get(sample.sample_index, 0)
    if drop_count > 1:
        return [_issue(sample=sample, replica_id=None, trained_rows=[], skipped_rows=[], drop_count=drop_count)]

    issues = []
    for replica in replica_outcomes:
        trained_rows = replica.trained_by_sample.get(sample.sample_index, [])
        skipped_rows = replica.skipped_by_sample.get(sample.sample_index, [])
        valid = (
            not trained_rows and not skipped_rows
            if drop_count == 1
            else _outputs_have_exactly_one_outcome(trained_rows, skipped_rows)
        )
        if not valid:
            issues.append(
                _issue(
                    sample=sample,
                    replica_id=replica.replica_id,
                    trained_rows=_describe_outputs(trained_rows),
                    skipped_rows=_describe_outputs(skipped_rows),
                    drop_count=drop_count,
                )
            )
    return issues


def _issue(
    *,
    sample: _IssuedSample,
    replica_id: str | None,
    trained_rows: list[str],
    skipped_rows: list[str],
    drop_count: int,
) -> SampleResolutionIssue:
    return SampleResolutionIssue(
        description=_resolution_description(
            trained_rows=trained_rows,
            skipped_rows=skipped_rows,
            drop_count=drop_count,
        ),
        group_index=sample.group_index,
        slot=sample.slot,
        sample_index=sample.sample_index,
        replica_id=replica_id,
        trained_rows=trained_rows,
        skipped_rows=skipped_rows,
        drop_count=drop_count,
    )


def _resolution_description(*, trained_rows: list[str], skipped_rows: list[str], drop_count: int) -> str:
    if drop_count > 1:
        return "source sample was explicitly dropped more than once"
    if drop_count == 1:
        return "source sample had an output outcome and was explicitly dropped"
    if not trained_rows and not skipped_rows:
        return "source sample had no training outcome"
    return "source sample does not have one complete set of output outcomes"
