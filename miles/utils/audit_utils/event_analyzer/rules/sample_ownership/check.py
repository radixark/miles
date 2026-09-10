from datetime import datetime, timedelta

from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.issued import _issued_samples, _IssuedSample
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import (
    SampleOwnershipIssue,
    SampleResolutionIssue,
)
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.witness import (
    _current_witnesses,
    _describe_rows,
    _rows_by_sample,
    _rows_have_one_outcome,
)
from miles.utils.audit_utils.event_analyzer.utils import filter_by_type
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    Event,
    ExplicitlyDroppedSamplesEvent,
    TrainingSampleCount,
)


def check(
    events: list[Event],
    *,
    grace_period: timedelta,
    now: datetime,
) -> list[SampleOwnershipIssue]:
    if grace_period < timedelta():
        raise ValueError("grace_period must be non-negative")

    issued, identity_issues = _issued_samples(filter_by_type(events, DataSourceIssuedSamplesEvent))
    drops: dict[int, int] = {}
    for event in filter_by_type(events, ExplicitlyDroppedSamplesEvent):
        for sample_index in event.sample_indices:
            drops[sample_index] = drops.get(sample_index, 0) + 1

    mature = [sample for sample in issued if now - sample.issued_at >= grace_period]
    if not mature:
        return identity_issues

    current_witnesses, witness_issues = _current_witnesses(events)
    if witness_issues:
        return [*identity_issues, *witness_issues]
    current_rows = [
        (
            witness.replica_id,
            _rows_by_sample(witness.sample_counts),
            _rows_by_sample(witness.skipped_nonfinite_sample_counts),
        )
        for witness in current_witnesses
    ]
    return [
        *identity_issues,
        *(
            issue
            for sample in mature
            for issue in _resolution_issues(sample=sample, current_rows=current_rows, drops=drops)
        ),
    ]


def _resolution_issues(
    *,
    sample: _IssuedSample,
    current_rows: list[
        tuple[
            str,
            dict[int, list[TrainingSampleCount]],
            dict[int, list[TrainingSampleCount]],
        ]
    ],
    drops: dict[int, int],
) -> list[SampleResolutionIssue]:
    drop_count = drops.get(sample.sample_index, 0)
    if drop_count > 1:
        return [_issue(sample=sample, replica_id=None, trained_rows=[], skipped_rows=[], drop_count=drop_count)]

    issues = []
    for replica_id, trained_by_sample, skipped_by_sample in current_rows:
        trained_rows = trained_by_sample.get(sample.sample_index, [])
        skipped_rows = skipped_by_sample.get(sample.sample_index, [])
        valid = (
            not trained_rows and not skipped_rows
            if drop_count == 1
            else _rows_have_one_outcome(trained_rows, skipped_rows)
        )
        if not valid:
            issues.append(
                _issue(
                    sample=sample,
                    replica_id=replica_id,
                    trained_rows=_describe_rows(trained_rows),
                    skipped_rows=_describe_rows(skipped_rows),
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
        return "mature issued sample was explicitly dropped more than once"
    if drop_count == 1:
        return "mature issued sample had a row outcome and was explicitly dropped"
    if not trained_rows and not skipped_rows:
        return "mature issued sample had no training outcome"
    return "mature issued sample does not have one complete set of row outcomes"
