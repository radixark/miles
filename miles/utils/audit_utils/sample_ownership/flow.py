from collections.abc import Iterator
from typing import Any

from miles.rollout.data_source import DataSource
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    IssuedSampleGroup,
)
from miles.utils.types import Sample


def record_data_source_issues(data_source: DataSource) -> None:
    get_samples = data_source.get_samples

    def get_samples_and_record(num_samples: int) -> list[list[Sample]]:
        groups = get_samples(num_samples)
        _log_issued_groups(groups)
        return groups

    data_source.get_samples = get_samples_and_record


def _log_issued_groups(groups: list[list[Sample]]) -> None:
    if not groups or not is_event_logger_initialized():
        return

    issued_groups = [
        IssuedSampleGroup(
            group_index=_require_identity(group[0].group_index, "group_index"),
            sample_indices=[_require_identity(sample.index, "sample index") for sample in group],
        )
        for group in groups
    ]
    get_event_logger().log(
        DataSourceIssuedSamplesEvent,
        dict(groups=issued_groups),
        print_log=False,
    )


def log_dropped_samples(samples: list[Sample], *, reason: str, rollout_id: int | None = None) -> None:
    if not samples or not is_event_logger_initialized():
        return

    log_dropped_sample_indices(
        [_source_sample_index(sample) for sample in samples],
        reason=reason,
        rollout_id=rollout_id,
    )


def log_dropped_sample_indices(
    sample_indices: list[int],
    *,
    reason: str,
    rollout_id: int | None = None,
) -> None:
    sample_indices = list(dict.fromkeys(sample_indices))
    if not sample_indices or not is_event_logger_initialized():
        return

    get_event_logger().log(
        ExplicitlyDroppedSamplesEvent,
        dict(sample_indices=sample_indices, reason=reason, rollout_id=rollout_id),
        print_log=False,
    )


def log_dropped_groups(
    before: list[Any],
    after: list[Any],
    *,
    reason: str,
    rollout_id: int | None = None,
) -> None:
    retained = {_source_sample_index(sample) for sample in _iter_samples(after)}
    removed = [sample for sample in _iter_samples(before) if _source_sample_index(sample) not in retained]
    log_dropped_samples(removed, reason=reason, rollout_id=rollout_id)


def _require_identity(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"DataSource returned a sample without a {name}")
    return value


def _source_sample_index(sample: Sample) -> int:
    index = sample.source_sample_index if sample.source_sample_index is not None else sample.index
    return _require_identity(index, "source sample index")


def _iter_samples(node: list[Any]) -> Iterator[Sample]:
    for item in node:
        if isinstance(item, Sample):
            yield item
        else:
            yield from _iter_samples(item)
