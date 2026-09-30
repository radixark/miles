from collections.abc import Iterable

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent

__all__ = ["check"]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Check: all engines of one weight publication must hold exactly the same weights."""
    issues: list[ChecksumMismatchIssue] = []
    for event in events:
        if isinstance(event, InferenceEngineWeightChecksumEvent):
            issues += list(_check_one_publication(event))
    return issues


def _check_one_publication(event: InferenceEngineWeightChecksumEvent) -> Iterable[ChecksumMismatchIssue]:
    baseline, *others = event.engine_snapshots
    for snapshot in others:
        yield from compare_flat_dicts(
            a=baseline.tensor_checksums,
            b=snapshot.tensor_checksums,
            label_a=f"version_{event.weight_version}/cell_{baseline.cell_id}",
            label_b=f"version_{event.weight_version}/cell_{snapshot.cell_id}",
        )
