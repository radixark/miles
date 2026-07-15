from collections.abc import Iterable

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent

__all__ = ["check"]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Check: all engines of one rollout must hold exactly the same weights."""
    issues: list[ChecksumMismatchIssue] = []
    for event in events:
        if isinstance(event, InferenceEngineWeightChecksumEvent):
            issues += list(_check_one_rollout(event))
    return issues


def _check_one_rollout(event: InferenceEngineWeightChecksumEvent) -> Iterable[ChecksumMismatchIssue]:
    engines = event.engine_checksums
    if len(engines) < 2:
        return
    baseline = engines[0]
    for engine_index in range(1, len(engines)):
        yield from compare_flat_dicts(
            a=baseline,
            b=engines[engine_index],
            label_a=f"rollout_{event.rollout_id}/engine_0",
            label_b=f"rollout_{event.rollout_id}/engine_{engine_index}",
        )
