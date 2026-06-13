"""Rule: every rollout engine must receive identical weights for a given rollout.

Functional sibling of cross_replica_weight_checksum, but rollout-side: it groups
EngineWeightChecksumEvents by rollout_id and compares each engine's merged
checksums against engine 0 of the same rollout.
"""

from collections.abc import Iterable

from miles.utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.event_logger.models import EngineWeightChecksumEvent, Event

__all__ = ["check"]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Check: all engines of one rollout must hold exactly the same weights."""

    checksum_events = [e for e in events if isinstance(e, EngineWeightChecksumEvent)]
    if not checksum_events:
        return []

    events_by_rollout: dict[int, list[EngineWeightChecksumEvent]] = {}
    for event in checksum_events:
        events_by_rollout.setdefault(event.rollout_id, []).append(event)

    all_mismatches: list[ChecksumMismatchIssue] = []
    for rollout_id in sorted(events_by_rollout.keys()):
        all_mismatches += list(_check_one_rollout(events=events_by_rollout[rollout_id]))

    return all_mismatches


def _check_one_rollout(events: list[EngineWeightChecksumEvent]) -> Iterable[ChecksumMismatchIssue]:
    by_engine = sorted(events, key=lambda e: e.engine_index)
    baseline = by_engine[0]
    for other in by_engine[1:]:
        yield from compare_flat_dicts(
            a=baseline.checksums,
            b=other.checksums,
            label_a=_compute_label(baseline),
            label_b=_compute_label(other),
        )


def _compute_label(event: EngineWeightChecksumEvent) -> str:
    return f"rollout_{event.rollout_id}/engine_{event.engine_index}"
