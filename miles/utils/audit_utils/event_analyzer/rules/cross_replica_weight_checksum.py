from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import (
    ChecksumMismatchIssue,
    compare_flat_dicts,
    flatten_nested,
)
from miles.utils.audit_utils.event_logger.models import Event, TrainEngineLocalWeightChecksumEvent
from miles.utils.audit_utils.process_identity import TrainProcessIdentity

__all__ = ["ChecksumMismatchIssue", "check"]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """
    Check: weight checksum across replicas should be exactly the same
    """

    checksum_events = [e for e in events if isinstance(e, TrainEngineLocalWeightChecksumEvent)]
    if not checksum_events:
        return []

    all_mismatches: list[ChecksumMismatchIssue] = []

    events_by_key: dict[tuple[int, int], list[TrainEngineLocalWeightChecksumEvent]] = {}
    for event in checksum_events:
        key = (event.rollout_id, event.attempt)
        events_by_key.setdefault(key, []).append(event)

    for key in sorted(events_by_key.keys()):
        all_mismatches += list(_check_one_step(events=events_by_key[key]))

    return all_mismatches


def _get_rank_key(event: TrainEngineLocalWeightChecksumEvent) -> int:
    if isinstance(event.source, TrainProcessIdentity):
        return event.source.rank_within_cell
    return -1


def _check_one_step(events: list[TrainEngineLocalWeightChecksumEvent]) -> Iterable[ChecksumMismatchIssue]:
    # Group events by rank_within_cell so we only compare across replicas (cell_index),
    # not across TP/PP/EP ranks within the same cell (which have different param shards).
    # TODO: group by (component, rank_within_cell) once critic checksum events are supported.
    #  Currently only actor emits TrainEngineLocalWeightChecksumEvent.
    by_rank: dict[int, list[TrainEngineLocalWeightChecksumEvent]] = defaultdict(list)
    for event in events:
        by_rank[_get_rank_key(event)].append(event)

    for rank_events in by_rank.values():
        first = rank_events[0]
        first_flat = _flatten_event(first)
        for other in rank_events[1:]:
            yield from compare_flat_dicts(
                a=first_flat,
                b=_flatten_event(other),
                label_a=_compute_label(first),
                label_b=_compute_label(other),
            )


def _compute_label(event: TrainEngineLocalWeightChecksumEvent) -> str:
    return f"rollout_{event.rollout_id}/{event.source.to_name()}"


def _flatten_event(event: TrainEngineLocalWeightChecksumEvent) -> dict[str, Any]:
    """Flatten all fields of an event into a flat dict with dot-separated keys."""
    return flatten_nested(event.state.model_dump(), prefix="")
