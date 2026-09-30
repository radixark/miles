from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["check", "settled_published_updates"]


class WeightUpdateCoverageIssue(FrozenStrictBaseModel):
    debug_weight_update_id: str
    weight_version: int
    description: str


def check(events: list[Event], *, include_latest: bool = False) -> list[WeightUpdateCoverageIssue]:
    """Check: every settled published weight update has one checksum record covering the engines it updated."""
    checksums: dict[str, list[InferenceEngineWeightChecksumEvent]] = {}
    for event in events:
        if isinstance(event, InferenceEngineWeightChecksumEvent):
            checksums.setdefault(event.debug_weight_update_id, []).append(event)

    issues: list[WeightUpdateCoverageIssue] = []
    for result in settled_published_updates(events, include_latest=include_latest):
        records = checksums.get(result.debug_weight_update_id, [])
        if len(records) != 1:
            issues.append(_issue(result, f"{len(records)} engine checksum records instead of one"))
            continue
        observed = {snapshot.cell_id: snapshot.workers_hash for snapshot in records[0].engine_snapshots}
        expected = {cell_id: result.snapshot_cell_id_to_hashes[cell_id] for cell_id in result.updated_cell_ids}
        if observed != expected:
            issues.append(_issue(result, f"checksum record covers {sorted(observed)} instead of {sorted(expected)}"))
    return issues


def settled_published_updates(events: list[Event], *, include_latest: bool = False) -> list[WeightUpdateResultEvent]:
    results = [event for event in events if isinstance(event, WeightUpdateResultEvent)]
    latest = max(
        (event.timestamp for event in events if isinstance(event, (WeightUpdateResultEvent, TrainGroupStepEndEvent))),
        default=None,
    )
    return [
        result
        for result in results
        if result.published_version is not None and (include_latest or result.timestamp < latest)
    ]


def _issue(result: WeightUpdateResultEvent, description: str) -> WeightUpdateCoverageIssue:
    assert result.published_version is not None
    return WeightUpdateCoverageIssue(
        debug_weight_update_id=result.debug_weight_update_id,
        weight_version=result.published_version,
        description=description,
    )
