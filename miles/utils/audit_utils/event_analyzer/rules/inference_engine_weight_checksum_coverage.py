from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineChecksumCollectionFailedEvent,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["check", "settled_published_updates"]


class WeightUpdateCoverageIssue(FrozenStrictBaseModel):
    debug_weight_update_id: str
    weight_version: int
    description: str
    is_warning_only: bool = False


def check(events: list[Event], *, include_latest: bool = False) -> list[WeightUpdateCoverageIssue]:
    """Check: every settled published weight update has one checksum record covering the engines it updated."""
    checksums: dict[str, list[InferenceEngineWeightChecksumEvent | InferenceEngineChecksumCollectionFailedEvent]] = {}
    for event in events:
        if isinstance(event, (InferenceEngineWeightChecksumEvent, InferenceEngineChecksumCollectionFailedEvent)):
            checksums.setdefault(event.debug_weight_update_id, []).append(event)

    issues: list[WeightUpdateCoverageIssue] = []
    consecutive_failure_events: dict[str | None, int] = {}
    for result in sorted(
        settled_published_updates(events, include_latest=include_latest), key=lambda event: event.timestamp
    ):
        records = checksums.get(result.debug_weight_update_id, [])
        if len(records) != 1:
            issues.append(_issue(result, f"{len(records)} engine checksum records instead of one"))
            continue
        expected = {cell_id: result.snapshot_cell_id_to_hashes[cell_id] for cell_id in result.updated_cell_ids}
        record = records[0]
        if (
            not isinstance(result.source, TrainerControllerProcessIdentity)
            or record.trainer_model_id != result.source.model_id
            or record.weight_version != result.published_version
            or record.debug_trainer_load_state_timestamp != result.debug_trainer_load_state_timestamp
        ):
            issues.append(_issue(result, "checksum record names another publication"))
            continue
        if isinstance(record, InferenceEngineChecksumCollectionFailedEvent):
            if record.published_cell_id_to_hashes != expected:
                issues.append(_issue(result, "checksum collection failure names another publication"))
            else:
                count = consecutive_failure_events.get(record.trainer_model_id, 0) + 1
                consecutive_failure_events[record.trainer_model_id] = count
                issues.append(
                    _issue(
                        result,
                        f"checksum collection failed: {record.reason}",
                        is_warning_only=count < 3,
                    )
                )
        else:
            observed = {snapshot.cell_id: snapshot.workers_hash for snapshot in record.engine_snapshots}
            if observed != expected:
                issues.append(
                    _issue(result, f"checksum record covers {sorted(observed)} instead of {sorted(expected)}")
                )
            else:
                consecutive_failure_events[record.trainer_model_id] = 0
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


def _issue(
    result: WeightUpdateResultEvent, description: str, *, is_warning_only: bool = False
) -> WeightUpdateCoverageIssue:
    assert result.published_version is not None
    return WeightUpdateCoverageIssue(
        debug_weight_update_id=result.debug_weight_update_id,
        weight_version=result.published_version,
        description=description,
        is_warning_only=is_warning_only,
    )
