from collections.abc import Sequence
from datetime import datetime

from miles.utils.audit_utils.event_analyzer.rules import (
    inference_engine_weight_checksum_consistency,
    inference_engine_weight_movement,
)
from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineWeightChecksumEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


def assert_published_weight_checksums(
    events: Sequence[Event],
    *,
    publication_since: datetime | None = None,
    minimum_publications: int = 1,
) -> None:
    publications: dict[tuple[str | None, str, int, str], dict[str, str]] = {}
    checksums: dict[tuple[str | None, str, int, str], dict[str, str]] = {}
    updates: set[tuple[str | None, str, str]] = set()
    interrupted_window: set[tuple[str | None, str, int, str]] = set()
    for event in events:
        if isinstance(event, WeightUpdateResultEvent) and event.published_version is not None:
            assert isinstance(
                event.source, TrainerControllerProcessIdentity
            ), "Weight publication lacks trainer identity"
            assert event.version_epoch and event.update_id, "Publication lacks epoch or update identity"
            update = (event.source.model_id, event.version_epoch, event.update_id)
            assert update not in updates, f"Repeated weight update identity: {update}"
            updates.add(update)
            key = (event.source.model_id, event.version_epoch, event.published_version, event.update_id)
            assert event.updated_cell_ids, f"Published version has no updated engine: {key}"
            assert len(set(event.updated_cell_ids)) == len(event.updated_cell_ids), f"Repeated updated engine: {key}"
            assert not set(event.updated_cell_ids).intersection(
                event.failed_cell_ids
            ), f"Published engine is also reported failed: {key}"
            publications[key] = {cell_id: event.target_incarnations[cell_id] for cell_id in event.updated_cell_ids}
            if publication_since is not None and event.timestamp < publication_since:
                interrupted_window.add(key)
        elif isinstance(event, InferenceEngineWeightChecksumEvent):
            assert (
                event.weight_version is not None and event.engine_snapshots
            ), "Checksum evidence lacks version or instances"
            assert event.version_epoch and event.update_id, "Checksum lacks epoch or update identity"
            key = (event.trainer_model_id, event.version_epoch, event.weight_version, event.update_id)
            assert key not in checksums, f"Duplicate checksum publication: {key}"
            checksums[key] = {snapshot.cell_id: snapshot.workers_hash for snapshot in event.engine_snapshots}
    publications = {key: value for key, value in publications.items() if key not in interrupted_window}
    checksums = {key: value for key, value in checksums.items() if key not in interrupted_window}
    assert (
        len(publications) >= minimum_publications
    ), f"Expected at least {minimum_publications} successful weight publications"
    assert checksums == publications, "Checksum versions or engine incarnations do not cover every publication"
    assert_weight_checksum_history(events)


def assert_weight_checksum_history(events: Sequence[Event]) -> None:
    assert not inference_engine_weight_checksum_consistency.check(list(events)), "Same-version engine weights differ"
    assert not inference_engine_weight_movement.check(
        list(events)
    ), "Adjacent published weights failed movement checks"
