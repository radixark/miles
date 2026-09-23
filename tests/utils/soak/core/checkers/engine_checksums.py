from collections.abc import Sequence
from datetime import datetime

from tests.utils.soak.core.views import PublishedWeightUpdateKey, published_weight_updates

from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_consistency
from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent


def assert_engine_checksums_cover_published_updates(
    events: Sequence[Event],
    *,
    published_since: datetime | None = None,
    min_published_updates: int = 1,
) -> None:
    published = published_weight_updates(events)
    interrupted = {
        key for key, result in published.items() if published_since is not None and result.timestamp < published_since
    }

    expected = {
        key: {cell_id: result.snapshot_cell_id_to_hashes[cell_id] for cell_id in result.updated_cell_ids}
        for key, result in published.items()
        if key not in interrupted
    }

    checksums: dict[PublishedWeightUpdateKey, dict[str, str]] = {}
    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent):
            continue
        key = PublishedWeightUpdateKey(
            event.trainer_model_id,
            event.debug_trainer_load_state_timestamp,
            event.weight_version,
            event.debug_weight_update_id,
        )
        assert key not in checksums, f"Duplicate checksum publication: {key}"
        checksums[key] = {snapshot.cell_id: snapshot.workers_hash for snapshot in event.engine_snapshots}

    assert (
        len(expected) >= min_published_updates
    ), f"Expected at least {min_published_updates} published weight updates"
    assert {
        key: value for key, value in checksums.items() if key not in interrupted
    } == expected, "Checksum versions or engine incarnations do not cover every publication"
    assert_same_version_engines_agree(events)


def assert_same_version_engines_agree(events: Sequence[Event]) -> None:
    assert not inference_engine_weight_checksum_consistency.check(list(events)), "Same-version engine weights differ"
