from pathlib import Path

from tests.utils.deploy.hot_restart.evidence import read_discarded_event_dirs
from tests.utils.soak.core.checkers.engine_checksums import (
    assert_engine_checksums_cover_published_updates,
    assert_same_version_engines_agree,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import tail_started_at

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


def assert_engine_checksums_after_take_overs(events: list[SoakEvent], *, source: Path) -> None:
    assert_engine_checksums_cover_published_updates(
        read_events(source), published_since=tail_started_at(events), min_published_updates=2
    )

    checksums: dict[tuple[float, int, str], InferenceEngineWeightChecksumEvent] = {}
    for directory in [*read_discarded_event_dirs(str(source.parent)), source]:
        for event in read_events(directory):
            if isinstance(event, InferenceEngineWeightChecksumEvent):
                key = (event.debug_trainer_load_state_timestamp, event.weight_version, event.debug_weight_update_id)
                assert checksums.setdefault(key, event) == event, f"Conflicting checksum records: {key}"
    assert_same_version_engines_agree(list(checksums.values()))
