from pathlib import Path

from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import tail_started_at, weight_update_results

from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_coverage
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent

MIN_PUBLICATIONS_AFTER_TAKE_OVERS: int = 2


def assert_publications_after_take_overs(events: list[SoakEvent], *, source: Path) -> None:
    published_since = tail_started_at(events)
    training_events = read_events(source)
    published = [
        result
        for result in weight_update_results(training_events)
        if result.published_version is not None and result.timestamp >= published_since
    ]
    assert len(published) >= MIN_PUBLICATIONS_AFTER_TAKE_OVERS, (
        f"Only {len(published)} weight publications after the last take-over applied, expected at least "
        f"{MIN_PUBLICATIONS_AFTER_TAKE_OVERS}"
    )

    published_ids = {result.debug_weight_update_id for result in published}
    invalid_ids = {
        issue.debug_weight_update_id
        for issue in inference_engine_weight_checksum_coverage.check(training_events, include_latest=True)
    }
    assert any(
        isinstance(event, InferenceEngineWeightChecksumEvent)
        and event.debug_weight_update_id in published_ids - invalid_ids
        for event in training_events
    ), "No successfully audited weight publication after the last take-over"
