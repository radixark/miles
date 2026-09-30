from pathlib import Path

from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import tail_started_at, weight_update_results

from miles.utils.audit_utils.event_logger.logger import read_events

MIN_PUBLICATIONS_AFTER_TAKE_OVERS: int = 2


def assert_publications_after_take_overs(events: list[SoakEvent], *, source: Path) -> None:
    published_since = tail_started_at(events)
    published = [
        result
        for result in weight_update_results(read_events(source))
        if result.published_version is not None and result.timestamp >= published_since
    ]
    assert len(published) >= MIN_PUBLICATIONS_AFTER_TAKE_OVERS, (
        f"Only {len(published)} weight publications after the last take-over applied, expected at least "
        f"{MIN_PUBLICATIONS_AFTER_TAKE_OVERS}"
    )
