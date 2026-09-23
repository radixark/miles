from collections.abc import Iterable

from miles.utils.audit_utils.event_logger.models import EnvReportEvent, Event
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def trainer_env_reports(events: Iterable[Event]) -> list[EnvReportEvent]:
    return [
        event
        for event in events
        if isinstance(event, EnvReportEvent) and isinstance(event.source, TrainProcessIdentity)
    ]


def report_ranks(reports: Iterable[EnvReportEvent]) -> set[tuple[int, int]]:
    return {(report.source.cell_index, report.source.rank_within_cell) for report in reports}
