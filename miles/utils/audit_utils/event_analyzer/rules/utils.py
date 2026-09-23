from typing import Any

from miles.utils.audit_utils.event_logger.models import EnvReportEvent, Event
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def trainer_args(events: list[Event]) -> dict[str, Any] | None:
    reports = [
        event.report.process.args.values
        for event in events
        if isinstance(event, EnvReportEvent) and isinstance(event.source, TrainProcessIdentity)
    ]
    if not reports:
        return None
    assert all(report == reports[0] for report in reports), "Trainer ranks report different arguments"
    return reports[0]
