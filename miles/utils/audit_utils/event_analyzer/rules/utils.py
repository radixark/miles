from collections.abc import Sequence
from typing import Any

from miles.utils.audit_utils.event_logger.models import EnvReportEvent, Event
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def trainer_args(events: list[Event], *, names: Sequence[str]) -> dict[str, Any] | None:
    reports = [
        {name: event.report.process.args.values[name] for name in names}
        for event in events
        if isinstance(event, EnvReportEvent) and isinstance(event.source, TrainProcessIdentity)
    ]
    if not reports:
        return None
    assert all(
        report == reports[0] for report in reports
    ), f"Trainer ranks report different arguments for {sorted(names)}"
    return reports[0]
