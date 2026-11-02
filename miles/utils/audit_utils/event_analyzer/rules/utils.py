from collections.abc import Sequence
from datetime import datetime
from typing import Any

from miles.utils.audit_utils.event_logger.models import EnvReportEvent, Event
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def trainer_args(
    events: list[Event], *, names: Sequence[str], at: datetime, model_id: str | None
) -> dict[str, Any] | None:
    latest_by_rank = {
        (event.source.cell_index, event.source.rank_within_cell): event
        for event in sorted(events, key=lambda event: event.timestamp)
        if isinstance(event, EnvReportEvent)
        and isinstance(event.source, TrainProcessIdentity)
        and event.source.component == "actor"
        and event.source.model_id == model_id
        and event.timestamp <= at
    }
    reports = [{name: event.report.process.args.values[name] for name in names} for event in latest_by_rank.values()]
    if not reports:
        return None
    assert all(
        report == reports[0] for report in reports
    ), f"Trainer ranks report different arguments for {sorted(names)}"
    return reports[0]
