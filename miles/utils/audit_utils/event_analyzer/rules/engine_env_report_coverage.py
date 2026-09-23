from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_checksum_coverage import (
    settled_published_updates,
)
from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent, Event
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["check"]


class EngineEnvReportMissingIssue(FrozenStrictBaseModel):
    cell_id: str
    workers_hash: str
    debug_weight_update_id: str


def check(events: list[Event], *, include_latest: bool = False) -> list[EngineEnvReportMissingIssue]:
    """Check: every engine incarnation that took a settled published weight update reported its environment."""
    reported = {
        (event.cell_id, event.workers_hash)
        for event in events
        if isinstance(event, EngineEnvReportEvent) and event.workers_hash is not None
    }
    return [
        EngineEnvReportMissingIssue(
            cell_id=cell_id,
            workers_hash=result.snapshot_cell_id_to_hashes[cell_id],
            debug_weight_update_id=result.debug_weight_update_id,
        )
        for result in settled_published_updates(events, include_latest=include_latest)
        for cell_id in result.updated_cell_ids
        if (cell_id, result.snapshot_cell_id_to_hashes[cell_id]) not in reported
    ]
