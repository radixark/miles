from miles.utils.audit_utils.event_logger.models import Event, WeightUpdateTransferChecksumEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["check"]


class WeightUpdateChecksumIssue(FrozenStrictBaseModel):
    debug_weight_update_id: str
    cell_id: str
    workers_hash: str
    receiver_rank: int
    mismatched_names: list[str]


def check(events: list[Event]) -> list[WeightUpdateChecksumIssue]:
    """Check: every P2P write's received tensors hash like the send buffers they came from."""
    issues: list[WeightUpdateChecksumIssue] = []
    for event in events:
        if not isinstance(event, WeightUpdateTransferChecksumEvent):
            continue
        mismatched_names = sorted(
            name
            for name in event.sent_checksums.keys() | event.received_checksums.keys()
            if event.sent_checksums.get(name) != event.received_checksums.get(name)
        )
        if mismatched_names:
            issues.append(
                WeightUpdateChecksumIssue(
                    debug_weight_update_id=event.debug_weight_update_id,
                    cell_id=event.cell_id,
                    workers_hash=event.workers_hash,
                    receiver_rank=event.receiver_rank,
                    mismatched_names=mismatched_names,
                )
            )
    return issues
