from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ReconfigureInfo(FrozenStrictBaseModel):
    rollout_id: int
    src_cell_index: int | None
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]

    @staticmethod
    def from_event(event: CellReconfigureEvent) -> "ReconfigureInfo":
        return ReconfigureInfo(
            rollout_id=event.rollout_id,
            src_cell_index=event.src_cell_index,
            healed_cell_indices=event.healed_cell_indices,
            alive_cell_indices_after=event.alive_cell_indices_after,
        )


def assert_reconfigure_events(event_dir: Path, *, expected: list[ReconfigureInfo]) -> None:
    assert event_dir.is_dir(), f"Event directory {event_dir} does not exist or is not a directory"
    actual = [ReconfigureInfo.from_event(event) for event in load_reconfigure_events(event_dir)]
    assert actual == expected, (
        f"CellReconfigureEvent sequence mismatch in {event_dir}:\n" f"  expected: {expected}\n" f"  actual:   {actual}"
    )


def load_reconfigure_events(event_dir: Path) -> list[CellReconfigureEvent]:
    return [event for event in read_events(event_dir) if isinstance(event, CellReconfigureEvent)]
