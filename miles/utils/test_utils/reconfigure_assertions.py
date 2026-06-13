from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import CellReconfigureEvent
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
    actual = [ReconfigureInfo.from_event(event) for event in load_reconfigure_events(event_dir)]
    assert actual == expected, (
        f"CellReconfigureEvent sequence mismatch in {event_dir}:\n" f"  expected: {expected}\n" f"  actual:   {actual}"
    )


def assert_soak_reconfigure_events(event_dir: Path, *, num_successful_injections: int) -> None:
    events = load_reconfigure_events(event_dir)
    healings = [event for event in events if event.healed_cell_indices]

    assert num_successful_injections > 0, (
        f"Soak proved nothing in {event_dir}: the fault injector reported zero successful injections, "
        f"so no fault tolerance was exercised"
    )
    assert healings, (
        f"Healing witness failed in {event_dir}: the fault injector reported "
        f"{num_successful_injections} successful injection(s) but no healing event was emitted "
        f"(reconfigure events: {[ReconfigureInfo.from_event(event) for event in events]})"
    )

    print(
        f"Soak reconfigure witness assertion passed: {len(events)} reconfigure event(s) "
        f"({len(healings)} healing(s)) for {num_successful_injections} successful injection(s) in {event_dir}"
    )


def load_reconfigure_events(event_dir: Path) -> list[CellReconfigureEvent]:
    return [event for event in read_events(event_dir) if isinstance(event, CellReconfigureEvent)]
