"""Healing-witness assertions over CellReconfigureEvent for the FT e2e scenarios.

These provide positive proof that cell reconfigures (shrinks and healings) actually
executed during a run, at the expected rollouts and on the expected cells -- so any
future change that makes healing silently disappear turns the e2e tests red instead
of green-by-comparing-two-fault-free-runs.
"""

import logging
from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import CellReconfigureEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class ExpectedReconfigure(FrozenStrictBaseModel):
    """Expected shape of one CellReconfigureEvent (healing iff healed_cell_indices non-empty)."""

    rollout_id: int
    src_cell_index: int | None
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]


def assert_reconfigure_events(event_dir: Path, *, expected: list[ExpectedReconfigure]) -> None:
    """Assert event_dir contains exactly the expected ordered CellReconfigureEvent sequence.

    Healing counts are compared first so a missing healing fails with an explicit
    "expected N healing event(s), got M" message. Quorum ids must additionally be the
    contiguous sequence 1..len(events): a gap means a reconfigure attempt failed and
    was retried, which the strictly-scripted scenarios never expect.
    """
    actual_events = load_reconfigure_events(event_dir)
    actual = [_shape_of(event) for event in actual_events]

    expected_num_healings = sum(1 for e in expected if e.healed_cell_indices)
    actual_num_healings = sum(1 for e in actual if e.healed_cell_indices)
    assert actual_num_healings == expected_num_healings, (
        f"Healing witness failed in {event_dir}: expected {expected_num_healings} healing event(s), "
        f"got {actual_num_healings} (actual reconfigure sequence: {actual})"
    )

    assert actual == expected, (
        f"CellReconfigureEvent sequence mismatch in {event_dir}:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )

    actual_quorum_ids = [event.quorum_id for event in actual_events]
    expected_quorum_ids = list(range(1, len(actual_events) + 1))
    assert actual_quorum_ids == expected_quorum_ids, (
        f"CellReconfigureEvent quorum ids in {event_dir} are {actual_quorum_ids}, expected the contiguous "
        f"sequence {expected_quorum_ids}; a gap means a reconfigure attempt failed and was retried"
    )

    print(f"Reconfigure witness assertion passed: {len(actual)} event(s) in {event_dir} match {expected}")


def assert_soak_reconfigure_events(event_dir: Path, *, num_successful_injections: int, num_cells: int) -> None:
    """Soak-run healing witness: successful injections imply healings, and the run must end fully healed.

    Injection timing is random, so no exact sequence is pinned. Instead:
    - if the injector reported >=1 successful injection, at least one healing event must exist;
    - the last reconfigure (if any) must restore full cell membership.
    """
    events = load_reconfigure_events(event_dir)
    healings = [event for event in events if event.healed_cell_indices]

    if num_successful_injections > 0:
        assert healings, (
            f"Healing witness failed in {event_dir}: the fault injector reported "
            f"{num_successful_injections} successful injection(s) but no healing event was emitted "
            f"(reconfigure events: {[_shape_of(event) for event in events]})"
        )

    if events:
        full_membership = list(range(num_cells))
        assert events[-1].alive_cell_indices_after == full_membership, (
            f"Soak run must end fully healed: last reconfigure event in {event_dir} left alive cells "
            f"{events[-1].alive_cell_indices_after}, expected {full_membership}"
        )

    print(
        f"Soak reconfigure witness assertion passed: {len(events)} reconfigure event(s) "
        f"({len(healings)} healing(s)) for {num_successful_injections} successful injection(s) in {event_dir}"
    )


def load_reconfigure_events(event_dir: Path) -> list[CellReconfigureEvent]:
    """Read all CellReconfigureEvents under event_dir, in emission order.

    All reconfigure events come from the single driver-side events.jsonl, so file order
    is emission order.
    """
    return [event for event in read_events(event_dir) if isinstance(event, CellReconfigureEvent)]


def _shape_of(event: CellReconfigureEvent) -> ExpectedReconfigure:
    return ExpectedReconfigure(
        rollout_id=event.rollout_id,
        src_cell_index=event.src_cell_index,
        healed_cell_indices=event.healed_cell_indices,
        alive_cell_indices_after=event.alive_cell_indices_after,
    )
