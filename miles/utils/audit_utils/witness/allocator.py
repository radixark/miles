from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import WitnessAllocateIdEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessInfo(FrozenStrictBaseModel):
    witness_ids: list[int]
    stale_ids: list[int]


class WitnessIdAllocator:
    def __init__(self, *, buffer_size: int) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size ({buffer_size}) must be positive.")
        self._buffer_size = buffer_size
        self._counter: int = 0

    @property
    def counter(self) -> int:
        return self._counter

    def resume(self, counter: int) -> None:
        assert counter >= self._counter
        self._counter = counter

    def allocate(self, num_ids: int) -> WitnessInfo:
        if num_ids < 0:
            raise ValueError(f"num_ids ({num_ids}) must be non-negative.")
        assert num_ids <= self._buffer_size, (
            f"num_ids ({num_ids}) exceeds buffer_size ({self._buffer_size}). " f"Increase --witness-buffer-size."
        )
        ids = [(self._counter + i) % self._buffer_size for i in range(num_ids)]
        stale_ids = _compute_stale_ids(
            keep_count=int(self._buffer_size * 0.7),
            counter=self._counter + num_ids,
            buffer_size=self._buffer_size,
        )
        self._counter += num_ids
        return WitnessInfo(witness_ids=ids, stale_ids=stale_ids)


def read_persisted_witness_counter(event_dir: Path) -> int:
    """Recover the allocator counter from the (checkpoint-restored) event directory."""
    events = read_events(event_dir)
    return max((e.counter_after for e in events if isinstance(e, WitnessAllocateIdEvent)), default=0)


def _compute_stale_ids(*, keep_count: int, counter: int, buffer_size: int) -> list[int]:
    if counter == 0:
        return []
    num_stale = buffer_size - min(keep_count, counter, buffer_size)
    if num_stale == 0:
        return []

    head = counter % buffer_size
    return [(head + i) % buffer_size for i in range(num_stale)]
