from dataclasses import dataclass

from miles.utils.operation_contract import RegistrationKey


@dataclass
class TrainingStreamState:
    step: int = 0
    # True while the stream holds unstepped accumulated gradients.
    dirty: bool = False


class GradientWindowTracker:
    """Step/dirty authority for every live training stream."""

    def __init__(self) -> None:
        self._streams: dict[RegistrationKey, TrainingStreamState] = {}

    def _stream(self, key: RegistrationKey) -> TrainingStreamState:
        return self._streams.setdefault(key, TrainingStreamState())

    # ------------------------------ lifecycle ------------------------------

    def open(self, key: RegistrationKey) -> None:
        """Start tracking a registration's stream (idempotent)."""
        self._stream(key)

    def close(self, key: RegistrationKey) -> None:
        """Drop a retired registration's stream state."""
        self._streams.pop(key, None)

    # ------------------------------ queries ------------------------------

    def step_of(self, key: RegistrationKey) -> int:
        stream = self._streams.get(key)
        return stream.step if stream is not None else 0

    def is_dirty(self, key: RegistrationKey) -> bool:
        stream = self._streams.get(key)
        return stream is not None and stream.dirty

    # ------------------------------ transitions ------------------------------

    def mark_forward_backward_succeeded(self, key: RegistrationKey) -> None:
        self._stream(key).dirty = True

    def clear_after_executed_optim(self, key: RegistrationKey) -> None:
        self._stream(key).dirty = False

    def commit_step(self, key: RegistrationKey) -> int:
        stream = self._stream(key)
        stream.step += 1
        stream.dirty = False
        return stream.step

    def restore_step(self, key: RegistrationKey, step: int) -> None:
        stream = self._stream(key)
        stream.step = step
