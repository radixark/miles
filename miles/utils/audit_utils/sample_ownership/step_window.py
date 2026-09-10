from collections import deque
from datetime import datetime


class SampleOwnershipStepWindow:
    def __init__(self, grace_steps: int) -> None:
        if grace_steps < 0:
            raise ValueError("grace_steps must be non-negative")
        self._grace_steps = grace_steps
        self._step_starts: deque[datetime] = deque(maxlen=grace_steps)

    def complete_step(self, *, started_at: datetime) -> None:
        self._step_starts.append(started_at)

    def mature_before(self, *, now: datetime) -> datetime | None:
        if self._grace_steps == 0:
            return now
        if len(self._step_starts) < self._grace_steps:
            return None
        return self._step_starts[0]
