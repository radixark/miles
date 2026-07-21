import time
from dataclasses import dataclass


@dataclass(frozen=True)
class HeartbeatStatus:
    last_active_timestamp: float
    bump_count: int


class SimpleHeartbeat:
    def __init__(self) -> None:
        self._status = HeartbeatStatus(last_active_timestamp=time.time(), bump_count=0)

    def bump(self) -> None:
        self._status = HeartbeatStatus(
            last_active_timestamp=time.time(),
            bump_count=self._status.bump_count + 1,
        )

    def status(self) -> HeartbeatStatus:
        return self._status
