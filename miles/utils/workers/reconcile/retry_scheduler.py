# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

from miles.utils.test_utils.clock import Clock

KeyT = TypeVar("KeyT", bound=Hashable)

POLL_INTERVAL = 1.0


@dataclass
class _RetryInfo:
    failures: int
    retry_at: float | None


class RetryScheduler(Generic[KeyT]):
    def __init__(
        self,
        *,
        on_retry: Callable[[KeyT], None],
        failure_base_delay: float,
        failure_max_delay: float,
        clock: Clock,
    ) -> None:
        assert failure_base_delay > 0, f"{failure_base_delay=} must be positive"
        assert failure_max_delay >= failure_base_delay, f"{failure_max_delay=} must be >= {failure_base_delay=}"

        self._on_retry = on_retry
        self._failure_base_delay = failure_base_delay
        self._failure_max_delay = failure_max_delay
        self._max_backoff_exponent = max(0, math.ceil(math.log2(failure_max_delay / failure_base_delay)))
        self._clock = clock

        self._infos: dict[KeyT, _RetryInfo] = {}
        self._poller = asyncio.create_task(self._poll())
        self._shutdown = False

    def note_failure(self, key: KeyT) -> None:
        if self._shutdown:
            return

        info = self._infos.get(key)
        failures = 1 if info is None else info.failures + 1
        exponent = min(failures - 1, self._max_backoff_exponent)
        delay = min(self._failure_base_delay * 2**exponent, self._failure_max_delay)

        self._infos[key] = _RetryInfo(failures=failures, retry_at=self._clock.time() + delay)

    def note_success(self, key: KeyT) -> None:
        self._infos.pop(key, None)

    async def shutdown(self) -> None:
        self._shutdown = True
        self._infos = {}

        self._poller.cancel()
        await asyncio.gather(self._poller, return_exceptions=True)

    async def _poll(self) -> None:
        while True:
            await self._clock.sleep(POLL_INTERVAL)

            now = self._clock.time()
            due = [key for key, info in self._infos.items() if info.retry_at is not None and info.retry_at <= now]
            for key in due:
                self._infos[key].retry_at = None
                self._on_retry(key)
