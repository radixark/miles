from __future__ import annotations

import abc
import asyncio
import heapq
import time
from typing import NamedTuple


class Clock(abc.ABC):
    @abc.abstractmethod
    def time(self) -> float: ...

    @abc.abstractmethod
    async def sleep(self, seconds: float) -> None: ...


class RealClock(Clock):
    def time(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


_DRAIN_ITERATIONS = 20


class _Waiter(NamedTuple):
    target: float
    seq: int
    future: asyncio.Future[None]


class FakeClock(Clock):
    """Deterministic clock for testing async time-dependent code.

    ``sleep()`` suspends the caller until ``elapse()`` advances the clock
    past the target time. This gives tests precise control over which
    sleeps resolve and when.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._waiters: list[_Waiter] = []
        self._counter: int = 0

    def time(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds < 0:
            return

        target = self._now + seconds
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._counter += 1
        heapq.heappush(self._waiters, _Waiter(target=target, seq=self._counter, future=future))
        self._resolve_ready()
        await future

    async def elapse(self, seconds: float) -> None:
        assert seconds >= 0, f"Cannot elapse negative time: {seconds}"
        self._now += seconds
        self._resolve_ready()
        # Drain: yield enough times for resolved coroutines to run through
        # sync code and register their next sleep.
        for _ in range(_DRAIN_ITERATIONS):
            await asyncio.sleep(0)
            self._resolve_ready()

    def _resolve_ready(self) -> None:
        while self._waiters and self._waiters[0].target <= self._now:
            waiter = heapq.heappop(self._waiters)
            if not waiter.future.done():
                waiter.future.set_result(None)

    @property
    def pending_count(self) -> int:
        return sum(1 for w in self._waiters if not w.future.done())
