from __future__ import annotations

import abc
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_WAIT_DEAD_PROBE_INTERVAL_SECONDS = 1.0


class WorkerUnreachableError(Exception):
    pass


class BaseWorkerHandle(abc.ABC):
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float) -> None: ...

    async def wait_dead(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if await self._probe_is_dead():
                return
            if time.monotonic() >= deadline:
                logger.error("Timed out after %.0fs waiting for %r to die; proceeding anyway", timeout, self)
                return
            await asyncio.sleep(_WAIT_DEAD_PROBE_INTERVAL_SECONDS)

    @abc.abstractmethod
    async def _probe_is_dead(self) -> bool: ...
