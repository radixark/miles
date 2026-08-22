from __future__ import annotations

import abc
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_WAIT_DEAD_PROBE_INTERVAL_SECONDS = 1.0


class WorkerUnreachableError(Exception):
    pass


class WorkerStillBusyError(Exception):
    pass


class BaseWorkerHandle(abc.ABC):
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None: ...

    async def wait_idle(self, *, timeout: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot tell whether the worker is running a call")

    async def wait_dead(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if await self.probe_is_dead():
                return
            if time.monotonic() >= deadline:
                logger.error("Timed out after %.0fs waiting for %r to die; proceeding anyway", timeout, self)
                return
            await asyncio.sleep(_WAIT_DEAD_PROBE_INTERVAL_SECONDS)

    @abc.abstractmethod
    async def probe_is_dead(self) -> bool: ...
