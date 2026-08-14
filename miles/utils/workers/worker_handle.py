from __future__ import annotations

import abc


class WorkerUnreachableError(Exception):
    pass


class BaseWorkerHandle(abc.ABC):
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float) -> None: ...
