from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import ray

from miles.utils.workers.worker_handle import (
    _WAIT_DEAD_PROBE_INTERVAL_SECONDS,
    BaseWorkerHandle,
    WorkerUnreachableError,
)

logger = logging.getLogger(__name__)


class RayWorkerHandle(BaseWorkerHandle):
    def __init__(self, actor_handle: ray.actor.ActorHandle) -> None:
        self._actor_handle = actor_handle

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return await getattr(self._actor_handle, name).remote(*args, **kwargs)
            except ray.exceptions.RayActorError as e:
                raise WorkerUnreachableError(f"Worker died or is unreachable when calling {name!r}: {e!r}") from e

        return call

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        del allow_server_uuid_change

        try:
            await asyncio.wait_for(self._actor_handle.__ray_ready__.remote(), timeout=timeout)
        except ray.exceptions.RayActorError as e:
            raise WorkerUnreachableError(f"Worker died before becoming ready: {e!r}") from e
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise WorkerUnreachableError(f"Worker not ready within {timeout}s") from e

    async def wait_idle(self, *, timeout: float) -> None:
        raise NotImplementedError(
            "a ray actor does not track the calls it is running, so nobody can wait for it to go idle; only the "
            "rpc communication backend answers this"
        )

    async def probe_is_dead(self) -> bool:
        try:
            await asyncio.wait_for(
                self._actor_handle.__ray_ready__.remote(), timeout=_WAIT_DEAD_PROBE_INTERVAL_SECONDS
            )
        except ray.exceptions.ActorUnavailableError as e:
            logger.info("Worker death probe was inconclusive; the actor is temporarily unavailable: %r", e)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return True
        except (TimeoutError, asyncio.TimeoutError):
            pass
        return False
