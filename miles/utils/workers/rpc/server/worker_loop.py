from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)


class _DedicatedWorkerLoop:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="rpc-worker-loop", daemon=True)
        self._closed = False
        self._stop_requested = False
        self._thread.start()

    async def call(self, *, method: Callable[..., Any], kwargs: dict[str, Any], is_async: bool) -> Any:
        if self._closed:
            raise RuntimeError("The worker event loop is closed")
        future = asyncio.run_coroutine_threadsafe(
            self._invoke(method=method, kwargs=kwargs, is_async=is_async), self._loop
        )
        return await asyncio.wrap_future(future)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop))
        finally:
            self._loop.call_soon_threadsafe(self._stop)
            await asyncio.to_thread(self._thread.join)

    async def _invoke(self, *, method: Callable[..., Any], kwargs: dict[str, Any], is_async: bool) -> Any:
        if is_async:
            return await method(**kwargs)
        return method(**kwargs)

    async def _shutdown(self) -> None:
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._loop.shutdown_asyncgens()
        await self._loop.shutdown_default_executor()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        except BaseException:
            logger.error("The dedicated worker event loop crashed", exc_info=True)
        finally:
            if not self._stop_requested:
                try:
                    logger.critical("The dedicated worker event loop stopped unexpectedly; exiting the worker process")
                finally:
                    os._exit(1)
            self._loop.close()

    def _stop(self) -> None:
        self._stop_requested = True
        self._loop.stop()
