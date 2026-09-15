# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from miles.utils.workers.reconcile.object_store import KeyMapFn, ObjectStore
from miles.utils.workers.reconcile.source_event import ParentKey, SourceEvent, SourceWatchFn
from miles.utils.workers.reconcile.work_queue import WorkQueue

logger = logging.getLogger(__name__)

ReconcileFn = Callable[[ParentKey], Awaitable[None]]


class ReconcileLoop:
    """A source stream feeds a store; every changed parent key is reconciled once, level-triggered.

    Args:
        source: Opens an async generator of `SourceEvent`.
        reconcile: Takes a parent key only and re-derives from `get_by_parent()`, whose objects
            are read-only. Must be idempotent, since delivery is at-least-once, and must not block
            on I/O, since one worker serves every parent key.
    """

    def __init__(
        self,
        *,
        source: SourceWatchFn,
        reconcile: ReconcileFn,
        key_map: KeyMapFn | None = None,
    ) -> None:
        self._source = source
        self._reconcile = reconcile

        self._store = ObjectStore(key_map=key_map)
        self._queue: WorkQueue[ParentKey] = WorkQueue()

        self._start_called = False
        self._tasks: list[asyncio.Task[None]] = []
        self._stream: AsyncGenerator[SourceEvent, None] | None = None

    async def __aenter__(self) -> ReconcileLoop:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def start(self) -> None:
        assert not self._start_called, "ReconcileLoop.start() must be called exactly once"
        self._start_called = True

        self._tasks = [asyncio.create_task(self._worker_loop()), asyncio.create_task(self._consume_loop())]

    async def stop(self) -> None:
        assert self._start_called, "ReconcileLoop.stop() must come after start()"
        assert asyncio.current_task() not in self._tasks, (
            "ReconcileLoop.stop() waits for the worker, so it cannot be awaited from inside reconcile; "
            "call asyncio.create_task(loop.stop()) instead"
        )

        self._queue.shutdown()

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await self._aclose_stream()
        self._tasks = []

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        return self._store.get_by_parent(parent_key)

    def _enqueue_all(self, parent_keys: set[ParentKey]) -> None:
        for parent_key in sorted(parent_keys):
            self._queue.add(parent_key)

    async def _worker_loop(self) -> None:
        while True:
            parent_key = await self._queue.get()
            if parent_key is None:
                return
            try:
                await self._reconcile(parent_key)
            except Exception:
                logger.error(f"ReconcileLoop reconcile failed {parent_key=}", exc_info=True)

    async def _consume_loop(self) -> None:
        stream = self._source()
        self._stream = stream
        try:
            async for event in stream:
                self._enqueue_all(self._store.handle_event(event))
        except Exception:
            logger.error("ReconcileLoop source stream failed, no further events will arrive", exc_info=True)
            return
        logger.error("ReconcileLoop source stream ended, no further events will arrive")

    async def _aclose_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            await stream.aclose()
