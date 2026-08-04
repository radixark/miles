# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.object_store import KeyMapFn, ObjectStore
from miles.utils.workers.reconcile.source_event import ParentKey, SourceWatchFn
from miles.utils.workers.reconcile.source_stream_driver import SourceStreamDriver
from miles.utils.workers.reconcile.work_queue import WorkQueue

logger = logging.getLogger(__name__)

ReconcileFn = Callable[[ParentKey], Awaitable[None]]


class ReconcileLoop:
    """A source stream feeds a store; every changed parent key is reconciled once, level-triggered.

    Args:
        source: Opens every stream with a `ReplaceEvent` carrying the whole world, and is
            restartable: a stream that ends or fails is reopened.
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
        source_retry_delay: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        assert source_retry_delay > 0, f"{source_retry_delay=} must be positive"

        self._reconcile = reconcile
        self._clock = clock or RealClock()

        self._store = ObjectStore(key_map=key_map)
        self._queue: WorkQueue[ParentKey] = WorkQueue()
        self._driver = SourceStreamDriver(
            source=source,
            store=self._store,
            on_affected=self._enqueue_all,
            retry_delay=source_retry_delay,
            clock=self._clock,
        )

        self._start_called = False
        self._tasks: list[asyncio.Task[None]] = []

    async def __aenter__(self) -> ReconcileLoop:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def start(self) -> None:
        assert not self._start_called, "ReconcileLoop.start() must be called exactly once"
        self._start_called = True

        driver_task = asyncio.create_task(self._driver.run())
        try:
            await self._driver.wait_for_sync()
        except asyncio.CancelledError:
            driver_task.cancel()
            await asyncio.gather(driver_task, return_exceptions=True)
            raise

        self._tasks = [asyncio.create_task(self._worker_loop()), driver_task]

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

        await self._driver.aclose()
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
