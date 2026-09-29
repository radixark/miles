# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable

from miles.utils.test_utils.clock import Clock
from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import ParentKey, ReplaceEvent, SourceEvent, SourceWatchFn

logger = logging.getLogger(__name__)


class SourceStreamDriver:
    def __init__(
        self,
        *,
        source: SourceWatchFn,
        store: ObjectStore,
        on_affected: Callable[[set[ParentKey]], None],
        retry_delay: float,
        clock: Clock,
    ) -> None:
        self._source = source
        self._store = store
        self._on_affected = on_affected
        self._retry_delay = retry_delay
        self._clock = clock
        self._stream: AsyncGenerator[SourceEvent, None] | None = None
        self._synced = asyncio.Event()

    async def run(self) -> None:
        while True:
            stream: AsyncGenerator[SourceEvent, None] | None = None
            try:
                stream = self._source()
                self._stream = stream
                await self._pump(stream)
                logger.warning("SourceStreamDriver source stream ended, reopening")
            except Exception:
                logger.error("SourceStreamDriver source stream failed, reopening", exc_info=True)
            finally:
                self._stream = None
                await _aclose_logging_failure(stream)
            await self._clock.sleep(self._retry_delay)

    async def wait_for_sync(self) -> None:
        await self._synced.wait()

    async def aclose(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            await stream.aclose()

    async def _pump(self, stream: AsyncGenerator[SourceEvent, None]) -> None:
        first = True
        async for event in stream:
            if first and not isinstance(event, ReplaceEvent):
                raise RuntimeError(f"A source stream must open with ReplaceEvent, got {event=}")
            first = False
            self._on_affected(self._store.handle_event(event))
            if isinstance(event, ReplaceEvent):
                self._synced.set()


async def _aclose_logging_failure(stream: AsyncGenerator[SourceEvent, None] | None) -> None:
    if stream is None:
        return
    try:
        await stream.aclose()
    except Exception:
        logger.error("SourceStreamDriver failed to close a source stream", exc_info=True)
