from __future__ import annotations

import asyncio
from collections import deque

from tests.fast.utils.workers.reconcile.utils import settle

from miles.utils.workers.reconcile.work_queue import WorkQueue


class TestWorkQueue:
    async def test_shutdown_wakes_a_worker_blocked_on_an_empty_queue(self):
        """A worker parked on get() must be released by shutdown, not only by cancellation."""
        queue: WorkQueue[str] = WorkQueue()
        got: list[str | None] = []

        async def consume() -> None:
            got.append(await queue.get())

        task = asyncio.create_task(consume())
        await settle()
        assert got == []

        queue.shutdown()
        await task
        assert got == [None]

    async def test_a_key_added_after_shutdown_is_dropped(self):
        """Nothing enters the queue once it is shut down, not even to sit there unread."""
        queue: WorkQueue[str] = WorkQueue()
        queue.shutdown()
        queue.add("cell-a")

        assert await queue.get() is None
        assert queue._keys == deque()

    async def test_a_requeued_key_keeps_its_original_position(self):
        """Re-adding a queued key must not push it behind keys that arrived later."""
        queue: WorkQueue[str] = WorkQueue()
        queue.add("cell-a")
        queue.add("cell-b")
        queue.add("cell-a")

        assert [await queue.get(), await queue.get()] == ["cell-a", "cell-b"]

    async def test_a_duplicate_key_is_dispatched_once(self):
        """The queue is a dedup set, not a multiset."""
        queue: WorkQueue[str] = WorkQueue()
        queue.add("cell-a")
        queue.add("cell-a")
        queue.add("cell-b")

        assert [await queue.get(), await queue.get()] == ["cell-a", "cell-b"]
        assert queue._keys == deque()
