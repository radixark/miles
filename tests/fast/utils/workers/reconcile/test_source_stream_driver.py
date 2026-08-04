from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.fast.utils.workers.reconcile.utils import FakeSource, StreamEnd, StreamError, make_pod, pod_cell, settle

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import ParentKey, ReplaceEvent, UpsertEvent
from miles.utils.workers.reconcile.source_stream_driver import SourceStreamDriver


class DriverHarness:
    def __init__(self, *, source: FakeSource, retry_delay: float = 1.0) -> None:
        self.source = source
        self.store = ObjectStore(key_map=pod_cell)
        self.affected: list[set[ParentKey]] = []
        self.clock = FakeClock()
        self.driver = SourceStreamDriver(
            source=source,
            store=self.store,
            on_affected=self.affected.append,
            retry_delay=retry_delay,
            clock=self.clock,
        )
        self.task = asyncio.create_task(self.driver.run())

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.driver.aclose()


def replace(*pods: Any) -> ReplaceEvent:
    return ReplaceEvent(objects={pod.metadata.name: pod for pod in pods})


class TestSyncBarrier:
    async def test_wait_for_sync_blocks_until_the_first_replace(self):
        """The barrier is what keeps a half-filled store from being reconciled."""
        harness = DriverHarness(source=FakeSource())
        await settle()
        waiter = asyncio.create_task(harness.driver.wait_for_sync())
        await settle()
        assert not waiter.done()

        harness.source.emit(replace(make_pod("pod-0")))
        await settle()

        assert waiter.done()
        await harness.close()

    async def test_a_later_replace_keeps_the_barrier_open(self):
        """Once released the barrier stays released, so a relist never re-blocks a running loop."""
        harness = DriverHarness(source=FakeSource())
        await settle()
        harness.source.emit(replace(make_pod("pod-0")))
        await settle()

        harness.source.emit(replace())
        await settle()

        await asyncio.wait_for(harness.driver.wait_for_sync(), timeout=1)
        await harness.close()


class TestStreamProtocol:
    async def test_events_are_applied_to_the_store_and_reported(self):
        """Every event lands in the store before its parents are announced."""
        harness = DriverHarness(source=FakeSource())
        await settle()
        harness.source.emit(replace(make_pod("pod-0", cell="cell-a")))
        await settle()

        assert harness.affected == [{"cell-a"}]
        assert [pod.metadata.name for pod in harness.store.get_by_parent("cell-a")] == ["pod-0"]
        await harness.close()

    async def test_a_stream_not_opening_with_a_replace_is_reopened(self):
        """A stream whose first event is incremental cannot be trusted to describe the whole world."""
        harness = DriverHarness(source=FakeSource())
        await settle()
        waiter = asyncio.create_task(harness.driver.wait_for_sync())
        harness.source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()

        assert harness.source.closed_count == 1
        assert not waiter.done()

        await harness.clock.elapse(1.0)
        await settle()

        assert harness.source.open_count == 2
        assert harness.affected == []
        assert not waiter.done()
        waiter.cancel()
        await harness.close()


class TestReopen:
    async def test_a_stream_that_ends_is_reopened_after_the_retry_delay(self):
        """A source that returns is not a source that is finished."""
        harness = DriverHarness(source=FakeSource(), retry_delay=5.0)
        await settle()
        harness.source.emit(replace(make_pod("pod-0")))
        await settle()

        harness.source.emit(StreamEnd())
        await settle()
        assert harness.source.open_count == 1

        await harness.clock.elapse(5.0)
        await settle()
        assert harness.source.open_count == 2
        await harness.close()

    async def test_a_failing_stream_is_closed_before_the_next_one_opens(self):
        """The dead stream must not be left holding a connection while its replacement runs."""
        harness = DriverHarness(source=FakeSource())
        await settle()
        harness.source.emit(StreamError(RuntimeError("connection reset")))
        await settle()
        await harness.clock.elapse(1.0)
        await settle()

        assert harness.source.closed_count == 1
        assert harness.source.open_count == 2
        await harness.close()

    async def test_a_source_factory_that_raises_is_retried(self):
        """Failing to open at all is the same failure as failing mid-stream."""
        harness = DriverHarness(source=FakeSource(fail_calls=1))
        await settle()
        assert harness.source.open_count == 1

        await harness.clock.elapse(1.0)
        await settle()

        assert harness.source.open_count == 2
        await harness.close()


class TestClose:
    async def test_cancelling_the_driver_closes_the_stream_it_was_iterating(self):
        """Teardown must not leave the source generator suspended and holding its connection."""
        harness = DriverHarness(source=FakeSource())
        await settle()
        assert harness.source.closed_count == 0

        harness.task.cancel()
        await asyncio.gather(harness.task, return_exceptions=True)

        assert harness.source.closed_count == 1
        await harness.driver.aclose()

    async def test_aclose_without_a_stream_is_a_no_op(self):
        """Closing a driver that never opened anything is not an error."""
        harness = DriverHarness(source=FakeSource())
        await harness.close()

        with pytest.raises(asyncio.CancelledError):
            harness.task.result()
