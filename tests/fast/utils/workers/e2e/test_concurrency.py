import asyncio
import time


def _span(events, tag):
    start = next(e for e in events if e.tag == tag and e.phase == "start")
    end = next(e for e in events if e.tag == tag and e.phase == "end")
    return start.at, end.at


def _overlaps(first, second) -> bool:
    return first[0] < second[1] and second[0] < first[1]


class TestSyncSerialization:
    async def test_same_group_calls_do_not_overlap(self, handle, tag):
        """Two sync calls in one concurrency group run strictly one after another."""
        first = asyncio.create_task(handle.demo_sleep_sync(tag=f"{tag}a", seconds=0.4))
        await asyncio.sleep(0.1)
        second = asyncio.create_task(handle.demo_sleep_sync(tag=f"{tag}b", seconds=0.4))
        await asyncio.gather(first, second)

        events = await handle.report_events()
        assert not _overlaps(_span(events, f"{tag}a"), _span(events, f"{tag}b"))

    async def test_same_group_keeps_submission_order(self, handle, tag):
        """Queued sync calls execute in the order they were accepted."""
        await asyncio.gather(*[handle.demo_sleep_sync(tag=f"{tag}{i}", seconds=0.05) for i in range(4)])

        events = await handle.report_events()
        starts = [e.tag for e in events if e.phase == "start" and e.tag.startswith(tag)]
        assert starts == sorted(starts, key=lambda name: int(name[len(tag) :]))

    async def test_same_group_uses_one_thread(self, handle, tag):
        """A concurrency group is backed by a single executor thread."""
        names = await asyncio.gather(*[handle.demo_instant_sync(tag=f"{tag}{i}") for i in range(5)])
        assert len(set(names)) == 1 and names[0].startswith("rpc-")

    async def test_blocked_group_drains_after_release(self, handle, tag):
        """A blocked group resumes its backlog once the blocker is released."""
        blocked = asyncio.create_task(handle.demo_block_sync(tag=f"{tag}block"))
        await asyncio.sleep(0.2)
        queued = asyncio.create_task(handle.demo_instant_sync(tag=f"{tag}queued"))

        await asyncio.sleep(0.2)
        assert not queued.done(), "a queued call must wait for its group"

        await handle.release(tag=f"{tag}block")
        await asyncio.gather(blocked, queued)


class TestDeclaredConcurrencyGroups:
    async def test_declared_group_method_is_callable(self, handle, tag):
        """An @rpc-decorated method is exposed and runs on an executor thread."""
        assert (await handle.demo_instant_on_left(tag=tag)).startswith("rpc-left")

    async def test_same_declared_group_serializes(self, handle, tag):
        """Two calls declaring the same group run one after another, never overlapping."""
        first = asyncio.create_task(handle.demo_sleep_on_left(tag=f"{tag}a", seconds=0.4))
        await asyncio.sleep(0.1)
        second = asyncio.create_task(handle.demo_sleep_on_left(tag=f"{tag}b", seconds=0.4))
        await asyncio.gather(first, second)

        events = await handle.report_events()
        assert not _overlaps(_span(events, f"{tag}a"), _span(events, f"{tag}b"))

    async def test_distinct_declared_groups_run_in_parallel(self, handle, tag):
        """Two sync methods in different declared groups overlap, each on its own thread."""
        started = time.monotonic()
        threads = await asyncio.gather(
            handle.demo_sleep_on_left(tag=f"{tag}left", seconds=0.5),
            handle.demo_sleep_on_right(tag=f"{tag}right", seconds=0.5),
        )
        assert time.monotonic() - started < 0.9
        assert threads[0].startswith("rpc-left") and threads[1].startswith("rpc-right")

    async def test_declared_group_is_isolated_from_the_default_group(self, handle, tag):
        """Blocking a declared group leaves the default group free."""
        blocked = asyncio.create_task(handle.demo_block_on_left(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_sync(a=1, b=1) == 2

        await handle.release(tag=tag)
        await blocked

    async def test_default_rpc_decorator_keeps_async_method_on_loop(self, handle, tag):
        """The default rpc decorator keeps an async method on the event loop."""
        assert await handle.demo_async_on_left(tag=tag) == "MainThread"

    async def test_async_method_ignores_blocked_named_sync_group(self, handle, tag):
        """An async method remains free while a named sync group is blocked."""
        blocked = asyncio.create_task(handle.demo_block_on_left(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_async_on_left(tag=f"{tag}free") == "MainThread"

        await handle.release(tag=tag)
        await blocked


class TestGroupIsolation:
    async def test_sync_and_async_run_in_parallel(self, handle, tag):
        """A sync call on an executor and an async call on the loop overlap in time."""
        started = time.monotonic()
        await asyncio.gather(
            handle.demo_sleep_sync(tag=f"{tag}default", seconds=0.5),
            handle.demo_sleep_async(tag=f"{tag}async", seconds=0.5),
        )
        assert time.monotonic() - started < 0.9

    async def test_blocked_sync_group_leaves_the_event_loop_free(self, handle, tag, raw):
        """A blocked sync call must not stall the server's event loop."""
        blocked = asyncio.create_task(handle.demo_block_sync(tag=tag))
        await asyncio.sleep(0.2)

        started = time.monotonic()
        response = await raw.get("/v1/health")
        assert response.status_code == 200 and time.monotonic() - started < 2.0

        await handle.release(tag=tag)
        await blocked

    async def test_blocked_sync_group_leaves_async_methods_free(self, handle, tag):
        """Async methods keep running while a sync group is blocked."""
        blocked = asyncio.create_task(handle.demo_block_sync(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_instant_async(tag=f"{tag}free") == "MainThread"

        await handle.release(tag=tag)
        await blocked

    async def test_blocked_async_call_leaves_others_free(self, handle, tag):
        """A suspended async call does not stall other calls."""
        blocked = asyncio.create_task(handle.demo_block_async(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_sync(a=1, b=1) == 2

        await handle.release(tag=tag)
        await blocked


class TestAsyncExecution:
    async def test_async_methods_run_on_the_loop_thread(self, handle, tag):
        """Async methods run on the event loop rather than an executor thread."""
        assert await handle.demo_instant_async(tag=tag) == "MainThread"

    async def test_sync_methods_run_off_the_loop_thread(self, handle, tag):
        """Sync methods run on a worker thread, never on the loop."""
        assert (await handle.demo_instant_sync(tag=tag)).startswith("rpc-")

    async def test_async_calls_interleave(self, handle, tag):
        """Twenty concurrent async sleeps finish concurrently, not serially."""
        started = time.monotonic()
        await asyncio.gather(*[handle.demo_sleep_async(tag=f"{tag}{i}", seconds=0.3) for i in range(20)])
        assert time.monotonic() - started < 3.0
