import asyncio

import pytest


class TestCallerCancellation:
    async def test_cancelling_the_await_does_not_stop_the_worker(self, server, make_handle, tag):
        """Cancelling on the caller side abandons the result, it does not cancel the work."""
        handle = make_handle(server)
        pending = asyncio.create_task(handle.demo_count_after_sleep(tag=tag, seconds=2.0))
        await asyncio.sleep(0.5)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await asyncio.sleep(2.5)
        assert await make_handle(server).report_counter(tag=tag) == 1

    async def test_cancelling_one_call_leaves_the_others(self, handle, tag):
        """Cancelling one call does not disturb concurrent calls on the same handle."""
        cancelled = asyncio.create_task(handle.demo_sleep_async(tag=f"{tag}c", seconds=3.0))
        kept = asyncio.create_task(handle.demo_sleep_async(tag=f"{tag}k", seconds=1.0))
        await asyncio.sleep(0.3)

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        assert await kept == f"{tag}k"

    async def test_handle_is_reusable_after_a_cancelled_call(self, handle, tag):
        """A cancelled call leaves the handle in a usable state."""
        pending = asyncio.create_task(handle.demo_sleep_async(tag=tag, seconds=3.0))
        await asyncio.sleep(0.3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        assert await handle.demo_sync(a=1, b=1) == 2

    async def test_cancelling_wait_ready_is_clean(self, make_handle):
        """Cancelling wait_ready against a dead server raises CancelledError promptly."""
        handle = make_handle("http://127.0.0.1:9")
        pending = asyncio.create_task(handle.wait_ready(timeout=30.0))
        await asyncio.sleep(0.5)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    async def test_result_of_a_cancelled_call_stays_available(self, server, make_handle, raw, tag):
        """The outcome the caller walked away from is still on the server."""
        handle = make_handle(server)
        pending = asyncio.create_task(handle.demo_sleep_async(tag=tag, seconds=2.0))
        await asyncio.sleep(0.3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await asyncio.sleep(2.5)
        events = await make_handle(server).report_events()
        assert any(event.tag == tag and event.phase == "end" for event in events)
