import asyncio
import time

import pytest

from miles.utils.workers.rpc.client import call as client_module


async def _submit(raw, method: str, call_id: str, query: dict):
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query})


class TestLongPollTiming:
    async def test_poll_returns_as_soon_as_the_call_finishes(self, raw, tag):
        """A long poll is woken by completion instead of waiting out its timeout."""
        await _submit(raw, "demo_sleep_async", tag, {"tag": tag, "seconds": 1.0})

        started = time.monotonic()
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 30.0})).json()
        elapsed = time.monotonic() - started

        assert body["status"] == "success"
        assert 0.5 < elapsed < 10.0

    async def test_poll_waits_out_its_timeout_while_pending(self, raw, handle, tag):
        """A poll on a still-running call blocks for the timeout it asked for."""
        await _submit(raw, "demo_block_async", tag, {"tag": tag})

        started = time.monotonic()
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 2.0})).json()
        elapsed = time.monotonic() - started

        assert body["status"] == "pending"
        assert 1.5 < elapsed < 8.0
        await handle.release(tag=tag)

    async def test_zero_timeout_does_not_block(self, raw, handle, tag):
        """A zero timeout is a non-blocking status check."""
        await _submit(raw, "demo_block_async", tag, {"tag": tag})

        started = time.monotonic()
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 0.0})).json()

        assert body["status"] == "pending" and time.monotonic() - started < 2.0
        await handle.release(tag=tag)

    async def test_result_completed_before_first_poll_is_not_lost(self, raw, tag):
        """A call that finishes before anyone polls keeps its outcome."""
        await _submit(raw, "demo_sync", tag, {"a": 1, "b": 2})
        await asyncio.sleep(1.0)

        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 0.0})).json()
        assert body == {"status": "success", "result": 3, "error": None}

    async def test_many_waiters_are_all_woken(self, raw, handle, tag):
        """Every concurrent poller of one call is released when it completes."""
        await _submit(raw, "demo_block_async", tag, {"tag": tag})

        polls = [raw.get(f"/v1/calls/{tag}", params={"timeout": 15.0}) for _ in range(20)]
        pollers = asyncio.gather(*polls)
        await asyncio.sleep(0.5)
        await handle.release(tag=tag)

        bodies = [response.json() for response in await pollers]
        assert all(body["status"] == "success" for body in bodies)


class TestClientPollLoop:
    async def test_call_spanning_several_poll_windows(self, server, make_handle, monkeypatch, tag):
        """A call outliving one poll window is stitched together by the client loop."""
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 1.0)

        handle = make_handle(server, call_timeout_seconds=60.0)
        assert await handle.demo_sleep_async(tag=tag, seconds=4.0) == tag

    async def test_client_timeout_does_not_cancel_the_worker(self, server, make_handle, tag):
        """Giving up on the client side leaves the server running the call to completion."""
        impatient = make_handle(server, call_timeout_seconds=1.0)
        with pytest.raises(TimeoutError):
            await impatient.demo_count_after_sleep(tag=tag, seconds=3.0)

        await asyncio.sleep(3.5)
        assert await make_handle(server).report_counter(tag=tag) == 1

    async def test_poll_requests_stay_within_the_call_budget(self, server, make_handle, proxy_to, handle, tag):
        """Each poll asks for no more than the remaining call budget."""
        proxy = await proxy_to()
        impatient = make_handle(proxy, call_timeout_seconds=2.0)

        pending = asyncio.create_task(impatient.demo_block_async(tag=tag))
        with pytest.raises(TimeoutError):
            await pending

        polls = [r for r in proxy.requests if r.verb == "GET" and "/v1/calls/" in r.path]
        assert polls
        await handle.release(tag=tag)


class TestCallTimeout:
    async def test_pending_call_raises_timeout(self, server, make_handle, handle, tag):
        """A call still pending past its budget raises TimeoutError."""
        impatient = make_handle(server, call_timeout_seconds=2.0)

        started = time.monotonic()
        with pytest.raises(TimeoutError) as exc_info:
            await impatient.demo_block_async(tag=tag)

        assert 1.0 < time.monotonic() - started < 10.0
        assert "demo_block_async" in str(exc_info.value)
        await handle.release(tag=tag)

    async def test_generous_timeout_does_not_interrupt(self, server, make_handle, tag):
        """A slow call under a generous budget completes normally."""
        patient = make_handle(server, call_timeout_seconds=60.0)
        assert await patient.demo_sleep_sync(tag=tag, seconds=3.0) == tag

    async def test_timeout_message_names_the_call(self, server, make_handle, handle, tag):
        """The timeout error identifies the call so the caller can investigate it."""
        impatient = make_handle(server, call_timeout_seconds=1.0)

        with pytest.raises(TimeoutError) as exc_info:
            await impatient.demo_block_async(tag=tag)

        assert "call id" in str(exc_info.value)
        await handle.release(tag=tag)
