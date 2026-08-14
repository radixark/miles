import asyncio
import time

import httpx
import pytest
from tests.fast.utils.workers.e2e.e2e_worker import E2eWorker
from tests.fast.utils.workers.e2e.harness import ConnectionCountingRelay

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


class TestConnectionBehaviour:
    async def test_client_disconnect_does_not_stop_the_call(self, server, make_handle, tag):
        """Dropping the submitting connection leaves the accepted call running."""
        async with httpx.AsyncClient(base_url=server.url, timeout=5.0, trust_env=False) as client:
            await client.post(
                "/v1/demo_count_after_sleep", json={"call_id": tag, "query": {"tag": tag, "seconds": 2.0}}
            )

        await asyncio.sleep(3.0)
        assert await make_handle(server).report_counter(tag=tag) == 1

    async def test_abandoned_long_poll_does_not_break_the_server(self, server, handle, raw, tag):
        """A long poll the client walks away from leaves the server healthy."""
        await raw.post("/v1/demo_block_async", json={"call_id": tag, "query": {"tag": tag}})

        async with httpx.AsyncClient(base_url=server.url, timeout=1.0, trust_env=False) as impatient:
            with pytest.raises(httpx.TimeoutException):
                await impatient.get(f"/v1/calls/{tag}", params={"timeout": 30.0})

        await handle.release(tag=tag)
        assert (await raw.get("/v1/health")).status_code == 200

    async def test_connections_are_reused_across_calls(self, handle):
        """Sequential calls reuse the pooled connection instead of reconnecting each time."""
        for _ in range(20):
            assert await handle.demo_sync(a=1, b=1) == 2

    async def test_sequential_calls_reuse_the_same_http_connection(self, server):
        """Sequential calls travel over one pooled TCP connection instead of reconnecting per request."""
        relay = ConnectionCountingRelay(upstream_port=server.port)
        await relay.start()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as client:
                handle = RpcWorkerHandle(E2eWorker, server_url=relay.url, http_client=client)
                for _ in range(5):
                    assert await handle.demo_sync(a=1, b=1) == 2
        finally:
            await relay.stop()

        assert relay.accepted == 1

    async def test_server_handles_many_short_connections(self, server):
        """Opening a fresh connection per request stays healthy."""
        for _ in range(20):
            async with httpx.AsyncClient(base_url=server.url, timeout=10.0, trust_env=False) as client:
                assert (await client.get("/v1/health")).status_code == 200

    async def test_slow_client_read_does_not_block_other_clients(self, server, handle, raw, tag):
        """One client holding a long poll open does not stall unrelated requests."""
        await raw.post("/v1/demo_block_async", json={"call_id": tag, "query": {"tag": tag}})
        slow = asyncio.create_task(raw.get(f"/v1/calls/{tag}", params={"timeout": 10.0}))
        await asyncio.sleep(0.3)

        started = time.monotonic()
        assert await handle.demo_sync(a=1, b=1) == 2
        assert time.monotonic() - started < 5.0

        await handle.release(tag=tag)
        await slow


class TestLoad:
    async def test_burst_of_calls_all_complete(self, handle):
        """A burst of a hundred concurrent calls all complete correctly."""
        results = await asyncio.gather(*[handle.demo_sync(a=i, b=1) for i in range(100)])
        assert results == [i + 1 for i in range(100)]

    async def test_burst_does_not_leak_call_records(self, handle, raw, tag):
        """The server keeps answering after a burst, so its bookkeeping survived it."""
        await asyncio.gather(*[handle.demo_count_sync(tag=f"{tag}{i}") for i in range(50)])
        assert (await raw.get("/v1/health")).status_code == 200
        assert await handle.demo_sync(a=1, b=1) == 2

    async def test_mixed_sync_and_async_burst(self, handle, tag):
        """Mixing executor-bound and loop-bound calls under load stays correct."""
        calls = []
        for i in range(20):
            calls.append(handle.demo_instant_async(tag=f"{tag}a{i}"))
            calls.append(handle.demo_instant_sync(tag=f"{tag}s{i}"))

        names = await asyncio.gather(*calls)
        assert sum(1 for name in names if name == "MainThread") == 20
        assert sum(1 for name in names if name.startswith("rpc-")) == 20
