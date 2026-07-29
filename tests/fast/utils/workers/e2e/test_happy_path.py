import asyncio
import time

import httpx

from tests.fast.utils.workers.e2e.harness import READY_TIMEOUT_SECONDS


class TestRoundtrip:
    async def test_sync_method(self, handle):
        """A sync method answers over real HTTP with its declared type."""
        result = await handle.demo_sync(a=3, b=4)
        assert result == 7 and isinstance(result, int)

    async def test_async_method(self, handle):
        """An async method roundtrips a nested payload unchanged."""
        assert await handle.demo_async(value={"k": [1, "x", None]}) == {"k": [1, "x", None]}

    async def test_default_parameter_omitted(self, handle):
        """An omitted defaulted argument uses the worker-side default."""
        assert await handle.demo_default_arg() == "hello world"

    async def test_default_parameter_overridden(self, handle):
        """Passing the defaulted argument overrides it."""
        assert await handle.demo_default_arg(name="miles") == "hello miles"

    async def test_explicit_none_differs_from_omitted(self, handle):
        """An explicit None reaches the worker as None."""
        assert await handle.demo_optional_arg(name=None) == "None"

    async def test_sequential_calls_on_one_handle(self, handle):
        """Consecutive calls on one handle stay independent."""
        assert await handle.demo_sync(a=1, b=1) == 2
        assert await handle.demo_sync(a=2, b=2) == 4

    async def test_two_handles_one_server(self, server, make_handle):
        """Two independent handles can drive the same server."""
        first, second = make_handle(server), make_handle(server)
        await first.wait_ready(timeout=READY_TIMEOUT_SECONDS)
        assert await first.demo_sync(a=1, b=2) == 3
        assert await second.demo_sync(a=3, b=4) == 7

    async def test_call_runs_in_the_server_subprocess(self, handle, server):
        """The worker really executes in the spawned process, not in the test process."""
        assert await handle.report_pid() == server.process.pid


class TestReadiness:
    async def test_wait_ready_returns_promptly(self, server, make_handle):
        """wait_ready returns as soon as the server answers instead of waiting out its timeout."""
        started = time.monotonic()
        await make_handle(server).wait_ready(timeout=30.0)
        assert time.monotonic() - started < 5.0

    async def test_health_endpoint(self, raw):
        """The health endpoint answers ok."""
        response = await raw.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestConcurrentCalls:
    async def test_many_concurrent_calls_from_one_handle(self, handle):
        """Fifty concurrent calls over one connection pool all return their own result."""
        results = await asyncio.gather(*[handle.demo_async(value={"i": i}) for i in range(50)])
        assert results == [{"i": i} for i in range(50)]

    async def test_concurrent_calls_from_several_handles(self, server, make_handle):
        """Concurrent calls from several handles do not cross results."""
        handles = [make_handle(server) for _ in range(5)]
        await handles[0].wait_ready(timeout=READY_TIMEOUT_SECONDS)
        results = await asyncio.gather(*[h.demo_sync(a=i, b=i) for i, h in enumerate(handles)])
        assert results == [0, 2, 4, 6, 8]


class TestTypedPayloads:
    async def test_scalar_types_keep_their_python_type(self, handle):
        """Scalars keep their type instead of collapsing to strings or ints."""
        assert await handle.demo_async(value={"f": 1.5, "b": True, "s": "1", "n": None}) == {
            "f": 1.5,
            "b": True,
            "s": "1",
            "n": None,
        }

    async def test_unicode_payload(self, handle):
        """Non-ascii text survives the real socket unchanged."""
        text = "中文 🚀 \\ \" '"
        assert await handle.demo_async(value={"t": text}) == {"t": text}


class TestManualProtocol:
    async def test_submit_then_poll_by_hand(self, raw):
        """The documented submit + poll pair works without the client."""
        submit = await raw.post("/v1/demo_sync", json={"call_id": "manual-1", "query": {"a": 2, "b": 5}})
        assert submit.status_code == 200
        assert submit.json() == {"status": "submitted"}

        for _ in range(50):
            poll = await raw.get("/v1/calls/manual-1", params={"timeout": 1.0})
            assert poll.status_code == 200
            if poll.json()["status"] != "pending":
                break
        assert poll.json() == {"status": "success", "result": 7, "error": None}

    async def test_finished_call_can_be_polled_repeatedly(self, raw):
        """A finished outcome stays retrievable for later polls."""
        await raw.post("/v1/demo_sync", json={"call_id": "manual-2", "query": {"a": 1, "b": 1}})
        for _ in range(50):
            body = (await raw.get("/v1/calls/manual-2", params={"timeout": 5.0})).json()
            if body["status"] != "pending":
                break
        assert body == {"status": "success", "result": 2, "error": None}

    async def test_second_client_sees_the_outcome(self, raw, server):
        """Call state belongs to the server, not to the connection that submitted it."""
        await raw.post("/v1/demo_sync", json={"call_id": "manual-3", "query": {"a": 4, "b": 4}})
        async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as other:
            for _ in range(50):
                body = (await other.get("/v1/calls/manual-3", params={"timeout": 5.0})).json()
                if body["status"] != "pending":
                    break
        assert body["result"] == 8
