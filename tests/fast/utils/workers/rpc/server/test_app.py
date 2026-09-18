import asyncio
import contextlib
import threading
import uuid
from collections.abc import AsyncIterator
from typing import NamedTuple

import httpx
import pytest
from fastapi.responses import JSONResponse

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, rpc
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    EXPECTED_BOOT_UUID_HEADER,
    IN_FLIGHT_PATH,
    CallStatusResponse,
)
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


class _Item(StrictBaseModel):
    name: str
    value: int


class _Worker:
    def __init__(self):
        self.calls = 0
        self.done_event = threading.Event()
        self.release_slow = threading.Event()
        self.slow_started = threading.Event()
        self.barrier = threading.Barrier(parties=2, timeout=5.0)
        self.order: list[str] = []

    def demo_sync(self, a: int, b: int) -> int:
        self.calls += 1
        self.done_event.set()
        return a + b

    async def demo_async_model(self, name: str) -> _Item:
        return _Item(name=name, value=len(name))

    def demo_raises(self, message: str) -> None:
        raise RuntimeError(message)

    @rpc(concurrency_group="serial")
    def demo_slow(self, tag: str) -> str:
        self.slow_started.set()
        self.order.append(f"{tag}_start")
        assert self.release_slow.wait(timeout=5.0)
        self.order.append(f"{tag}_end")
        return tag

    @rpc(concurrency_group="serial")
    def demo_fast(self, tag: str) -> str:
        self.order.append(f"{tag}_start")
        return tag

    @rpc(concurrency_group="left")
    def demo_meet_left(self) -> str:
        self.barrier.wait()
        return "left"

    @rpc(concurrency_group="right")
    def demo_meet_right(self) -> str:
        self.barrier.wait()
        return "right"

    def demo_tag(self, tag: str) -> str:
        return tag

    def demo_tag_upper(self, tag: str) -> str:
        return tag.upper()


class _AsyncCancelWorker:
    async def demo_cancel_self(self) -> str:
        raise asyncio.CancelledError


@contextlib.asynccontextmanager
async def _client(worker: object) -> AsyncIterator[httpx.AsyncClient]:
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


class _Submitted(NamedTuple):
    call_id: str
    response: httpx.Response


async def _submit(client: httpx.AsyncClient, method: str, query: dict, call_id: str | None = None) -> _Submitted:
    call_id = call_id if call_id is not None else uuid.uuid4().hex
    response = await client.post(f"/v1/{method}", json={"call_id": call_id, "query": query})
    return _Submitted(call_id=call_id, response=response)


async def _poll_until_done(client: httpx.AsyncClient, call_id: str) -> dict:
    for _ in range(100):
        query_response = await client.get(f"/v1/calls/{call_id}", params={"timeout": 1.0})
        assert query_response.status_code == 200
        body = query_response.json()
        if body["status"] != "pending":
            return body
    raise AssertionError("call never finished")


async def _call(client: httpx.AsyncClient, method: str, query: dict) -> dict:
    submitted = await _submit(client, method, query)
    assert submitted.response.status_code == 200
    return await _poll_until_done(client, submitted.call_id)


class TestRoundtrip:
    async def test_sync_method_success(self):
        """A sync method roundtrips its typed result through submit + query."""
        async with _client(_Worker()) as client:
            body = await _call(client, "demo_sync", {"a": 1, "b": 2})
            assert body == {"status": "success", "result": 3, "error": None}

    async def test_async_method_success(self):
        """An async method runs on the event loop and returns a model result."""
        async with _client(_Worker()) as client:
            body = await _call(client, "demo_async_model", {"name": "abc"})
            assert body == {"status": "success", "result": {"name": "abc", "value": 3}, "error": None}

    async def test_business_exception_becomes_failed_envelope(self):
        """Worker exceptions surface as 200 + failed envelope with a traceback."""
        async with _client(_Worker()) as client:
            body = await _call(client, "demo_raises", {"message": "kaboom"})
            assert body["status"] == "failed"
            assert "RuntimeError" in body["error"] and "kaboom" in body["error"]

    async def test_health_endpoint(self):
        """The health endpoint answers ok."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/health")
            assert response.status_code == 200 and response.json() == {"status": "ok"}


class TestProtocolErrors:
    async def test_unknown_method_404(self):
        """Submitting to an unknown method returns 404."""
        async with _client(_Worker()) as client:
            submitted = await _submit(client, "nope", {})
            assert submitted.response.status_code == 404

    async def test_invalid_query_400(self):
        """A query failing pydantic validation returns 400."""
        async with _client(_Worker()) as client:
            submitted = await _submit(client, "demo_sync", {"a": "x", "b": 2})
            assert submitted.response.status_code == 400

    async def test_unknown_call_id_404(self):
        """Querying an unknown call id returns 404."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/calls/missing", params={"timeout": 0.0})
            assert response.status_code == 404

    async def test_resubmit_different_payload_409(self):
        """Reusing a call id with a different payload returns 409."""
        async with _client(_Worker()) as client:
            first = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert first.response.status_code == 200
            second = await _submit(client, "demo_sync", {"a": 9, "b": 9}, call_id="fixed")
            assert second.response.status_code == 409

    async def test_same_call_id_across_methods_409(self):
        """Reusing a call id on a different method returns 409 rather than the other result."""
        async with _client(_Worker()) as client:
            first = await client.post("/v1/demo_tag", json={"call_id": "fixed", "query": {"tag": "x"}})
            assert first.status_code == 200
            second = await client.post("/v1/demo_tag_upper", json={"call_id": "fixed", "query": {"tag": "x"}})
            assert second.status_code == 409

    async def test_missing_query_field_400(self):
        """An envelope without the query field returns 400, not FastAPI's default 422."""
        async with _client(_Worker()) as client:
            response = await client.post("/v1/demo_sync", json={"call_id": "c1"})
            assert response.status_code == 400

    async def test_non_object_query_400(self):
        """An envelope whose query is not an object returns 400."""
        async with _client(_Worker()) as client:
            response = await client.post("/v1/demo_sync", json={"call_id": "c1", "query": [1, 2]})
            assert response.status_code == 400

    async def test_extra_envelope_field_400(self):
        """An envelope carrying unknown fields returns 400."""
        async with _client(_Worker()) as client:
            response = await client.post("/v1/demo_sync", json={"call_id": "c1", "query": {"a": 1, "b": 2}, "junk": 1})
            assert response.status_code == 400

    async def test_boot_uuid_mismatch_header_returns_412(self) -> None:
        """A stale boot UUID request header is refused before scheduling."""
        worker = _Worker()
        async with _client(worker) as client:
            response = await client.post(
                "/v1/demo_sync",
                headers={EXPECTED_BOOT_UUID_HEADER: "stale"},
                json={"call_id": "c1", "query": {"a": 1, "b": 2}},
            )
            assert response.status_code == 412
            assert response.headers[BOOT_UUID_HEADER] != "stale"
            assert not worker.done_event.is_set()
            assert worker.calls == 0

    async def test_stale_boot_uuid_is_refused_before_routing_and_validation(self) -> None:
        """A stale expectation wins over routing and validation, so a restart is never reported as a 404 or 400."""
        async with _client(_Worker()) as client:
            unknown_route = await client.post(
                "/v1/nope", headers={EXPECTED_BOOT_UUID_HEADER: "stale"}, json={"call_id": "c1", "query": {}}
            )
            malformed = await client.post(
                "/v1/demo_sync", headers={EXPECTED_BOOT_UUID_HEADER: "stale"}, json={"call_id": "c1"}
            )

            assert unknown_route.status_code == 412
            assert malformed.status_code == 412

    @pytest.mark.parametrize(
        ("method", "query", "status_code"),
        [("nope", {}, 404), ("demo_sync", {"a": "x", "b": 2}, 400)],
    )
    async def test_rejected_submission_does_not_reserve_call_id(self, method: str, query: dict, status_code: int):
        """A submission refused before scheduling leaves its call id free for a corrected retry."""
        async with _client(_Worker()) as client:
            rejected = await _submit(client, method, query, call_id="reused")
            assert rejected.response.status_code == status_code

            accepted = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="reused")
            assert accepted.response.status_code == 200
            assert await _poll_until_done(client, "reused") == {"status": "success", "result": 3, "error": None}

    async def test_invalid_poll_timeout_400(self):
        """A negative long-poll timeout is a client error, reported as 400."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/calls/whatever", params={"timeout": -1.0})
            assert response.status_code == 400

    async def test_error_responses_carry_boot_uuid_header(self):
        """4xx responses carry the boot uuid header just like successful ones."""
        async with _client(_Worker()) as client:
            not_found = await client.post("/v1/nope", json={"call_id": "c1", "query": {}})
            malformed = await client.post("/v1/demo_sync", json={"call_id": "c1"})
            assert BOOT_UUID_HEADER in not_found.headers
            assert BOOT_UUID_HEADER in malformed.headers

    async def test_unhandled_route_error_still_carries_boot_uuid_header(self):
        """An exception escaping a route becomes a 500 that still identifies the serving process."""
        app = create_rpc_app(_Worker())

        @app.get("/v1/boom")
        async def boom() -> None:
            raise RuntimeError("boom")

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/v1/boom")

        assert response.status_code == 500
        assert BOOT_UUID_HEADER in response.headers
        assert response.json() == {"detail": "unhandled rpc server error"}

    async def test_a_route_cannot_forge_the_boot_uuid_header(self):
        """The boot uuid header is authoritative, so a header set downstream is overwritten, not kept."""
        app = create_rpc_app(_Worker())

        @app.get("/v1/forged")
        async def forged() -> JSONResponse:
            return JSONResponse(content={}, headers={BOOT_UUID_HEADER: "forged"})

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                health = await client.get("/v1/health")
                response = await client.get("/v1/forged")

        assert response.headers[BOOT_UUID_HEADER] == health.headers[BOOT_UUID_HEADER]


class TestDuplicateCalls:
    async def test_resubmit_same_payload_returns_409(self) -> None:
        """Resubmitting an identical call id fails loudly without rerunning."""
        worker = _Worker()
        async with _client(worker) as client:
            first = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert first.response.json() == {"status": "submitted"}
            assert await asyncio.to_thread(worker.done_event.wait, 5.0)

            second = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert second.response.status_code == 409
            assert "already submitted" in second.response.json()["detail"]
            assert worker.calls == 1


class TestPendingAndCompletion:
    async def test_pending_then_success(self):
        """A long-running call reports pending until it finishes."""
        worker = _Worker()
        async with _client(worker) as client:
            submitted = await _submit(client, "demo_slow", {"tag": "s"})
            pending = await client.get(f"/v1/calls/{submitted.call_id}", params={"timeout": 0.0})
            assert pending.json()["status"] == "pending"

            worker.release_slow.set()
            body = await _poll_until_done(client, submitted.call_id)
            assert body == {"status": "success", "result": "s", "error": None}

    async def test_sync_call_completes_without_polling(self):
        """A submitted sync call runs to completion even if never polled."""
        worker = _Worker()
        async with _client(worker) as client:
            await _submit(client, "demo_sync", {"a": 1, "b": 2})
            assert await asyncio.to_thread(worker.done_event.wait, 5.0)
            assert worker.calls == 1


class TestConcurrencyGroups:
    async def test_same_group_serializes(self):
        """Two calls in one concurrency group run strictly one after another."""
        worker = _Worker()
        async with _client(worker) as client:
            slow_submitted = await _submit(client, "demo_slow", {"tag": "demo_slow"})
            assert await asyncio.to_thread(worker.slow_started.wait, 5.0)
            fast_submitted = await _submit(client, "demo_fast", {"tag": "demo_fast"})

            await asyncio.sleep(0.05)
            assert worker.order == ["demo_slow_start"]

            worker.release_slow.set()
            await _poll_until_done(client, slow_submitted.call_id)
            await _poll_until_done(client, fast_submitted.call_id)
            assert worker.order == ["demo_slow_start", "demo_slow_end", "demo_fast_start"]

    async def test_different_groups_run_in_parallel(self):
        """Calls in different concurrency groups run concurrently."""
        worker = _Worker()
        async with _client(worker) as client:
            left = await _submit(client, "demo_meet_left", {})
            right = await _submit(client, "demo_meet_right", {})
            left_body = await _poll_until_done(client, left.call_id)
            right_body = await _poll_until_done(client, right.call_id)
            assert left_body["result"] == "left" and right_body["result"] == "right"


class TestCancellationOutcome:
    async def test_worker_cancellation_yields_terminal_outcome(self):
        """A worker method raising CancelledError still records a terminal failed outcome."""
        async with _client(_AsyncCancelWorker()) as client:
            body = await _call(client, "demo_cancel_self", {})
            assert body["status"] == "failed"
            assert "CancelledError" in body["error"]

    async def test_cancellation_is_re_raised_after_the_outcome_is_recorded(self):
        """The execution records its terminal outcome and then stays cancelled instead of swallowing it."""
        specs = collect_rpc_method_specs(_AsyncCancelWorker)
        executor = RpcCallExecutor(worker=_AsyncCancelWorker(), specs=specs)
        recorded: list[CallStatusResponse] = []

        with pytest.raises(asyncio.CancelledError):
            await executor._run(
                spec=specs["demo_cancel_self"],
                kwargs={},
                call_id="c1",
                finish=lambda *, outcome: recorded.append(outcome),
            )

        assert [outcome.status for outcome in recorded] == ["failed"]


class TestBootUuid:
    async def test_boot_uuid_header_stable_within_server(self):
        """All responses of one server carry the same boot uuid header."""
        async with _client(_Worker()) as client:
            first = await client.get("/v1/health")
            second = await client.get("/v1/health")
            assert first.headers[BOOT_UUID_HEADER] == second.headers[BOOT_UUID_HEADER]

    async def test_boot_uuid_differs_across_servers(self):
        """Two server instances have different boot_uuids."""
        async with _client(_Worker()) as first_client:
            first = (await first_client.get("/v1/health")).headers[BOOT_UUID_HEADER]
        async with _client(_Worker()) as second_client:
            second = (await second_client.get("/v1/health")).headers[BOOT_UUID_HEADER]
        assert first != second


class TestTheInFlightEndpoint:
    async def test_a_worker_running_nothing_reports_nothing_in_flight(self) -> None:
        """A caller waiting a worker out reads this to decide the worker has really let go."""
        async with _client(_Worker()) as client:
            response = await client.get(IN_FLIGHT_PATH)

            assert response.status_code == 200 and response.json() == {"call_ids": []}

    async def test_it_names_the_call_the_worker_is_running(self) -> None:
        """This is what a caller waits out, so a running call has to show up under its own id."""
        worker = _Worker()
        async with _client(worker) as client:
            submitted = await _submit(client, "demo_slow", {"tag": "a"}, call_id="c1")
            assert submitted.response.status_code == 200
            assert worker.slow_started.wait(timeout=5.0)

            assert (await client.get(IN_FLIGHT_PATH)).json() == {"call_ids": ["c1"]}

            worker.release_slow.set()
            await _poll_until_done(client, "c1")

            assert (await client.get(IN_FLIGHT_PATH)).json() == {"call_ids": []}
