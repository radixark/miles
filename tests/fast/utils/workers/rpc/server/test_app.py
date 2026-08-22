import asyncio
import contextlib
import dataclasses
import hashlib
import json
import sys
import threading
import tracemalloc
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import NamedTuple

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, rpc
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    EXPECTED_BOOT_UUID_HEADER,
    IN_FLIGHT_PATH,
    CallStatusResponse,
    SubmitRequest,
)
from miles.utils.workers.rpc.server import store as store_module
from miles.utils.workers.rpc.server.app import _RequestBodyLimitMiddleware, create_rpc_app
from miles.utils.workers.rpc.server.core import RpcServer
from miles.utils.workers.rpc.server.executor import RpcCallExecutor
from miles.utils.workers.rpc.server.store import CallStore


class _Item(StrictBaseModel):
    name: str
    value: int


class _Worker:
    def __init__(self):
        self.calls = 0
        self.done_event = threading.Event()
        self.release_slow = threading.Event()
        self.slow_finished = threading.Event()
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
        self.slow_finished.set()
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
        self.order.append(tag)
        return tag

    def demo_tag_upper(self, tag: str) -> str:
        return tag.upper()

    @rpc(max_serialized_outcome_bytes=1024)
    def demo_declared_result_budget(self, value: str) -> str:
        self.calls += 1
        return value

    @rpc(concurrency_group="heartbeat_status")
    def get_heartbeat_status(self) -> str:
        return "alive"


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


def _request_digest(*, method: str, query: dict) -> str:
    canonical = json.dumps(
        {"method": method, "query": query}, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


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
    async def test_resubmit_same_payload_after_completion_is_409_and_never_reruns(self) -> None:
        """A completed call still owns its id: the resubmit is refused loudly and the work is not rerun."""
        worker = _Worker()
        async with _client(worker) as client:
            first = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert first.response.json() == {"status": "submitted"}
            assert await asyncio.to_thread(worker.done_event.wait, 5.0)

            second = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert second.response.status_code == 409
            assert worker.calls == 1

    async def test_resubmit_same_payload_while_pending_is_409_and_never_queues_twice(self) -> None:
        """An in-flight call owns its id: the resubmit is refused loudly and the work runs exactly once."""
        worker = _Worker()
        async with _client(worker) as client:
            first = await _submit(client, "demo_slow", {"tag": "slow"}, call_id="fixed")
            assert first.response.json() == {"status": "submitted"}
            assert await asyncio.to_thread(worker.slow_started.wait, 5.0)

            second = await _submit(client, "demo_slow", {"tag": "slow"}, call_id="fixed")
            assert second.response.status_code == 409
            worker.release_slow.set()

            assert (await _poll_until_done(client, "fixed"))["status"] == "success"
            assert worker.order == ["slow_start", "slow_end"]

    async def test_retrieved_outcome_remains_idempotent_past_the_old_short_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrieved outcome stays pollable past the old short ttl, and its id stays owned."""
        from miles.utils.workers.rpc.server import store as store_module

        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        worker = _Worker()
        async with _client(worker) as client:
            first = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert first.response.status_code == 200
            assert (await _poll_until_done(client, "fixed"))["status"] == "success"

            now[0] += 301.0
            assert (await _poll_until_done(client, "fixed"))["status"] == "success"

            second = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert second.response.status_code == 409
            assert worker.calls == 1


class TestAcknowledgement:
    async def test_acknowledgement_is_idempotent_and_prevents_reexecution(self) -> None:
        """Repeated ACK and a late duplicate keep one execution and one compact tombstone."""
        worker = _Worker()
        query = {"a": 1, "b": 2}
        async with _client(worker) as client:
            submitted = await _submit(client, "demo_sync", query, call_id="fixed")
            assert (await _poll_until_done(client, "fixed"))["status"] == "success"
            ack_body = {"request_digest": _request_digest(method="demo_sync", query=query)}

            first_ack = await client.post(f"/v1/calls/{submitted.call_id}/ack", json=ack_body)
            second_ack = await client.post(f"/v1/calls/{submitted.call_id}/ack", json=ack_body)
            poll_after_ack = await client.get(f"/v1/calls/{submitted.call_id}", params={"timeout": 0.0})
            duplicate = await _submit(client, "demo_sync", query, call_id="fixed")
            conflicting = await _submit(client, "demo_sync", {"a": 2, "b": 2}, call_id="fixed")

        assert first_ack.status_code == second_ack.status_code == 200
        assert first_ack.json() == second_ack.json() == {"status": "acknowledged"}
        assert poll_after_ack.status_code == 410
        assert duplicate.response.status_code == conflicting.response.status_code == 409
        assert "outcome was already acknowledged" in duplicate.response.json()["detail"]
        assert "already belongs to another request" in conflicting.response.json()["detail"]
        assert worker.calls == 1

    async def test_pending_call_acknowledgement_is_rejected(self) -> None:
        """ACK cannot discard the state of a call that has not reached a terminal outcome."""
        worker = _Worker()
        query = {"tag": "slow"}
        async with _client(worker) as client:
            await _submit(client, "demo_slow", query, call_id="fixed")
            assert await asyncio.to_thread(worker.slow_started.wait, 5.0)

            response = await client.post(
                "/v1/calls/fixed/ack",
                json={"request_digest": _request_digest(method="demo_slow", query=query)},
            )
            worker.release_slow.set()
            await _poll_until_done(client, "fixed")

        assert response.status_code == 409

    async def test_wrong_digest_does_not_discard_the_pollable_outcome(self) -> None:
        """A mismatched ACK is a conflict and leaves the terminal outcome available for its real caller."""
        worker = _Worker()
        query = {"a": 1, "b": 2}
        async with _client(worker) as client:
            await _submit(client, "demo_sync", query, call_id="fixed")
            expected = await _poll_until_done(client, "fixed")
            wrong_ack = await client.post(
                "/v1/calls/fixed/ack",
                json={"request_digest": _request_digest(method="demo_sync", query={"a": 9, "b": 9})},
            )
            still_pollable = await client.get("/v1/calls/fixed", params={"timeout": 0.0})

        assert wrong_ack.status_code == 409
        assert still_pollable.status_code == 200
        assert still_pollable.json() == expected


class TestCapacity:
    def test_oversized_call_id_is_rejected_before_executor_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A call id above the UTF-8 byte limit is a client error before executor ownership."""
        server = RpcServer(worker=_Worker())
        started: list[str] = []
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: started.append(kwargs["call_id"]))
        oversized = "€" * (store_module.MAX_CALL_ID_BYTES // len("€".encode()) + 1)

        with pytest.raises(HTTPException) as exc_info:
            server.submit_call(
                method_name="demo_sync",
                request=SubmitRequest(call_id=oversized, query={"a": 1, "b": 2}),
            )

        assert getattr(exc_info.value, "status_code", None) == 400
        assert started == []

    async def test_non_utf8_call_id_is_rejected_before_executor_start(self) -> None:
        """A JSON lone surrogate becomes a client error instead of an unhandled encoding failure."""
        worker = _Worker()
        async with _client(worker) as client:
            response = await client.post(
                "/v1/demo_sync",
                content=b'{"call_id":"\\ud800","query":{"a":1,"b":2}}',
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 400
        assert worker.calls == 0

    def test_capacity_rejection_happens_before_executor_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A saturated store returns retryable 503 without starting the newly submitted method."""
        server = RpcServer(worker=_Worker())
        server._store = CallStore(max_active_calls=1)
        server._store.begin(call_id="occupied", fingerprint=b"x" * hashlib.sha256().digest_size)
        started: list[str] = []
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: started.append(kwargs["call_id"]))

        request = SubmitRequest(call_id="rejected", query={"a": 1, "b": 2})
        with pytest.raises(HTTPException) as exc_info:
            server.submit_call(method_name="demo_sync", request=request)

        assert getattr(exc_info.value, "status_code", None) == 503
        assert started == []

    def test_declared_result_budget_flows_from_rpc_metadata_into_admission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RpcServer reserves the method-declared outcome maximum before starting its executor."""
        server = RpcServer(worker=_Worker())
        reservations: list[tuple[int, int]] = []
        original_begin = server._store.begin

        def record_reservation(**kwargs: object) -> bool:
            reservations.append(
                (
                    int(kwargs["request_reservation_bytes"]),
                    int(kwargs["outcome_reservation_bytes"]),
                )
            )
            return original_begin(**kwargs)

        monkeypatch.setattr(server._store, "begin", record_reservation)
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: None)

        server.submit_call(
            method_name="demo_declared_result_budget",
            request=SubmitRequest(call_id="c1", query={"value": "small"}),
        )

        canonical = json.dumps(
            {"method": "demo_declared_result_budget", "query": {"value": "small"}},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        assert reservations == [(len(canonical), 1024)]

    @pytest.mark.parametrize(
        ("method_name", "query", "control_plane"),
        [
            ("demo_sync", {"a": 1, "b": 2}, False),
            ("get_heartbeat_status", {}, True),
        ],
    )
    def test_executor_start_failure_rolls_back_before_returning_500(
        self,
        method_name: str,
        query: dict,
        control_plane: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A data or control executor-start failure releases its id and every admission reservation."""
        server = RpcServer(worker=_Worker())
        server._specs[method_name] = dataclasses.replace(server._specs[method_name], control_plane=control_plane)

        def fail_start(**kwargs: object) -> None:
            raise RuntimeError("injected start failure")

        monkeypatch.setattr(server._executor, "start", fail_start)
        request = SubmitRequest(call_id="fixed", query=query)

        with pytest.raises(RuntimeError, match="injected start failure"):
            server.submit_call(method_name=method_name, request=request)

        assert not server._store.contains("fixed")
        assert server._store.stats.active_calls == 0
        assert server._store.stats.control_calls == 0
        assert server._store.stats.queued_request_bytes == 0
        assert server._store.stats.control_queued_request_bytes == 0
        assert server._store.stats.reserved_outcome_bytes == 0
        assert server._store.stats.control_reserved_outcome_bytes == 0

    def test_outcome_capacity_rejection_does_not_start_the_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A method whose declared outcome reservation cannot fit is rejected before executor ownership."""
        server = RpcServer(worker=_Worker())
        server._store = CallStore(max_unacknowledged_outcome_bytes=512)
        started: list[str] = []
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: started.append(kwargs["call_id"]))

        with pytest.raises(HTTPException) as exc_info:
            server.submit_call(
                method_name="demo_declared_result_budget",
                request=SubmitRequest(call_id="rejected", query={"value": "small"}),
            )

        assert getattr(exc_info.value, "status_code", None) == 503
        assert started == []

    def test_tombstone_capacity_rejection_does_not_start_the_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A full deduplication horizon rejects a new id before executor ownership."""
        server = RpcServer(worker=_Worker())
        server._store = CallStore(max_tombstones=1)
        digest = bytes.fromhex(_request_digest(method="demo_sync", query={"a": 1, "b": 2}))
        server._store.begin(call_id="existing", fingerprint=digest)
        server._store.finish(call_id="existing", outcome=CallStatusResponse(status="success", result=3))
        server._store.acknowledge(call_id="existing", fingerprint=digest)
        started: list[str] = []
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: started.append(kwargs["call_id"]))

        with pytest.raises(HTTPException) as exc_info:
            server.submit_call(
                method_name="demo_sync",
                request=SubmitRequest(call_id="new", query={"a": 1, "b": 2}),
            )

        assert getattr(exc_info.value, "status_code", None) == 503
        assert started == []

    def test_control_rpc_can_start_when_data_admission_is_saturated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A heartbeat RPC uses its bounded control reserve instead of misclassifying a healthy full worker."""
        server = RpcServer(worker=_Worker())
        server._specs["get_heartbeat_status"] = dataclasses.replace(
            server._specs["get_heartbeat_status"], control_plane=True
        )
        server._store = CallStore(max_active_calls=1, max_control_calls=1)
        started: list[str] = []
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: started.append(kwargs["call_id"]))

        server.submit_call(
            method_name="demo_sync",
            request=SubmitRequest(call_id="data", query={"a": 1, "b": 2}),
        )
        server.submit_call(
            method_name="get_heartbeat_status",
            request=SubmitRequest(call_id="heartbeat", query={}),
        )

        assert started == ["data", "heartbeat"]

    @pytest.mark.parametrize("saturation", ["request", "outcome"])
    def test_control_rpc_can_start_when_data_byte_budget_is_saturated(
        self, saturation: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The heartbeat reserve is independent from both queued-request and retained-outcome data budgets."""
        store_kwargs = (
            {"max_queued_request_bytes": 1} if saturation == "request" else {"max_unacknowledged_outcome_bytes": 512}
        )
        server = RpcServer(worker=_Worker())
        server._specs["get_heartbeat_status"] = dataclasses.replace(
            server._specs["get_heartbeat_status"], control_plane=True
        )
        server._store = CallStore(**store_kwargs)
        server._store.begin(
            call_id="data",
            fingerprint=hashlib.sha256(b"data").digest(),
            request_reservation_bytes=1 if saturation == "request" else 0,
            outcome_reservation_bytes=512,
        )
        started: list[str] = []
        monkeypatch.setattr(server._executor, "start", lambda **kwargs: started.append(kwargs["call_id"]))

        server.submit_call(
            method_name="get_heartbeat_status",
            request=SubmitRequest(call_id="heartbeat", query={}),
        )

        assert started == ["heartbeat"]

    async def test_a_multi_megabyte_request_reaches_its_worker(self) -> None:
        """No per-request wire-size cap is imposed by default, so a large declared payload still executes."""
        worker = _Worker()
        async with _client(worker) as client:
            response = await _submit(
                client,
                "demo_tag",
                {"tag": "x" * (4 * 1024 * 1024)},
                call_id="large",
            )

        assert response.response.json() == {"status": "submitted"}

    async def test_disconnected_chunked_requests_terminate_without_task_leaks(self) -> None:
        """A client disconnect during body streaming terminates each middleware call without reading forever."""
        downstream_calls = 0

        async def downstream(scope: dict, receive: object, send: object) -> None:
            nonlocal downstream_calls
            downstream_calls += 1

        middleware = _RequestBodyLimitMiddleware(downstream, boot_uuid="boot")
        scope = {"type": "http", "headers": []}

        def make_disconnecting_receive() -> Callable[[], Awaitable[dict]]:
            messages = [
                {"type": "http.request", "body": b"partial", "more_body": True},
                {"type": "http.disconnect"},
            ]

            async def receive() -> dict:
                if messages:
                    return messages.pop(0)
                return {"type": "http.disconnect"}

            return receive

        async def ignored_send(message: dict) -> None:
            raise AssertionError(f"disconnect must not emit a response: {message}")

        await asyncio.wait_for(
            asyncio.gather(*(middleware(scope, make_disconnecting_receive(), ignored_send) for _ in range(100))),
            timeout=0.5,
        )

        assert downstream_calls == 0

    async def test_oversized_chunk_is_rejected_before_copying_it(self) -> None:
        """One huge ASGI chunk is rejected before a second body-sized allocation is made."""
        oversized = b"x" * (8 * 1024 * 1024)
        messages = [{"type": "http.request", "body": oversized, "more_body": False}]
        responses: list[dict] = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(message: dict) -> None:
            responses.append(message)

        async def downstream(scope: dict, receive: object, send: object) -> None:
            raise AssertionError("oversized body must not reach the downstream app")

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=1024,
        )
        tracemalloc.start()
        await middleware({"type": "http", "headers": []}, receive, send)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert responses[0]["status"] == 503
        assert peak < 1024 * 1024

    async def test_concurrent_request_bodies_share_bounded_data_and_control_reserves(self) -> None:
        """Concurrent data bodies share one cap while heartbeat ingress retains a separate bounded reserve."""
        first_waiting = asyncio.Event()
        release_first = asyncio.Event()
        downstream_paths: list[str] = []
        responses: dict[str, list[dict]] = {"first": [], "second": [], "control": []}

        async def downstream(scope: dict, receive: Callable[[], Awaitable[dict]], send: object) -> None:
            downstream_paths.append(scope["path"])
            await receive()

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=8,
            max_control_aggregate_bytes=6,
            control_paths=frozenset({"/v1/get_heartbeat_status"}),
        )

        first_messages = [{"type": "http.request", "body": b"123456", "more_body": True}]

        async def first_receive() -> dict:
            if first_messages:
                return first_messages.pop(0)
            first_waiting.set()
            await release_first.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def one_chunk(body: bytes) -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        async def send_to(name: str, message: dict) -> None:
            responses[name].append(message)

        first = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
                first_receive,
                lambda message: send_to("first", message),
            )
        )
        await asyncio.wait_for(first_waiting.wait(), timeout=1.0)
        await middleware(
            {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
            lambda: one_chunk(b"789"),
            lambda message: send_to("second", message),
        )
        await middleware(
            {"type": "http", "headers": [], "path": "/v1/get_heartbeat_status", "method": "POST"},
            lambda: one_chunk(b"abc"),
            lambda message: send_to("control", message),
        )
        release_first.set()
        await first

        assert responses["second"][0]["status"] == 503
        assert "/v1/get_heartbeat_status" in downstream_paths

    @pytest.mark.parametrize("chunk", [b"", b"x"])
    async def test_concurrent_slow_streams_share_bounded_data_and_control_request_slots(self, chunk: bytes) -> None:
        """Empty and tiny slow streams cannot create unbounded data tasks or consume the control request lane."""
        streams_started = [asyncio.Event(), asyncio.Event()]
        release_streams = asyncio.Event()
        data_responses: list[dict] = []
        control_paths: list[str] = []

        def slow_receive(index: int) -> Callable[[], Awaitable[dict]]:
            sent = False

            async def receive() -> dict:
                nonlocal sent
                if not sent:
                    sent = True
                    streams_started[index].set()
                    return {"type": "http.request", "body": chunk, "more_body": True}
                await release_streams.wait()
                return {"type": "http.request", "body": b"", "more_body": False}

            return receive

        async def downstream(scope: dict, receive: Callable[[], Awaitable[dict]], send: object) -> None:
            control_paths.append(scope["path"])
            await receive()

        async def forbidden_receive() -> dict:
            raise AssertionError("a request rejected by the aggregate slot cap must not read its body")

        async def send_data(message: dict) -> None:
            data_responses.append(message)

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_in_flight_requests=2,
            max_control_in_flight_requests=1,
            control_paths=frozenset({"/v1/get_heartbeat_status"}),
        )
        streams = [
            asyncio.create_task(
                middleware(
                    {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
                    slow_receive(index),
                    send_data,
                )
            )
            for index in range(2)
        ]
        await asyncio.gather(*(event.wait() for event in streams_started))

        await middleware(
            {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
            forbidden_receive,
            send_data,
        )

        async def one_chunk() -> dict:
            return {"type": "http.request", "body": b"c", "more_body": False}

        await middleware(
            {"type": "http", "headers": [], "path": "/v1/get_heartbeat_status", "method": "POST"},
            one_chunk,
            send_data,
        )
        release_streams.set()
        await asyncio.gather(*streams)

        assert data_responses[0]["status"] == 503
        assert "/v1/get_heartbeat_status" in control_paths
        assert middleware._data_in_flight_requests == 0
        assert middleware._control_in_flight_requests == 0

    async def test_rejected_body_releases_unaccounted_source_before_backpressured_send(self) -> None:
        """A rejected chunk is neither retained nor left outside accounting while its response is backpressured."""
        first_waiting = asyncio.Event()
        release_first = asyncio.Event()
        reject_sending = asyncio.Event()
        release_reject = asyncio.Event()
        source = b"x" * 1024
        source_reference_count = sys.getrefcount(source)
        source_sent = False

        async def downstream(scope: dict, receive: Callable[[], Awaitable[dict]], send: object) -> None:
            await receive()

        async def first_receive() -> dict:
            if not first_waiting.is_set():
                first_waiting.set()
                return {"type": "http.request", "body": b"a", "more_body": True}
            await release_first.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def rejected_receive() -> dict:
            nonlocal source_sent
            assert not source_sent
            source_sent = True
            return {"type": "http.request", "body": source, "more_body": False}

        async def ignored_send(message: dict) -> None:
            pass

        async def blocked_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                assert sys.getrefcount(source) == source_reference_count
                assert middleware._data_aggregate_bytes == 1
                reject_sending.set()
                await release_reject.wait()

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=1,
        )
        first = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
                first_receive,
                ignored_send,
            )
        )
        await asyncio.wait_for(first_waiting.wait(), timeout=1.0)
        rejected = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
                rejected_receive,
                blocked_send,
            )
        )
        await asyncio.wait_for(reject_sending.wait(), timeout=1.0)

        release_reject.set()
        await rejected
        release_first.set()
        await first
        assert middleware._data_aggregate_bytes == 0

    @pytest.mark.parametrize("path", ["/v1/demo", "/v1/get_heartbeat_status"])
    @pytest.mark.parametrize("stage", ["partial-body", "downstream", "rejected-response"])
    async def test_cancellation_releases_every_ingress_reservation(self, path: str, stage: str) -> None:
        """Cancellation during body, downstream, or rejection releases both resource lanes completely."""
        entered = asyncio.Event()
        never = asyncio.Event()
        receive_calls = 0

        async def receive() -> dict:
            nonlocal receive_calls
            receive_calls += 1
            if stage == "partial-body":
                if receive_calls == 1:
                    return {"type": "http.request", "body": b"x", "more_body": True}
                entered.set()
                await never.wait()
            return {
                "type": "http.request",
                "body": b"xx" if stage == "rejected-response" else b"x",
                "more_body": False,
            }

        async def downstream(scope: dict, replay: Callable[[], Awaitable[dict]], send: object) -> None:
            await replay()
            entered.set()
            await never.wait()

        async def send(message: dict) -> None:
            if stage == "rejected-response" and message["type"] == "http.response.start":
                entered.set()
                await never.wait()

        aggregate_bytes = 1 if stage == "rejected-response" else 2
        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=aggregate_bytes,
            max_control_aggregate_bytes=aggregate_bytes,
            control_paths=frozenset({"/v1/get_heartbeat_status"}),
        )
        task = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": path, "method": "POST"},
                receive,
                send,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert middleware._data_aggregate_bytes == 0
        assert middleware._control_aggregate_bytes == 0
        assert middleware._data_in_flight_requests == 0
        assert middleware._control_in_flight_requests == 0

    @pytest.mark.parametrize("path", ["/v1/demo", "/v1/get_heartbeat_status"])
    @pytest.mark.parametrize("stage", ["receive", "downstream"])
    async def test_escaping_exceptions_release_every_ingress_reservation(self, path: str, stage: str) -> None:
        """Receive and downstream failures release body bytes and request ownership in both lanes."""
        receive_calls = 0

        async def receive() -> dict:
            nonlocal receive_calls
            receive_calls += 1
            if stage == "receive" and receive_calls == 2:
                raise RuntimeError("receive failed")
            return {"type": "http.request", "body": b"x", "more_body": stage == "receive"}

        async def downstream(scope: dict, replay: Callable[[], Awaitable[dict]], send: object) -> None:
            await replay()
            raise RuntimeError("downstream failed")

        async def send(message: dict) -> None:
            pass

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=2,
            max_control_aggregate_bytes=2,
            control_paths=frozenset({"/v1/get_heartbeat_status"}),
        )
        with pytest.raises(RuntimeError, match=f"{stage} failed"):
            await middleware(
                {"type": "http", "headers": [], "path": path, "method": "POST"},
                receive,
                send,
            )

        assert middleware._data_aggregate_bytes == 0
        assert middleware._control_aggregate_bytes == 0
        assert middleware._data_in_flight_requests == 0
        assert middleware._control_in_flight_requests == 0

    async def test_data_request_slot_saturation_preserves_every_get_control_route(self) -> None:
        """Health, in-flight inspection, and call polling retain their control lane while data streams are full."""
        data_waiting = asyncio.Event()
        release_data = asyncio.Event()
        downstream_paths: list[str] = []
        data_receive_calls = 0

        async def data_receive() -> dict:
            nonlocal data_receive_calls
            data_receive_calls += 1
            if data_receive_calls == 1:
                data_waiting.set()
                return {"type": "http.request", "body": b"", "more_body": True}
            await release_data.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def one_message() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def downstream(scope: dict, receive: Callable[[], Awaitable[dict]], send: object) -> None:
            downstream_paths.append(scope["path"])
            await receive()

        async def send(message: dict) -> None:
            pass

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_in_flight_requests=1,
            max_control_in_flight_requests=1,
        )
        data = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
                data_receive,
                send,
            )
        )
        await asyncio.wait_for(data_waiting.wait(), timeout=1.0)
        for path in ("/v1/health", "/v1/in-flight", "/v1/calls/c1"):
            await middleware(
                {"type": "http", "headers": [], "path": path, "method": "GET"},
                one_message,
                send,
            )
        release_data.set()
        await data

        assert downstream_paths == ["/v1/health", "/v1/in-flight", "/v1/calls/c1", "/v1/demo"]

    @pytest.mark.parametrize("path", ["/v1/demo", "/v1/get_heartbeat_status"])
    async def test_backpressured_capacity_responses_use_a_bounded_rejection_lane(self, path: str) -> None:
        """Each lane bounds slow overload responses and drops later requests without reading their bodies."""
        rejection_started = asyncio.Event()
        release_rejection = asyncio.Event()

        async def downstream(scope: dict, receive: object, send: object) -> None:
            raise AssertionError("an overflow request must not reach the downstream app")

        async def forbidden_receive() -> dict:
            raise AssertionError("an overflow request must not read its body")

        async def blocked_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                rejection_started.set()
                await release_rejection.wait()

        async def forbidden_send(message: dict) -> None:
            raise AssertionError("a saturated rejection lane must fail closed without awaiting a response")

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_in_flight_requests=0,
            max_control_in_flight_requests=0,
            max_data_in_flight_rejections=1,
            max_control_in_flight_rejections=1,
            control_paths=frozenset({"/v1/get_heartbeat_status"}),
        )
        first = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": path, "method": "POST"},
                forbidden_receive,
                blocked_send,
            )
        )
        await asyncio.wait_for(rejection_started.wait(), timeout=1.0)
        await asyncio.wait_for(
            middleware(
                {"type": "http", "headers": [], "path": path, "method": "POST"},
                forbidden_receive,
                forbidden_send,
            ),
            timeout=0.1,
        )

        expected_data = int(path == "/v1/demo")
        assert middleware._data_in_flight_rejections == expected_data
        assert middleware._control_in_flight_rejections == 1 - expected_data
        assert middleware._data_aggregate_bytes == 0
        assert middleware._control_aggregate_bytes == 0
        assert middleware._data_in_flight_requests == 0
        assert middleware._control_in_flight_requests == 0
        release_rejection.set()
        await first
        assert middleware._data_in_flight_rejections == 0
        assert middleware._control_in_flight_rejections == 0

    async def test_ingress_reservation_accumulates_across_chunks(self) -> None:
        """Chunks that each fit the ingress budget still exhaust it once their running total crosses it."""
        messages = [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": True},
        ]
        responses: list[dict] = []
        body_reads = 0

        async def receive() -> dict:
            nonlocal body_reads
            if body_reads >= len(messages):
                return {"type": "http.disconnect"}
            message = messages[body_reads]
            body_reads += 1
            return message

        async def send(message: dict) -> None:
            responses.append(message)

        async def downstream(scope: dict, receive: object, send: object) -> None:
            raise AssertionError("a body over the ingress budget must not reach the downstream app")

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=5,
        )
        await middleware(
            {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
            receive,
            send,
        )

        assert responses[0]["status"] == 503
        assert body_reads == 2
        assert middleware._data_aggregate_bytes == 0

    async def test_multichunk_assembly_reserves_both_source_and_joined_body_bytes(self) -> None:
        """Joining multiple chunks cannot transiently exceed the shared ingress reservation."""
        messages = [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
        responses: list[dict] = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(message: dict) -> None:
            responses.append(message)

        async def downstream(scope: dict, receive: object, send: object) -> None:
            raise AssertionError("an unreserved joined-body copy must not reach the downstream app")

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=11,
        )
        await middleware(
            {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
            receive,
            send,
        )

        assert responses[0]["status"] == 503
        assert middleware._data_aggregate_bytes == 0

    async def test_only_canonical_post_control_routes_use_the_control_ingress_reserve(self) -> None:
        """Wrong methods and suffix lookalikes cannot consume the independent control ingress budget."""
        first_waiting = asyncio.Event()
        release_first = asyncio.Event()
        downstream: list[tuple[str, str]] = []
        responses: dict[str, list[dict]] = {"get-ack": [], "fake-ack": [], "ack": [], "heartbeat": []}

        async def app(scope: dict, receive: Callable[[], Awaitable[dict]], send: object) -> None:
            downstream.append((scope["method"], scope["path"]))
            await receive()

        middleware = _RequestBodyLimitMiddleware(
            app,
            boot_uuid="boot",
            max_data_aggregate_bytes=2,
            max_control_aggregate_bytes=2,
            control_paths=frozenset({"/v1/get_heartbeat_status"}),
        )
        first_messages = [{"type": "http.request", "body": b"x", "more_body": True}]

        async def first_receive() -> dict:
            if first_messages:
                return first_messages.pop(0)
            first_waiting.set()
            await release_first.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def one_byte() -> dict:
            return {"type": "http.request", "body": b"y", "more_body": False}

        async def send_to(name: str, message: dict) -> None:
            responses[name].append(message)

        first = asyncio.create_task(
            middleware(
                {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
                first_receive,
                lambda message: send_to("first", message),
            )
        )
        await asyncio.wait_for(first_waiting.wait(), timeout=1.0)
        for name, method, path in (
            ("get-ack", "GET", "/v1/calls/c1/ack"),
            ("fake-ack", "POST", "/not-rpc/ack"),
            ("ack", "POST", "/v1/calls/c1/ack"),
            ("heartbeat", "POST", "/v1/get_heartbeat_status"),
        ):
            await middleware(
                {"type": "http", "headers": [], "path": path, "method": method},
                one_byte,
                lambda message, name=name: send_to(name, message),
            )
        release_first.set()
        await first

        assert responses["get-ack"][0]["status"] == 503
        assert responses["fake-ack"][0]["status"] == 503
        assert ("POST", "/v1/calls/c1/ack") in downstream
        assert ("POST", "/v1/get_heartbeat_status") in downstream

    async def test_joined_body_does_not_retain_source_chunks_across_downstream_await(self) -> None:
        """Accepted multi-chunk bodies release source references before awaiting the downstream app."""
        source_chunks = [b"a" * 1024, b"b" * 1024]
        source_reference_count = sys.getrefcount(source_chunks[1])
        messages = [
            {"type": "http.request", "body": source_chunks[0], "more_body": True},
            {"type": "http.request", "body": source_chunks[1], "more_body": False},
        ]
        retained_references: list[int] = []

        async def receive() -> dict:
            return messages.pop(0)

        async def downstream(scope: dict, receive: Callable[[], Awaitable[dict]], send: object) -> None:
            retained_references.append(sys.getrefcount(source_chunks[1]))
            await receive()

        async def send(message: dict) -> None:
            pass

        middleware = _RequestBodyLimitMiddleware(
            downstream,
            boot_uuid="boot",
            max_data_aggregate_bytes=8192,
        )
        await middleware(
            {"type": "http", "headers": [], "path": "/v1/demo", "method": "POST"},
            receive,
            send,
        )

        assert retained_references == [source_reference_count]

    async def test_http_control_paths_and_heartbeat_converge_while_data_capacity_is_full(self) -> None:
        """Health, poll, in-flight, duplicate, ACK, and heartbeat stay usable at the data admission cap."""
        worker = _Worker()
        app = create_rpc_app(worker)
        app.state.rpc_server._specs["get_heartbeat_status"] = dataclasses.replace(
            app.state.rpc_server._specs["get_heartbeat_status"], control_plane=True
        )
        app.state.rpc_server._store = CallStore(max_active_calls=1, max_control_calls=1)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                accepted = await _submit(client, "demo_slow", {"tag": "slow"}, call_id="data")
                assert accepted.response.status_code == 200
                assert await asyncio.to_thread(worker.slow_started.wait, 5.0)

                health = await client.get("/v1/health")
                in_flight = await client.get(IN_FLIGHT_PATH)
                poll = await client.get("/v1/calls/data", params={"timeout": 0.0})
                duplicate = await _submit(client, "demo_slow", {"tag": "slow"}, call_id="data")
                rejected = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="other")
                heartbeat = await _submit(client, "get_heartbeat_status", {}, call_id="heartbeat")

                worker.release_slow.set()
                await _poll_until_done(client, "data")

        assert health.status_code == 200
        assert in_flight.json() == {"call_ids": ["data"]}
        assert poll.json()["status"] == "pending"
        assert duplicate.response.status_code == 409
        assert rejected.response.status_code == 503
        assert heartbeat.response.status_code == 200

    async def test_tombstone_saturation_preserves_ack_poll_duplicate_and_health(self) -> None:
        """A full tombstone budget rejects only new ids while existing call control paths remain usable."""
        worker = _Worker()
        app = create_rpc_app(worker)
        app.state.rpc_server._store = CallStore(max_active_calls=1, max_tombstones=1)
        query = {"a": 1, "b": 2}
        digest = _request_digest(method="demo_sync", query=query)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                await _submit(client, "demo_sync", query, call_id="fixed")
                await _poll_until_done(client, "fixed")
                first_ack = await client.post("/v1/calls/fixed/ack", json={"request_digest": digest})
                second_ack = await client.post("/v1/calls/fixed/ack", json={"request_digest": digest})
                poll = await client.get("/v1/calls/fixed", params={"timeout": 0.0})
                duplicate = await _submit(client, "demo_sync", query, call_id="fixed")
                poll_after_duplicate = await client.get("/v1/calls/fixed", params={"timeout": 0.0})
                rejected = await _submit(client, "demo_sync", query, call_id="new")
                health = await client.get("/v1/health")

                assert app.state.rpc_server._store.stats.tombstones == 1

        assert first_ack.status_code == second_ack.status_code == 200
        assert poll.status_code == poll_after_duplicate.status_code == 410
        assert duplicate.response.status_code == 409
        assert rejected.response.status_code == 503
        assert health.status_code == 200
        assert worker.calls == 1

    async def test_outcome_reservation_saturation_preserves_existing_control_paths(self) -> None:
        """A full serialized-outcome budget blocks new execution without blocking poll, duplicate, or health."""
        worker = _Worker()
        app = create_rpc_app(worker)
        app.state.rpc_server._store = CallStore(max_active_calls=2, max_unacknowledged_outcome_bytes=1024)
        query = {"value": "small"}
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                accepted = await _submit(client, "demo_declared_result_budget", query, call_id="fixed")
                await _poll_until_done(client, "fixed")
                duplicate = await _submit(client, "demo_declared_result_budget", query, call_id="fixed")
                rejected = await _submit(client, "demo_declared_result_budget", query, call_id="new")
                poll = await client.get("/v1/calls/fixed", params={"timeout": 0.0})
                health = await client.get("/v1/health")

        assert accepted.response.status_code == 200
        assert duplicate.response.status_code == 409
        assert rejected.response.status_code == 503
        assert poll.status_code == 200 and poll.json()["status"] == "success"
        assert health.status_code == 200
        assert worker.calls == 1

    async def test_oversized_result_becomes_a_small_pollable_failure_envelope(self) -> None:
        """A method exceeding its declared result limit returns a bounded failure instead of losing its call."""
        worker = _Worker()
        async with _client(worker) as client:
            submitted = await _submit(
                client,
                "demo_declared_result_budget",
                {"value": "x" * 2048},
                call_id="oversized-result",
            )
            outcome = await _poll_until_done(client, submitted.call_id)

        assert outcome["status"] == "failed"
        assert "RpcOutcomeTooLargeError" in outcome["error"]
        assert len(json.dumps(outcome).encode()) <= 1024


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
