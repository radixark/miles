import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import NamedTuple

import httpx

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.server.app import create_rpc_app


class _Item(StrictBaseModel):
    name: str
    value: int


class _Worker:
    async def demo_async_model(self, name: str) -> _Item:
        return _Item(name=name, value=len(name))

    async def demo_raises(self, message: str) -> None:
        raise RuntimeError(message)


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

    async def test_unknown_call_id_404(self):
        """Querying an unknown call id returns 404."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/calls/missing", params={"timeout": 0.0})
            assert response.status_code == 404

    async def test_invalid_poll_timeout_400(self):
        """A negative long-poll timeout is a client error, reported as 400."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/calls/whatever", params={"timeout": -1.0})
            assert response.status_code == 400
