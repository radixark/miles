import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

import httpx

from miles.utils.workers.rpc.server import core as core_module
from miles.utils.workers.rpc.server.app import create_rpc_app


async def _submit(raw, method: str, call_id: str, query: dict, **extra):
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query, **extra})


class _GatedWorker:
    def __init__(self) -> None:
        self.gate = asyncio.Event()

    async def demo_wait_for_gate(self) -> str:
        await self.gate.wait()
        return "released"


@contextlib.asynccontextmanager
async def _in_process_client(worker: object) -> AsyncIterator[httpx.AsyncClient]:
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


class TestMethodLookup:
    async def test_unknown_method_404(self, raw, tag):
        """Submitting to a method the worker does not define is a 404."""
        response = await _submit(raw, "no_such_method", tag, {})
        assert response.status_code == 404
        assert "no_such_method" in response.json()["detail"]

    async def test_private_method_not_exposed(self, raw, tag):
        """Underscore-prefixed worker methods are not reachable."""
        assert (await _submit(raw, "_bump", tag, {"tag": "x"})).status_code == 404

    async def test_dunder_not_exposed(self, raw, tag):
        """Dunder attributes are not reachable."""
        assert (await _submit(raw, "__init__", tag, {})).status_code == 404

    async def test_health_is_not_a_submit_target(self, raw, tag):
        """POST /v1/health falls through to method lookup and 404s."""
        assert (await _submit(raw, "health", tag, {})).status_code == 404

    async def test_calls_path_is_not_a_submit_target(self, raw):
        """POST /v1/calls/... is method lookup, not a call query."""
        assert (await raw.post("/v1/calls", json={"call_id": "x", "query": {}})).status_code == 404


class TestEnvelopeValidation:
    async def test_malformed_json_400(self, raw):
        """A body that is not JSON is a client error, normalized to 400."""
        response = await raw.post("/v1/demo_sync", content=b"{not json", headers={"content-type": "application/json"})
        assert response.status_code == 400

    async def test_missing_call_id_400(self, raw):
        """An envelope without call_id is rejected."""
        assert (await raw.post("/v1/demo_sync", json={"query": {"a": 1, "b": 2}})).status_code == 400

    async def test_missing_query_400(self, raw, tag):
        """An envelope without query is rejected."""
        assert (await raw.post("/v1/demo_sync", json={"call_id": tag})).status_code == 400

    async def test_extra_envelope_field_400(self, raw, tag):
        """Unknown envelope fields are rejected rather than ignored."""
        response = await raw.post("/v1/demo_sync", json={"call_id": tag, "query": {"a": 1, "b": 2}, "bogus": 1})
        assert response.status_code == 400

    async def test_query_not_an_object_400(self, raw, tag):
        """A non-object query is rejected."""
        assert (await raw.post("/v1/demo_sync", json={"call_id": tag, "query": [1, 2]})).status_code == 400


class TestQueryValidation:
    async def test_coercible_strings_accepted(self, raw, tag):
        """Coercible values are accepted, pinning the non-strict query behaviour."""
        assert (await _submit(raw, "demo_sync", tag, {"a": "3", "b": "4"})).status_code == 200
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 5.0})).json()
        assert body["result"] == 7

    async def test_unknown_kwarg_400(self, raw, tag):
        """An argument the method does not declare is rejected."""
        response = await _submit(raw, "demo_sync", tag, {"a": 1, "b": 2, "c": 3})
        assert response.status_code == 400 and "c" in response.json()["detail"]

    async def test_missing_required_kwarg_400(self, raw, tag):
        """A missing required argument is rejected."""
        response = await _submit(raw, "demo_sync", tag, {"a": 1})
        assert response.status_code == 400 and "b" in response.json()["detail"]

    async def test_wrong_type_400(self, raw, tag):
        """An argument that cannot be coerced is rejected."""
        assert (await _submit(raw, "demo_sync", tag, {"a": "not-int", "b": 2})).status_code == 400


class TestCallLookup:
    async def test_unknown_call_id_404(self, raw):
        """Polling an unknown call id is a 404."""
        response = await raw.get("/v1/calls/deadbeef", params={"timeout": 0.0})
        assert response.status_code == 404 and "deadbeef" in response.json()["detail"]

    async def test_negative_poll_timeout_400(self, raw, tag):
        """A negative poll timeout is a client error."""
        await _submit(raw, "demo_sync", tag, {"a": 1, "b": 1})
        assert (await raw.get(f"/v1/calls/{tag}", params={"timeout": -1})).status_code == 400

    async def test_non_numeric_poll_timeout_400(self, raw, tag):
        """A non-numeric poll timeout is a client error."""
        await _submit(raw, "demo_sync", tag, {"a": 1, "b": 1})
        assert (await raw.get(f"/v1/calls/{tag}", params={"timeout": "abc"})).status_code == 400

    async def test_huge_poll_timeout_accepted(self, raw, tag):
        """A poll timeout beyond the server cap is accepted rather than rejected."""
        await _submit(raw, "demo_count_sync", tag, {"tag": tag})
        response = await raw.get(f"/v1/calls/{tag}", params={"timeout": 999999})
        assert response.status_code == 200

    async def test_omitted_poll_timeout_uses_server_default(self, raw, tag):
        """Polling without a timeout parameter uses the server default and returns the finished outcome."""
        await _submit(raw, "demo_sync", tag, {"a": 2, "b": 3})
        assert (await raw.get(f"/v1/calls/{tag}", params={"timeout": 5.0})).json()["status"] == "success"

        body = (await raw.get(f"/v1/calls/{tag}")).json()
        assert body["status"] == "success" and body["result"] == 5


class TestPollTimeoutClamp:
    async def test_huge_poll_timeout_is_clamped_to_the_server_cap(self, monkeypatch):
        """A huge poll timeout on a still pending call is clamped, so pending comes back within the cap."""
        monkeypatch.setattr(core_module, "MAX_POLL_TIMEOUT_SECONDS", 0.3)
        worker = _GatedWorker()

        async with _in_process_client(worker) as client:
            submitted = await client.post("/v1/demo_wait_for_gate", json={"call_id": "gated", "query": {}})
            assert submitted.status_code == 200

            started_at = time.monotonic()
            try:
                polled = client.get("/v1/calls/gated", params={"timeout": 999999})
                response = await asyncio.wait_for(polled, timeout=5.0)
            finally:
                worker.gate.set()
            elapsed = time.monotonic() - started_at

            assert response.json()["status"] == "pending"
            assert elapsed < 3.0
