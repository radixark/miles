import uuid

import httpx
import pytest
from pydantic import ValidationError
from tests.fast.utils.workers.e2e.e2e_worker import E2eWorker

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


class TestNoRequestIsSent:
    async def test_unknown_method_is_an_attribute_error(self, dead_proxy, make_handle):
        """A method the worker does not define fails before any request goes out."""
        handle = make_handle(dead_proxy)
        with pytest.raises(AttributeError, match="no rpc method"):
            _ = handle.no_such_method
        assert dead_proxy.requests == []

    async def test_missing_required_argument(self, dead_proxy, make_handle):
        """A missing argument is caught locally, not by the server."""
        handle = make_handle(dead_proxy)
        with pytest.raises(ValidationError):
            await handle.demo_sync(a=1)
        assert dead_proxy.requests == []

    async def test_unknown_argument(self, dead_proxy, make_handle):
        """An argument the method does not declare is caught locally."""
        handle = make_handle(dead_proxy)
        with pytest.raises(ValidationError):
            await handle.demo_sync(a=1, b=2, c=3)
        assert dead_proxy.requests == []

    async def test_wrong_argument_type(self, dead_proxy, make_handle):
        """An uncoercible argument is caught locally."""
        handle = make_handle(dead_proxy)
        with pytest.raises(ValidationError):
            await handle.demo_sync(a="not-a-number", b=2)
        assert dead_proxy.requests == []

    async def test_positional_arguments_are_rejected(self, dead_proxy, make_handle):
        """Calls are keyword-only, so positional arguments fail locally."""
        handle = make_handle(dead_proxy)
        with pytest.raises(TypeError):
            await handle.demo_sync(1, 2)
        assert dead_proxy.requests == []


class TestHandleConstruction:
    def test_reserved_method_name_is_rejected(self):
        """A worker whose method shadows a handle attribute is refused at construction."""

        class Shadowing:
            async def wait_ready(self, timeout: float) -> None:
                pass

        with pytest.raises(TypeError, match="shadow"):
            RpcWorkerHandle(Shadowing, server_url="http://127.0.0.1:9")

    def test_worker_without_public_methods_is_rejected(self):
        """A worker with nothing to expose is refused."""

        class Empty:
            def _demo_hidden(self) -> int:
                return 1

        with pytest.raises(TypeError):
            RpcWorkerHandle(Empty, server_url="http://127.0.0.1:9")

    def test_worker_with_unannotated_method_is_rejected(self):
        """A method missing annotations is refused on the client too, matching the server."""

        class Unannotated:
            def demo_unannotated(self, x):
                return x

        with pytest.raises(TypeError):
            RpcWorkerHandle(Unannotated, server_url="http://127.0.0.1:9")

    async def test_trailing_slash_in_server_url(self, server, make_handle):
        """A server url with a trailing slash still produces valid request paths."""
        handle = make_handle(f"{server.url}/")
        assert await handle.demo_sync(a=1, b=1) == 2

    async def test_client_and_server_agree_on_the_method_set(self, server):
        """Every method the client exposes is routable on the server, and private ones are not."""
        handle = RpcWorkerHandle(E2eWorker, server_url=server.url)
        assert "demo_sync" in handle._specs and "_bump" not in handle._specs

        async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as client:
            for name in handle._specs:
                response = await client.post(
                    f"/v1/{name}", json={"call_id": uuid.uuid4().hex, "query": {"__unknown__": 1}}
                )
                assert response.status_code == 400, name

            private = await client.post("/v1/_bump", json={"call_id": uuid.uuid4().hex, "query": {}})
            assert private.status_code == 404
