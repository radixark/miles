import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.client import handle as rpc_handle_module
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import RpcWorkerCallError
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.worker_handle import WorkerUnreachableError


class _Item(StrictBaseModel):
    name: str
    value: int


class _Worker:
    def __init__(self) -> None:
        self.calls = 0

    async def demo_default_arg(self, a: int, b: int = 100) -> int:
        self.calls += 1
        return a + b

    async def demo_model(self, item: _Item) -> _Item:
        return item

    async def demo_raises(self) -> None:
        raise ValueError("exploded")


class _HookTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        app: Any,
        hook: Callable[[httpx.Request], httpx.Response | None] | None = None,
    ) -> None:
        self.requests = 0
        self.request_times: list[float] = []
        self.seen: list[httpx.Request] = []
        self._inner = httpx.ASGITransport(app=app) if app is not None else None
        self._hook = hook

    def switch_to(self, app: Any) -> None:
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        self.seen.append(request)
        self.request_times.append(time.monotonic())
        if self._hook is not None:
            replacement = self._hook(request)
            if replacement is not None:
                return replacement
        assert self._inner is not None
        return await self._inner.handle_async_request(request)

    def polls(self) -> list[httpx.Request]:
        return [r for r in self.seen if r.method == "GET" and "/v1/calls/" in str(r.url)]


@contextlib.asynccontextmanager
async def _running_app(worker: object) -> AsyncIterator:
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        yield app


@contextlib.asynccontextmanager
async def _handle_over(
    transport: httpx.AsyncBaseTransport,
    worker_cls: type = _Worker,
    call_timeout_seconds: float = 3600.0,
) -> AsyncIterator[RpcWorkerHandle]:
    async with httpx.AsyncClient(transport=transport) as http_client:
        yield RpcWorkerHandle(
            worker_cls,
            server_url="http://testserver",
            call_timeout_seconds=call_timeout_seconds,
            http_client=http_client,
        )


class TestTypedCalls:
    async def test_scalar_roundtrip(self):
        """A typed call returns the deserialized scalar result."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_default_argument_applied(self):
        """An omitted defaulted argument uses the declared default value."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            assert await handle.demo_default_arg(a=1) == 101

    async def test_model_result_revived(self):
        """Pydantic model results come back as real model instances."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            result = await handle.demo_model(item=_Item(name="x", value=7))
            assert result == _Item(name="x", value=7) and isinstance(result, _Item)

    async def test_remote_exception_raises_call_error(self):
        """A remote business exception raises RpcWorkerCallError carrying the traceback."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(RpcWorkerCallError, match="ValueError"):
                await handle.demo_raises()


class TestLocalValidation:
    async def test_worker_shadowing_handle_method_rejected(self):
        """A worker exposing a reserved handle method name fails at client creation."""

        class Shadowing:
            async def wait_ready(self, timeout: float) -> None:
                pass

        with pytest.raises(TypeError, match="shadow"):
            RpcWorkerHandle(Shadowing, server_url="http://testserver")

    async def test_method_typo_raises_attribute_error(self):
        """Calling a method the worker class does not define fails locally."""
        handle = RpcWorkerHandle(_Worker, server_url="http://testserver")
        with pytest.raises(AttributeError, match="no rpc method"):
            _ = handle.demo_defualt_arg

    async def test_bad_argument_type_fails_before_any_request(self):
        """Locally invalid arguments raise ValidationError without touching the network."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with pytest.raises(ValidationError):
                    await handle.demo_default_arg(a="not-an-int")
                assert transport.requests == 0

    async def test_unknown_argument_fails_before_any_request(self):
        """Unknown keyword arguments raise ValidationError without touching the network."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with pytest.raises(ValidationError):
                    await handle.demo_default_arg(a=1, nope=2)
                assert transport.requests == 0


class TestCallTimeout:
    async def test_pending_past_deadline_raises_timeout(self):
        """A call that never finishes inside its deadline raises TimeoutError."""

        class _NeverFinishes:
            async def demo_default_arg(self, a: int, b: int = 100) -> int:
                await asyncio.Event().wait()
                return a + b

        async with (
            _running_app(_NeverFinishes()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_NeverFinishes, call_timeout_seconds=0.2) as handle,
        ):
            with pytest.raises(TimeoutError):
                await handle.demo_default_arg(a=1)


class TestWaitReady:
    async def test_wait_ready_returns_when_healthy(self):
        """A live server satisfies wait_ready immediately."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            await handle.wait_ready(timeout=5.0)

    async def test_wait_ready_times_out_against_dead_server(self):
        """A server that never answers surfaces as WorkerUnreachableError."""
        async with _handle_over(_HookTransport(None, hook=_always_unreachable)) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=0.2)

    def test_default_ready_timeout_is_generous(self):
        """Readiness waits long enough for a heavy worker to import and load weights."""
        assert rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS == 600.0


def _always_unreachable(request: httpx.Request) -> httpx.Response | None:
    raise httpx.ConnectError("injected transport failure", request=request)
