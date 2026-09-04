import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.client import call as rpc_client_module
from miles.utils.workers.rpc.client import handle as rpc_handle_module
from miles.utils.workers.rpc.client import misc as rpc_misc_module
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import RpcProtocolError, RpcWorkerCallError
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.worker_handle import WorkerUnreachableError


class _Item(StrictBaseModel):
    name: str
    value: int


class _Worker:
    def __init__(self) -> None:
        self.block_forever = threading.Event()
        self.calls = 0

    def demo_default_arg(self, a: int, b: int = 100) -> int:
        self.calls += 1
        return a + b

    async def demo_model(self, item: _Item) -> _Item:
        return item

    def demo_raises(self) -> None:
        raise ValueError("exploded")

    def demo_hang(self) -> str:
        assert self.block_forever.wait(timeout=30.0)
        return "done"


class _WiderWorker(_Worker):
    def demo_extra(self) -> int:
        return 1


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


def _fail_hook(
    times: int,
    method: str | None = None,
    path_fragment: str = "",
    *,
    error_type: type[httpx.TransportError] = httpx.ConnectError,
) -> Callable[[httpx.Request], httpx.Response | None]:
    remaining = [times]

    def hook(request: httpx.Request) -> httpx.Response | None:
        matches = (method is None or request.method == method) and path_fragment in str(request.url)
        if matches and remaining[0] != 0:
            remaining[0] -= 1
            raise error_type("injected transport failure", request=request)
        return None

    return hook


def _status_hook(
    *,
    status_code: int,
    times: int,
    method: str,
    path_fragment: str = "",
) -> Callable[[httpx.Request], httpx.Response | None]:
    remaining = [times]

    def hook(request: httpx.Request) -> httpx.Response | None:
        if request.method == method and path_fragment in str(request.url) and remaining[0] != 0:
            remaining[0] -= 1
            return httpx.Response(status_code, text="upstream busy", request=request)
        return None

    return hook


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.5)
    monkeypatch.setattr(rpc_client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(rpc_client_module, "RETRY_MAX_DELAY_SECONDS", 0.1)
    monkeypatch.setattr(rpc_handle_module, "RETRY_INITIAL_DELAY_SECONDS", 0.02)


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
    follow_redirects: bool = False,
) -> AsyncIterator[RpcWorkerHandle]:
    async with httpx.AsyncClient(transport=transport, follow_redirects=follow_redirects) as http_client:
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


class TestSubmitRetry:
    async def test_never_reached_errors_retried_until_success(self, fast_retries: None) -> None:
        """Never-reached submit failures are retried until one succeeds."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "POST"))
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_server_error_submit_gives_up_without_retry(self, fast_retries: None) -> None:
        """A 5xx submit may have reached the worker, so it is never retried."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_status_hook(status_code=503, times=1, method="POST"))
            async with _handle_over(transport) as handle:
                with pytest.raises(WorkerUnreachableError):
                    await handle.demo_default_arg(a=1, b=2)

            assert len([r for r in transport.seen if r.method == "POST"]) == 1

    async def test_one_stalled_attempt_does_not_consume_the_whole_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A submit that stalls is abandoned on its own attempt budget, not on the whole retry window."""
        monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 20.0)
        monkeypatch.setattr(rpc_client_module, "SUBMIT_ATTEMPT_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(rpc_misc_module, "_ABORT_SLACK_SECONDS", 0.05)

        class _StallingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                await asyncio.sleep(60.0)
                raise AssertionError("the stalled request should have been abandoned")

        async with _handle_over(_StallingTransport()) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)
            assert time.monotonic() - started < 5.0

    async def test_a_submit_attempt_never_outlives_the_remaining_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the retry window is shorter than the attempt budget, the attempt is cut down to the window."""
        monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.5)
        monkeypatch.setattr(rpc_client_module, "SUBMIT_ATTEMPT_TIMEOUT_SECONDS", 30.0)

        class _TimeoutRecordingTransport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.read_timeouts: list[float] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.read_timeouts.append(request.extensions["timeout"]["read"])
                raise httpx.ConnectError("refused", request=request)

        transport = _TimeoutRecordingTransport()
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

        assert transport.read_timeouts
        assert max(transport.read_timeouts) <= 0.5

    async def test_submit_attempt_budget_defaults_stay_put(self) -> None:
        """The submit attempt budget and its abort slack keep the values the retry design assumes."""
        assert rpc_client_module.SUBMIT_ATTEMPT_TIMEOUT_SECONDS == 10.0
        assert rpc_misc_module._ABORT_SLACK_SECONDS == 1.0

    async def test_a_redirected_submit_is_never_followed(self) -> None:
        """A redirect is refused outright, so an injected client cannot deliver one submit twice."""

        def redirect_submits(request: httpx.Request) -> httpx.Response | None:
            if request.method != "POST":
                return None
            return httpx.Response(307, headers={"location": "http://elsewhere/v1/demo_default_arg"}, request=request)

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=redirect_submits)
            async with _handle_over(transport, follow_redirects=True) as handle:
                with pytest.raises(RpcProtocolError):
                    await handle.demo_default_arg(a=1, b=2)
            assert transport.requests == 1

    async def test_a_late_connect_timeout_is_still_classified_as_never_reached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The abort deadline sits behind the http timeout, so a connect timeout stays retryable."""
        monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 6.0)
        monkeypatch.setattr(rpc_client_module, "SUBMIT_ATTEMPT_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(rpc_client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.02)
        monkeypatch.setattr(rpc_client_module, "RETRY_MAX_DELAY_SECONDS", 0.05)
        monkeypatch.setattr(rpc_misc_module, "_ABORT_SLACK_SECONDS", 0.4)

        class _LateConnectTimeout(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.attempts = 0

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.attempts += 1
                await asyncio.sleep(request.extensions["timeout"]["connect"] + 0.2)
                raise httpx.ConnectTimeout("connect timed out", request=request)

        transport = _LateConnectTimeout()
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)
        assert transport.attempts > 1

    async def test_window_exhausted_raises_unreachable(self, fast_retries: None) -> None:
        """Retrying a never-reached submit past its window surfaces as unreachable."""
        async with _handle_over(_HookTransport(None, hook=_fail_hook(-1, "POST"))) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

    async def test_backoff_grows_between_attempts(self, fast_retries: None) -> None:
        """Submit retries back off instead of hammering the server."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(3, "POST"))
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

            gaps = [b - a for a, b in zip(transport.request_times, transport.request_times[1:], strict=False)]
            assert gaps[1] > gaps[0]

    async def test_protocol_error_not_retried(self) -> None:
        """A rejection the server is sure about is not retried."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_status_hook(status_code=404, times=1, method="POST"))
            async with _handle_over(transport) as handle:
                with pytest.raises(RpcProtocolError):
                    await handle.demo_default_arg(a=1, b=2)

            assert len([r for r in transport.seen if r.method == "POST"]) == 1


class TestCallTimeout:
    async def test_pending_past_deadline_raises_timeout(self):
        """A call that never finishes inside its deadline raises TimeoutError."""

        worker = _Worker()
        async with (
            _running_app(worker) as app,
            _handle_over(httpx.ASGITransport(app=app), call_timeout_seconds=0.3) as handle,
        ):
            with pytest.raises(TimeoutError):
                await handle.demo_hang()
            worker.block_forever.set()


class TestWaitReady:
    async def test_wait_ready_returns_when_healthy(self):
        """A live server satisfies wait_ready immediately."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            await handle.wait_ready(timeout=5.0)

    async def test_wait_ready_times_out_against_dead_server(self, fast_retries):
        """wait_ready keeps polling within its window and then raises WorkerUnreachableError."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=0.2)
            assert transport.requests > 1

    async def test_wait_ready_survives_initial_failures(self, fast_retries):
        """Readiness tolerates a few early failures before the server answers."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "GET"))
            async with _handle_over(transport) as handle:
                await handle.wait_ready(timeout=5.0)

    def test_default_ready_timeout_is_generous(self):
        """Readiness waits long enough for a heavy worker to import and load weights."""
        assert rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS == 600.0
