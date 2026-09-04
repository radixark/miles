import asyncio
import contextlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tests.fast.utils.workers.rpc.client.fake_transports import PollWindowRecordingTransport, StallingPollTransport

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.client import call as rpc_client_module
from miles.utils.workers.rpc.client import handle as rpc_handle_module
from miles.utils.workers.rpc.client import misc as rpc_misc_module
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import RpcProtocolError, RpcWorkerCallError, ServerRestartedError
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, EXPECTED_BOOT_UUID_HEADER, HEALTH_PATH
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


class _PositionalWorker:
    def demo_join(self, first: str, second: str, *, separator: str = "-") -> str:
        return f"{first}{separator}{second}"

    def demo_nothing(self) -> int:
        return 0


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


class _PollRecordingTransport(_HookTransport):
    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.poll_statuses: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        if request.method == "GET" and "/v1/calls/" in str(request.url):
            await response.aread()
            self.poll_statuses.append(response.json()["status"])
        return response


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
    require_stable_boot_uuid: bool = False,
    call_timeout_seconds: float = 3600.0,
    ready_timeout_seconds: float = rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS,
    follow_redirects: bool = False,
) -> AsyncIterator[RpcWorkerHandle]:
    async with httpx.AsyncClient(transport=transport, follow_redirects=follow_redirects) as http_client:
        yield RpcWorkerHandle(
            worker_cls,
            server_url="http://testserver",
            require_stable_boot_uuid=require_stable_boot_uuid,
            call_timeout_seconds=call_timeout_seconds,
            ready_timeout_seconds=ready_timeout_seconds,
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

    async def test_positional_arguments_are_bound_by_name(self):
        """A positional call is bound to parameter names client-side, so the wire stays keyword-shaped."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            assert await handle.demo_default_arg(1, 2) == 3

    async def test_a_positional_and_keyword_duplicate_is_rejected_locally(self):
        """The same parameter given twice is a caller bug and must not reach the server."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(TypeError, match="multiple values"):
                await handle.demo_default_arg(1, a=2)

    async def test_excess_positional_arguments_are_rejected_locally(self):
        """More positionals than the method declares is a caller bug and must not reach the server."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(TypeError, match="at most 2 positional arguments"):
                await handle.demo_default_arg(1, 2, 3)

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


class TestReadyHandshake:
    @pytest.mark.parametrize(
        ("handle_kwargs", "expected_timeout"),
        [
            ({}, rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS),
            ({"ready_timeout_seconds": 2.5}, 2.5),
        ],
    )
    async def test_stable_handshake_uses_configured_ready_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        handle_kwargs: dict[str, float],
        expected_timeout: float,
    ) -> None:
        """A stable-boot handshake uses the default or configured ready timeout."""
        observed_timeouts: list[float] = []

        async def wait_ready(*, timeout: float) -> None:
            observed_timeouts.append(timeout)

        async def run(call: rpc_client_module.RpcCall) -> int:
            return 3

        def fail_if_requested(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(fail_if_requested)) as http_client:
            handle = RpcWorkerHandle(
                _Worker,
                server_url="http://testserver",
                require_stable_boot_uuid=True,
                http_client=http_client,
                **handle_kwargs,
            )
            monkeypatch.setattr(handle, "wait_ready", wait_ready)
            monkeypatch.setattr(rpc_client_module.RpcCall, "run", run)

            assert await handle.demo_default_arg(a=1, b=2) == 3

        assert rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS == 600.0
        assert observed_timeouts == [expected_timeout]

    async def test_implicit_stable_handshake_honors_ready_timeout_before_submit(self) -> None:
        """The first call on a stable-boot handle really health-checks on the ready timeout before submitting."""

        class _HealthTimeoutRecordingTransport(_HookTransport):
            def __init__(self, app: Any) -> None:
                super().__init__(app)
                self.health_boot_uuid: str | None = None
                self.health_read_timeouts: list[float] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                is_health = request.method == "GET" and str(request.url).endswith(HEALTH_PATH)
                if is_health:
                    self.health_read_timeouts.append(request.extensions["timeout"]["read"])
                response = await super().handle_async_request(request)
                if is_health:
                    self.health_boot_uuid = response.headers[BOOT_UUID_HEADER]
                return response

        async with _running_app(_Worker()) as app:
            transport = _HealthTimeoutRecordingTransport(app)
            async with _handle_over(transport, require_stable_boot_uuid=True, ready_timeout_seconds=2.5) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert transport.seen[0].method == "GET" and str(transport.seen[0].url).endswith(HEALTH_PATH)
        submit_request = next(request for request in transport.seen if request.method == "POST")
        assert transport.health_boot_uuid is not None
        assert submit_request.headers[EXPECTED_BOOT_UUID_HEADER] == transport.health_boot_uuid
        assert transport.health_read_timeouts
        assert 2.0 < transport.health_read_timeouts[0] <= 2.5


class TestSubmitRetry:
    async def test_never_reached_errors_retried_until_success(self, fast_retries: None) -> None:
        """Never-reached submit failures are retried until one succeeds."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "POST"))
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3
                assert transport.requests >= 3

    async def test_pool_timeout_is_retried_until_submit_succeeds(self, fast_retries: None) -> None:
        """A pool timeout never handed the submit to the server, so it is retried until one lands."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "POST", error_type=httpx.PoolTimeout))
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

            assert len([r for r in transport.seen if r.method == "POST"]) == 3

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    async def test_server_error_submit_gives_up_without_retry(
        self,
        status_code: int,
        fast_retries: None,
    ) -> None:
        """A submit 5xx becomes unreachable after exactly one attempt."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_status_hook(status_code=status_code, times=1, method="POST"),
            )
            async with _handle_over(transport) as handle:
                with pytest.raises(WorkerUnreachableError):
                    await handle.demo_default_arg(a=1, b=2)

        assert transport.requests == 1

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
        """A dead server exhausts the retry window, which also bounds when requests stop."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

            window = rpc_client_module.SUBMIT_RETRY_WINDOW_SECONDS
            assert transport.request_times, "expected at least one submit attempt"
            assert max(transport.request_times) - started <= window

    async def test_backoff_grows_between_attempts(self, fast_retries: None) -> None:
        """Retry gaps grow instead of hammering the server at a fixed rate."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

            gaps = [b - a for a, b in zip(transport.request_times, transport.request_times[1:], strict=False)]
            assert len(gaps) >= 3, f"expected several retries, got {len(gaps)}"
            assert gaps[0] >= rpc_client_module.RETRY_INITIAL_DELAY_SECONDS
            assert gaps[1] > gaps[0]

    @pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ReadError])
    async def test_post_wire_submit_error_gives_up_without_retry(
        self,
        error_type: type[httpx.TransportError],
        fast_retries: None,
    ) -> None:
        """A post-wire submit transport error gives up after one attempt."""
        transport = _HookTransport(
            None,
            hook=_fail_hook(-1, "POST", error_type=error_type),
        )
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

        assert transport.requests == 1

    async def test_protocol_error_not_retried(self) -> None:
        """A 4xx protocol error raises immediately without retries."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_WiderWorker) as handle:
                with pytest.raises(RpcProtocolError) as exc_info:
                    await handle.demo_extra()
                assert exc_info.value.status_code == 404
                assert transport.requests == 1


class TestPollFailure:
    async def test_server_death_after_submit_exhausts_as_timeout(self, fast_retries: None) -> None:
        """Poll transport failures exhaust as TimeoutError."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(-1, "GET", "/v1/calls/"))
            async with _handle_over(transport, call_timeout_seconds=0.2) as handle:
                with pytest.raises(TimeoutError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_transient_poll_loss_recovers(self, fast_retries: None) -> None:
        """A poll failure before the deadline is retried and the call still returns."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "GET", "/v1/calls/"))
            async with _handle_over(transport, call_timeout_seconds=10.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_transient_poll_5xx_recovers(self, fast_retries: None) -> None:
        """A transient poll 5xx is retried and the call succeeds."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_status_hook(
                    status_code=503,
                    times=1,
                    method="GET",
                    path_fragment="/v1/calls/",
                ),
            )
            async with _handle_over(transport, call_timeout_seconds=2.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert len(transport.polls()) >= 2

    async def test_poll_non_200_raises_protocol_error(self, fast_retries: None) -> None:
        """A poll 4xx raises RpcProtocolError without retry."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_status_hook(
                    status_code=404,
                    times=1,
                    method="GET",
                    path_fragment="/v1/calls/",
                ),
            )
            async with _handle_over(transport, call_timeout_seconds=2.0) as handle:
                with pytest.raises(RpcProtocolError) as exc_info:
                    await handle.demo_default_arg(a=1, b=2)

        assert exc_info.value.status_code == 404
        assert len(transport.polls()) == 1

    async def test_long_poll_timeout_silently_repolls(self, fast_retries: None) -> None:
        """A long-poll timeout is silently retried."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_fail_hook(1, "GET", "/v1/calls/", error_type=httpx.ReadTimeout),
            )
            async with _handle_over(transport, call_timeout_seconds=2.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert len(transport.polls()) >= 2


class TestCallTimeout:
    async def test_pending_past_deadline_raises_timeout(self):
        """A call still pending past the call timeout raises, and polls never outlast the budget."""
        worker = _Worker()
        async with _running_app(worker) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, call_timeout_seconds=0.3) as handle:
                with pytest.raises(TimeoutError):
                    await handle.demo_hang()

                assert transport.polls(), "expected at least one poll"
                assert all(float(r.url.params["timeout"]) <= 0.3 for r in transport.polls())
                worker.block_forever.set()


class TestLongPoll:
    async def test_poll_leaves_the_server_room_to_answer_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The client asks the server to hang for less than it will wait, so a pending answer arrives."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.4)
        worker = _Worker()
        async with _running_app(worker) as app:
            transport = _PollRecordingTransport(app)
            async with _handle_over(transport, call_timeout_seconds=5.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(1.0)
                worker.block_forever.set()

                assert await pending == "done"
                assert "pending" in transport.poll_statuses
                assert all(float(request.url.params["timeout"]) < 0.4 for request in transport.polls())

    async def test_each_poll_waits_only_for_its_own_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A poll's local http timeout follows the poll window, not the whole remaining call budget."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.4)
        worker = _Worker()

        class _PollTimeoutRecordingTransport(_HookTransport):
            def __init__(self, app: Any) -> None:
                super().__init__(app)
                self.poll_read_timeouts: list[float] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if request.method == "GET" and "/v1/calls/" in str(request.url):
                    self.poll_read_timeouts.append(request.extensions["timeout"]["read"])
                return await super().handle_async_request(request)

        async with _running_app(worker) as app:
            transport = _PollTimeoutRecordingTransport(app)
            async with _handle_over(transport, call_timeout_seconds=30.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(1.0)
                worker.block_forever.set()

                assert await pending == "done"

        assert transport.poll_read_timeouts
        assert max(transport.poll_read_timeouts) <= 0.4

    async def test_outer_poll_timeout_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A poll abandoned by its own abort deadline is repolled instead of failing the call."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.1)
        monkeypatch.setattr(rpc_misc_module, "_ABORT_SLACK_SECONDS", 0.05)

        async with _running_app(_Worker()) as app:
            transport = StallingPollTransport(app, stalled_polls=2)
            async with _handle_over(transport, call_timeout_seconds=5.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert transport.polls >= 3

    @pytest.mark.parametrize("slack_seconds", [5.0, 0.1])
    async def test_poll_window_keeps_positive_server_wait_and_bounded_client_slack(
        self, monkeypatch: pytest.MonkeyPatch, slack_seconds: float
    ) -> None:
        """Every poll asks the server for a positive share of the window while keeping client-side slack."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.4)
        monkeypatch.setattr(rpc_client_module, "POLL_SLACK_SECONDS", slack_seconds)
        worker = _Worker()

        async with _running_app(worker) as app:
            transport = PollWindowRecordingTransport(app)
            async with _handle_over(transport, call_timeout_seconds=5.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(1.0)
                worker.block_forever.set()

                assert await pending == "done"

        assert len(transport.poll_windows) >= 2
        for window in transport.poll_windows:
            assert 0.0 < window.server_seconds < window.client_seconds <= 0.4
            assert window.server_seconds >= window.client_seconds / 2
            assert window.client_seconds - window.server_seconds <= slack_seconds

    async def test_poll_slack_default_stays_put(self) -> None:
        """The poll slack keeps the value that lets the server answer before the client gives up."""
        assert rpc_client_module.POLL_SLACK_SECONDS == 5.0

    async def test_restart_during_polling_is_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pinned handle notices a server swap that happens while it is already polling."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.1)
        worker = _Worker()
        async with _running_app(worker) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True, call_timeout_seconds=10.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(0.5)
                transport.switch_to(second_app)

                with pytest.raises(ServerRestartedError):
                    await pending
                worker.block_forever.set()


class TestBootUuid:
    async def test_restart_detected_when_required_stable(self):
        """A server restart between calls raises ServerRestartedError."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_restart_ignored_by_default(self):
        """Without the stable-boot-uuid requirement a restart is tolerated."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_stale_pin_refused_before_side_effects(self):
        """In stable mode a submit reaching a restarted server never runs there."""
        second_worker = _Worker()
        async with _running_app(_Worker()) as first_app, _running_app(second_worker) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)
                assert second_worker.calls == 0

    async def test_post_wire_loss_is_not_retried_on_restarted_server(self, fast_retries: None) -> None:
        """A post-wire submit loss is not retried onto a restarted server."""
        dropped: list[bool] = []
        second_worker = _Worker()

        def drop_first_submit_response(request: httpx.Request) -> httpx.Response | None:
            if request.method == "POST" and not dropped:
                dropped.append(True)
                raise httpx.ReadError("response lost after the server accepted it", request=request)
            return None

        async with _running_app(_Worker()) as first_app, _running_app(second_worker) as second_app:
            transport = _HookTransport(first_app, hook=drop_first_submit_response)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.switch_to(second_app)
                with pytest.raises(WorkerUnreachableError):
                    await handle.demo_default_arg(a=1, b=2)
                assert second_worker.calls == 0

    async def test_missing_header_rejected_in_stable_mode(self):
        """A response without the boot uuid header is refused in stable mode."""

        def strip_header(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"}, request=request)

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=strip_header)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_server_ignoring_the_expectation_header_is_still_refused(self) -> None:
        """A rolled-back server that never checks the expectation header is caught by the client."""

        def answer_from_another_boot(request: httpx.Request) -> httpx.Response | None:
            if request.method != "POST":
                return None
            return httpx.Response(
                200, json={"status": "submitted"}, headers={BOOT_UUID_HEADER: "some-other-boot"}, request=request
            )

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=answer_from_another_boot)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                with pytest.raises(ServerRestartedError, match="some-other-boot"):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_response_losing_the_header_after_pinning_is_refused(self) -> None:
        """Once pinned, a successful response without the header is a restart, not a silent pass."""

        def strip_header(request: httpx.Request) -> httpx.Response | None:
            if request.method != "POST":
                return None
            return httpx.Response(200, json={"status": "submitted"}, request=request)

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=strip_header)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                with pytest.raises(ServerRestartedError, match=BOOT_UUID_HEADER):
                    await handle.demo_default_arg(a=1, b=2)

    @pytest.mark.parametrize("status_code", [500, 502, 503])
    async def test_header_less_error_before_pinning_is_not_a_restart(
        self, fast_retries: None, status_code: int
    ) -> None:
        """A transient error response carrying no boot uuid is retried, not reported as a restart."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app, hook=_status_hook(status_code=status_code, times=1, method="GET", path_fragment="/v1/health")
            )
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                assert transport.requests > 1

    async def test_wait_ready_keeps_original_pin_after_restart(self) -> None:
        """wait_ready refuses a new boot after the first pin."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.wait_ready(timeout=5.0)


class TestWaitReady:
    async def test_wait_ready_returns_when_healthy(self):
        """wait_ready returns promptly against a healthy server."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            await handle.wait_ready(timeout=5.0)

    async def test_wait_ready_times_out_against_dead_server(self, fast_retries):
        """wait_ready keeps polling within its window and then raises WorkerUnreachableError."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=0.2)
            assert time.monotonic() - started <= 2.0
            assert transport.requests > 1

    async def test_wait_ready_survives_initial_failures(self, fast_retries):
        """wait_ready keeps retrying through initial connection failures."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2))
            async with _handle_over(transport) as handle:
                await handle.wait_ready(timeout=5.0)
                assert transport.requests >= 3

    async def test_wait_ready_probe_timeout_is_clamped_to_remaining_budget(self, fast_retries) -> None:
        """A health probe never gets more time than the caller's remaining readiness budget."""

        class _HealthTimeoutRecordingTransport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.observations: list[tuple[float, float]] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.observations.append((time.monotonic(), request.extensions["timeout"]["read"]))
                raise httpx.ConnectError("refused", request=request)

        transport = _HealthTimeoutRecordingTransport()
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=0.4)

        assert len(transport.observations) >= 2
        first_request_at, first_timeout = transport.observations[0]
        for request_at, read_timeout in transport.observations[1:]:
            elapsed = request_at - first_request_at
            assert read_timeout <= first_timeout - elapsed + 0.01
        assert transport.observations[-1][1] < first_timeout - 0.1

    async def test_wait_ready_protocol_error_is_not_retried(self) -> None:
        """A 4xx health response is a protocol error, so wait_ready gives up after one request."""
        transport = _HookTransport(
            None,
            hook=_status_hook(status_code=404, times=-1, method="GET", path_fragment=HEALTH_PATH),
        )
        async with _handle_over(transport) as handle:
            with pytest.raises(RpcProtocolError) as exc_info:
                await handle.wait_ready(timeout=5.0)

        assert exc_info.value.status_code == 404
        assert transport.requests == 1

    async def test_wait_ready_persistent_server_error_exhausts_as_unreachable(self, fast_retries) -> None:
        """A health endpoint stuck on 5xx is retried until the deadline and then reported unreachable."""
        timeout = 0.5
        transport = _HookTransport(
            None,
            hook=_status_hook(status_code=503, times=-1, method="GET", path_fragment=HEALTH_PATH),
        )
        async with _handle_over(transport) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=timeout)

        assert transport.requests >= 3
        assert transport.request_times[-1] - started >= timeout - 0.1


class TestPositionalCalls:
    async def test_positional_arguments_reach_the_worker_in_declaration_order(self):
        """Positionals bind to parameters left to right, so an order-sensitive method sees the caller's order."""
        async with (
            _running_app(_PositionalWorker()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_PositionalWorker) as handle,
        ):
            assert await handle.demo_join("left", "right") == "left-right"

    async def test_positional_arguments_travel_as_named_query_fields(self):
        """The wire stays keyword-shaped, so the server never has to know the caller used positionals."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                await handle.demo_join("left", "right")

            submits = [r for r in transport.seen if r.method == "POST"]
            assert [json.loads(r.content)["query"] for r in submits] == [
                {"first": "left", "second": "right", "separator": "-"}
            ]

    async def test_a_positional_and_keyword_mix_leaves_the_remaining_default_intact(self):
        """A short positional call omits the parameters it does not fill, so their declared defaults still apply."""
        async with (
            _running_app(_PositionalWorker()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_PositionalWorker) as handle,
        ):
            assert await handle.demo_join("left", second="right") == "left-right"

    async def test_a_keyword_only_parameter_is_reachable_alongside_positionals(self):
        """Positionals fill the ordinary parameters while a keyword-only parameter still arrives by name."""
        async with (
            _running_app(_PositionalWorker()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_PositionalWorker) as handle,
        ):
            assert await handle.demo_join("left", "right", separator="+") == "left+right"

    async def test_a_positional_for_a_keyword_only_parameter_fails_before_any_request(self):
        """A keyword-only parameter cannot be filled positionally, and the bad call never reaches the network."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                with pytest.raises(TypeError, match="at most 2 positional arguments"):
                    await handle.demo_join("left", "right", "+")
                assert transport.requests == 0

    async def test_a_duplicated_parameter_fails_before_any_request(self):
        """The same parameter given positionally and by keyword raises locally instead of one value winning."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                with pytest.raises(TypeError, match=r"multiple values for \['first'\]"):
                    await handle.demo_join("left", first="other")
                assert transport.requests == 0

    async def test_a_parameterless_method_rejects_a_positional_argument(self):
        """A method taking only the receiver has no parameter name to bind a positional to."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                with pytest.raises(TypeError, match="at most 0 positional arguments"):
                    await handle.demo_nothing(1)
                assert transport.requests == 0
