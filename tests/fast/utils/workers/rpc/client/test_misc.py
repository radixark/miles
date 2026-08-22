import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from pydantic import ValidationError

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.rpc.client.misc import (
    BootUuidPin,
    RetryableResponseError,
    RpcProtocolError,
    RpcTransport,
    ServerRestartedError,
)
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, EXPECTED_BOOT_UUID_HEADER, HealthResponse


def _response(*, status_code: int = 200, boot_uuid: str | None = None) -> httpx.Response:
    headers = {} if boot_uuid is None else {BOOT_UUID_HEADER: boot_uuid}
    return httpx.Response(status_code=status_code, headers=headers)


def _transport_over(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
    *,
    pin: BootUuidPin | None = None,
) -> tuple[RpcTransport, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    boot_uuid_pin = pin or BootUuidPin(required=False, worker_cls_name="Worker")
    transport = RpcTransport(
        server_url="http://testserver",
        http_client=client,
        boot_uuid_pin=boot_uuid_pin,
    )
    return transport, client


class TestBootUuidPin:
    def test_optional_pin_never_requires_handshake(self) -> None:
        """An optional pin never records a response boot UUID."""
        pin = BootUuidPin(required=False, worker_cls_name="Worker")

        pin.verify(_response(boot_uuid="boot-a"))

        assert pin.needs_handshake() is False
        assert pin.expected is None

    def test_required_pin_rejects_missing_header(self) -> None:
        """A required pin rejects a response missing the boot UUID header."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")

        with pytest.raises(ServerRestartedError, match=BOOT_UUID_HEADER):
            pin.verify(_response())

    def test_required_pin_keeps_first_value(self) -> None:
        """A required pin keeps the first observed boot UUID."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")

        pin.verify(_response(boot_uuid="boot-a"))
        pin.verify(_response(boot_uuid="boot-a"))

        assert pin.needs_handshake() is False
        assert pin.expected == "boot-a"

    def test_required_pin_refuses_a_different_value_afterwards(self) -> None:
        """Once pinned, a response from another boot UUID is a restart."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        with pytest.raises(ServerRestartedError, match="boot-b"):
            pin.verify(_response(boot_uuid="boot-b"))

        assert pin.expected == "boot-a"

    def test_required_pin_refuses_a_missing_header_afterwards(self) -> None:
        """Once pinned, a response without the header is a restart, not a silent pass."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        with pytest.raises(ServerRestartedError, match=BOOT_UUID_HEADER):
            pin.verify(_response())


class TestRpcTransport:
    async def test_returns_validated_response_model(self) -> None:
        """A successful request returns the validated protocol model."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"}, request=request)

        transport, client = _transport_over(handler)
        async with client:
            response = await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )

        assert response == HealthResponse()

    @pytest.mark.parametrize("status_code", [500, 502, 503, 599])
    async def test_server_status_raises_retryable_response(self, status_code: int) -> None:
        """Any 5xx response is classified as retryable."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text="busy", request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(RetryableResponseError, match=str(status_code)):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

    @pytest.mark.parametrize("status_code", [201, 400, 404, 409, 499])
    async def test_non_200_non_5xx_raises_protocol_error(self, status_code: int) -> None:
        """A non-200 non-5xx response is a protocol error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text="rejected", request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(RpcProtocolError) as exc_info:
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

        assert exc_info.value.status_code == status_code

    async def test_outer_timeout_enforces_request_budget(self) -> None:
        """The transport cancels a request exceeding its explicit budget."""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.Event().wait()
            return httpx.Response(200, request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(TimeoutError):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=0.01,
                    response_model=HealthResponse,
                )

    async def test_pinned_boot_uuid_is_sent_as_request_header(self) -> None:
        """A pinned transport sends the expected boot UUID on every request."""
        seen_headers: list[str | None] = []
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers.get(EXPECTED_BOOT_UUID_HEADER))
            return httpx.Response(
                200,
                headers={BOOT_UUID_HEADER: "boot-a"},
                json={"status": "ok"},
                request=request,
            )

        transport, client = _transport_over(handler, pin=pin)
        async with client:
            await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )
            await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )

        assert seen_headers == ["boot-a", "boot-a"]

    async def test_caller_headers_survive_but_cannot_override_the_pin(self) -> None:
        """Caller headers are passed through, while a caller-supplied expectation loses to the pinned uuid."""
        seen: list[httpx.Headers] = []
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return httpx.Response(200, headers={BOOT_UUID_HEADER: "boot-a"}, json={"status": "ok"}, request=request)

        transport, client = _transport_over(handler, pin=pin)
        async with client:
            await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
                headers={"x-trace": "keep-me", EXPECTED_BOOT_UUID_HEADER: "caller-guess"},
            )

        assert seen[0]["x-trace"] == "keep-me"
        assert seen[0][EXPECTED_BOOT_UUID_HEADER] == "boot-a"

    async def test_malformed_200_response_is_rejected(self) -> None:
        """A 200 response whose body is not JSON fails loudly instead of yielding a result."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all", request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(json.JSONDecodeError):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

    async def test_success_status_with_invalid_response_envelope_is_rejected(self) -> None:
        """A 200 response whose JSON does not match the protocol model is refused, not passed on."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "made-up", "extra": 1}, request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(ValidationError):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

    async def test_without_client_override_uses_general_http_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without an injected client the request goes through the shared general http client."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"status": "ok"}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setitem(GeneralHttpClientProvider._clients, asyncio.get_running_loop(), client)
        transport = RpcTransport(
            server_url="http://testserver",
            http_client=None,
            boot_uuid_pin=BootUuidPin(required=False, worker_cls_name="Worker"),
        )

        async with client:
            response = await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )

        assert response == HealthResponse()
        assert seen == ["http://testserver/v1/health"]

    async def test_boot_uuid_mismatch_status_raises_server_restarted(self) -> None:
        """A 412 response is mapped to ServerRestartedError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                412,
                headers={BOOT_UUID_HEADER: "boot-b"},
                text="stale boot",
                request=request,
            )

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(ServerRestartedError, match="restarted"):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )


class TestUnpinningThePin:
    def test_unpinning_forgets_the_pinned_value(self):
        """Only a wait that was told to expect a replaced server process may accept a new boot uuid."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        pin.unpin()

        assert pin.expected is None
        assert pin.needs_handshake() is True

    def test_an_unpinned_pin_adopts_the_next_boot_uuid(self):
        """The replacement process is the one this client drives from now on."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))
        pin.unpin()

        pin.verify(_response(boot_uuid="boot-b"))

        assert pin.expected == "boot-b"

    def test_an_unpinned_pin_is_strict_again_afterwards(self):
        """Fencing has to come back the moment readiness is established, or a silent restart passes."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.unpin()
        pin.verify(_response(boot_uuid="boot-b"))

        with pytest.raises(ServerRestartedError, match="boot-c"):
            pin.verify(_response(boot_uuid="boot-c"))

    def test_unpinning_an_optional_pin_changes_nothing(self):
        """A client that does not require a stable server has nothing to forget."""
        pin = BootUuidPin(required=False, worker_cls_name="Worker")

        pin.unpin()

        assert pin.expected is None
        assert pin.needs_handshake() is False


class TestRestoringThePin:
    def test_a_restored_pin_fences_against_the_process_it_was_pinned_to(self):
        """A readiness wait that never succeeded must not leave the fence open for the next ordinary call."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        pin.repin(pin.unpin())

        assert pin.expected == "boot-a"
        assert pin.needs_handshake() is False

    def test_unpinning_an_unpinned_pin_answers_nothing(self):
        """A handle that never handshook has nothing to restore, and must not invent a baseline."""
        assert BootUuidPin(required=True, worker_cls_name="Worker").unpin() is None
