import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from miles.utils.workers.rpc.client.misc import RpcProtocolError, RpcTransport
from miles.utils.workers.rpc.common.protocol import HealthResponse


def _transport_over(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
) -> tuple[RpcTransport, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = RpcTransport(server_url="http://testserver", http_client=client)
    return transport, client


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

    @pytest.mark.parametrize("status_code", [201, 400, 404, 409, 499, 500, 503])
    async def test_non_200_raises_protocol_error(self, status_code: int) -> None:
        """Any non-200 response is a protocol error carrying its status."""

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
