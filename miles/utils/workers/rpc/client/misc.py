from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    BOOT_UUID_MISMATCH_STATUS,
    EXPECTED_BOOT_UUID_HEADER,
)

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)

RETRY_INITIAL_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 10.0

_LOWEST_SERVER_ERROR_STATUS = 500

_ABORT_SLACK_SECONDS = 1.0


class ServerRestartedError(Exception):
    pass


class RpcProtocolError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class RpcWorkerCallError(Exception):
    pass


class RetryableResponseError(Exception):
    pass


RETRYABLE_ERRORS = (httpx.TransportError, TimeoutError, asyncio.TimeoutError, RetryableResponseError)

NEVER_REACHED_SERVER_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

WORKER_IS_GONE_ERRORS = (httpx.ConnectError,)


class RpcTransport:
    def __init__(self, *, server_url: str, http_client: httpx.AsyncClient | None, boot_uuid_pin: BootUuidPin) -> None:
        self._server_url = server_url.rstrip("/")
        self._http_client_override = http_client
        self._boot_uuid_pin = boot_uuid_pin

    async def request(
        self, method: str, path: str, *, seconds: float, response_model: type[_ResponseT], **kwargs: Any
    ) -> _ResponseT:
        headers = dict(kwargs.pop("headers", {}))
        if self._boot_uuid_pin.expected is not None:
            headers[EXPECTED_BOOT_UUID_HEADER] = self._boot_uuid_pin.expected
        response = await asyncio.wait_for(
            self._client.request(
                method,
                f"{self._server_url}{path}",
                timeout=seconds,
                headers=headers,
                follow_redirects=False,
                **kwargs,
            ),
            timeout=seconds + _ABORT_SLACK_SECONDS,
        )

        if response.status_code == BOOT_UUID_MISMATCH_STATUS:
            raise ServerRestartedError(f"rpc server restarted: {response.text}")
        if response.status_code >= _LOWEST_SERVER_ERROR_STATUS:
            raise RetryableResponseError(f"{method} {path} returned {response.status_code}")
        if response.status_code != 200:
            raise RpcProtocolError(
                f"{method} {path} rejected ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        self._boot_uuid_pin.verify(response)
        return response_model.model_validate(response.json())

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http_client_override is not None:
            return self._http_client_override
        return GeneralHttpClientProvider.client()


class BootUuidPin:
    def __init__(self, *, required: bool, worker_cls_name: str) -> None:
        self._required = required
        self._worker_cls_name = worker_cls_name
        self._value: str | None = None

    @property
    def expected(self) -> str | None:
        return self._value if self._required else None

    def needs_handshake(self) -> bool:
        return self._required and self._value is None

    def verify(self, response: httpx.Response) -> None:
        if not self._required:
            return

        boot_uuid = response.headers.get(BOOT_UUID_HEADER)
        if boot_uuid is None:
            raise ServerRestartedError(f"{self._worker_cls_name} response is missing the {BOOT_UUID_HEADER} header")
        if self._value is None:
            self._value = boot_uuid
        elif boot_uuid != self._value:
            raise ServerRestartedError(
                f"{self._worker_cls_name} response came from boot uuid {boot_uuid}, expected {self._value}"
            )
