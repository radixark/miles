from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from miles.utils.http_utils import GeneralHttpClientProvider

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)

RETRY_INITIAL_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 10.0

_LOWEST_SERVER_ERROR_STATUS = 500

_ABORT_SLACK_SECONDS = 1.0


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


class RpcTransport:
    def __init__(self, *, server_url: str, http_client: httpx.AsyncClient | None) -> None:
        self._server_url = server_url.rstrip("/")
        self._http_client_override = http_client

    async def request(
        self, method: str, path: str, *, seconds: float, response_model: type[_ResponseT], **kwargs: Any
    ) -> _ResponseT:
        response = await asyncio.wait_for(
            self._client.request(
                method, f"{self._server_url}{path}", timeout=seconds, follow_redirects=False, **kwargs
            ),
            timeout=seconds + _ABORT_SLACK_SECONDS,
        )

        if response.status_code >= _LOWEST_SERVER_ERROR_STATUS:
            raise RetryableResponseError(f"{method} {path} returned {response.status_code}")
        if response.status_code != 200:
            raise RpcProtocolError(
                f"{method} {path} rejected ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )
        return response_model.model_validate(response.json())

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http_client_override is not None:
            return self._http_client_override
        return GeneralHttpClientProvider.client()
