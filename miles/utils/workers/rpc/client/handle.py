from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.call import RpcCall
from miles.utils.workers.rpc.client.misc import RETRY_INITIAL_DELAY_SECONDS, RETRYABLE_ERRORS, RpcTransport
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec, collect_rpc_method_specs
from miles.utils.workers.rpc.common.protocol import HEALTH_PATH, HealthResponse
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

DEFAULT_CALL_TIMEOUT_SECONDS = 3600.0
DEFAULT_READY_TIMEOUT_SECONDS = 600.0

_HEALTH_TIMEOUT_SECONDS = 5.0


class RpcWorkerHandle(BaseWorkerHandle):
    def __init__(
        self,
        worker_cls: type,
        *,
        server_url: str,
        call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._specs = collect_rpc_method_specs(worker_cls)
        shadowed = sorted(name for name in self._specs if hasattr(type(self), name))
        if shadowed:
            raise TypeError(f"{worker_cls.__name__} rpc methods shadow handle attributes: {shadowed}")

        self._worker_cls_name = worker_cls.__name__
        self._call_timeout_seconds = call_timeout_seconds
        self._ready_timeout_seconds = ready_timeout_seconds
        self._transport = RpcTransport(server_url=server_url, http_client=http_client)

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_"):
            raise AttributeError(name)
        spec = self._specs.get(name)
        if spec is None:
            raise AttributeError(f"{self._worker_cls_name} has no rpc method {name!r}")

        async def call(**kwargs: Any) -> Any:
            return await self._perform_call(spec=spec, kwargs=kwargs)

        return call

    async def wait_ready(self, *, timeout: float) -> None:
        async def attempt(remaining: float) -> None:
            await self._transport.request(
                "GET", HEALTH_PATH, seconds=min(_HEALTH_TIMEOUT_SECONDS, remaining), response_model=HealthResponse
            )

        try:
            await retry_until_deadline(
                attempt,
                total_seconds=timeout,
                retry_on=RETRYABLE_ERRORS,
                initial_delay=RETRY_INITIAL_DELAY_SECONDS,
                backoff_factor=1.0,
            )
        except RETRYABLE_ERRORS as e:
            raise WorkerUnreachableError(
                f"{self._worker_cls_name} rpc server not ready within {timeout}s: {e!r}"
            ) from e

    async def _perform_call(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        call = RpcCall(
            spec=spec,
            kwargs=kwargs,
            worker_cls_name=self._worker_cls_name,
            transport=self._transport,
            call_timeout_seconds=self._call_timeout_seconds,
        )

        return await call.run()
