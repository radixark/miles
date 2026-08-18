from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.call import RpcCall
from miles.utils.workers.rpc.client.misc import (
    RETRY_INITIAL_DELAY_SECONDS,
    RETRYABLE_ERRORS,
    WORKER_IS_GONE_ERRORS,
    BootUuidPin,
    RpcTransport,
    ServerRestartedError,
)
from miles.utils.workers.rpc.common.metadata import (
    RpcMethodSpec,
    canonicalize_method_arguments,
    collect_rpc_method_specs,
)
from miles.utils.workers.rpc.common.protocol import HEALTH_PATH, HealthResponse
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

DEFAULT_CALL_TIMEOUT_SECONDS = 3600.0
DEFAULT_READY_TIMEOUT_SECONDS = 600.0

_HEALTH_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)


class RpcWorkerHandle(BaseWorkerHandle):
    def __init__(
        self,
        worker_cls: type,
        *,
        server_url: str,
        require_stable_boot_uuid: bool = False,
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
        self._boot_uuid_pin = BootUuidPin(required=require_stable_boot_uuid, worker_cls_name=worker_cls.__name__)
        self._transport = RpcTransport(
            server_url=server_url, http_client=http_client, boot_uuid_pin=self._boot_uuid_pin
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._worker_cls_name})"

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_"):
            raise AttributeError(name)
        spec = self._specs.get(name)
        if spec is None:
            raise AttributeError(f"{self._worker_cls_name} has no rpc method {name!r}")

        async def call(*args: Any, **kwargs: Any) -> Any:
            return await self._perform_call(
                spec=spec, kwargs=canonicalize_method_arguments(spec=spec, args=args, kwargs=kwargs)
            )

        return call

    async def submit_without_result(self, method_name: str, /, **kwargs: Any) -> None:
        spec = self._specs.get(method_name)
        if spec is None:
            raise AttributeError(f"{self._worker_cls_name} has no rpc method {method_name!r}")

        call = await self._prepare_call(
            spec=spec, kwargs=canonicalize_method_arguments(spec=spec, args=(), kwargs=kwargs)
        )
        await call.submit()

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        pinned_before = self._boot_uuid_pin.unpin() if allow_server_uuid_change else None

        async def attempt(remaining: float) -> None:
            if allow_server_uuid_change:
                self._boot_uuid_pin.unpin()
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
            if allow_server_uuid_change:
                self._boot_uuid_pin.repin(pinned_before)
            raise WorkerUnreachableError(
                f"{self._worker_cls_name} rpc server not ready within {timeout}s: {e!r}"
            ) from e

        if allow_server_uuid_change:
            assert not self._boot_uuid_pin.needs_handshake(), "the readiness probe did not pin the answering process"

    async def probe_is_dead(self) -> bool:
        try:
            await self._transport.request(
                "GET", HEALTH_PATH, seconds=_HEALTH_TIMEOUT_SECONDS, response_model=HealthResponse
            )
        except (ServerRestartedError, *WORKER_IS_GONE_ERRORS):
            return True
        except RETRYABLE_ERRORS:
            return False
        return False

    async def _perform_call(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        call = await self._prepare_call(spec=spec, kwargs=kwargs)
        return await call.run()

    async def _prepare_call(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> RpcCall:
        call = RpcCall(
            spec=spec,
            kwargs=kwargs,
            worker_cls_name=self._worker_cls_name,
            transport=self._transport,
            call_timeout_seconds=self._call_timeout_seconds,
        )

        if self._boot_uuid_pin.needs_handshake():
            await self.wait_ready(timeout=self._ready_timeout_seconds)

        return call
