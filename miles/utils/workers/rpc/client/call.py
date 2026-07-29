from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.client.misc import (
    NEVER_REACHED_SERVER_ERRORS,
    RETRY_INITIAL_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    RETRYABLE_ERRORS,
    RpcTransport,
    RpcWorkerCallError,
)
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import (
    CALL_STATUS_PATH,
    SUBMIT_PATH,
    CallStatusResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.worker_handle import WorkerUnreachableError

logger = logging.getLogger(__name__)

SUBMIT_RETRY_WINDOW_SECONDS = 60.0
SUBMIT_ATTEMPT_TIMEOUT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.05


class RpcCall:
    def __init__(
        self,
        *,
        spec: RpcMethodSpec,
        kwargs: dict[str, Any],
        worker_cls_name: str,
        transport: RpcTransport,
        call_timeout_seconds: float,
    ) -> None:
        self._spec = spec
        self._kwargs = kwargs
        self._query = spec.serializer.encode_query(kwargs)
        self._worker_cls_name = worker_cls_name
        self._transport = transport
        self._call_timeout_seconds = call_timeout_seconds
        self._call_id = uuid.uuid4().hex

    async def run(self) -> Any:
        log_structured(logger.debug, op="call", phase="start", **self._log_fields, args=sorted(self._kwargs))

        started_at = time.monotonic()
        await self._submit()
        outcome = await self._poll_until_done()
        elapsed = time.monotonic() - started_at

        ok = outcome.status != "failed"
        log_structured(logger.debug, op="call", phase="end", ok=ok, **self._log_fields, elapsed_s=round(elapsed, 3))
        if not ok:
            raise RpcWorkerCallError(f"{self._method_label} failed remotely:\n{outcome.error}")
        return self._spec.serializer.decode_result(outcome.result)

    async def _submit(self) -> None:
        request = SubmitRequest(call_id=self._call_id, query=self._query)

        try:
            await retry_until_deadline(
                lambda remaining: self._submit_attempt(request=request, remaining=remaining),
                total_seconds=SUBMIT_RETRY_WINDOW_SECONDS,
                retry_on=NEVER_REACHED_SERVER_ERRORS,
                initial_delay=RETRY_INITIAL_DELAY_SECONDS,
                max_delay=RETRY_MAX_DELAY_SECONDS,
                log_fields={**self._log_fields, "op": "submit"},
            )
        except RETRYABLE_ERRORS as e:
            log_structured(logger.warning, op="submit", phase="gave_up", **self._log_fields, error=repr(e))
            raise WorkerUnreachableError(f"{self._method_label} submit failed: {e!r}") from e

    async def _submit_attempt(self, *, request: SubmitRequest, remaining: float) -> None:
        await self._transport.request(
            "POST",
            SUBMIT_PATH.format(method_name=self._spec.name),
            seconds=min(SUBMIT_ATTEMPT_TIMEOUT_SECONDS, remaining),
            response_model=SubmitResponse,
            json=request.model_dump(exclude_none=True),
        )
        log_structured(logger.debug, op="submit", phase="accepted", **self._log_fields)

    async def _poll_until_done(self) -> CallStatusResponse:
        expires_at = time.monotonic() + self._call_timeout_seconds

        while time.monotonic() < expires_at:
            outcome = await self._transport.request(
                "GET",
                CALL_STATUS_PATH.format(call_id=self._call_id),
                seconds=max(0.0, expires_at - time.monotonic()),
                response_model=CallStatusResponse,
            )
            if outcome.status != "pending":
                return outcome
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        log_structured(
            logger.warning,
            op="poll",
            phase="still_pending",
            **self._log_fields,
            timeout_s=self._call_timeout_seconds,
        )
        raise TimeoutError(
            f"{self._method_label} (call id {self._call_id}) still pending after {self._call_timeout_seconds}s"
        )

    @property
    def _log_fields(self) -> dict[str, Any]:
        return {"tag": "rpc", "worker": self._worker_cls_name, "method": self._spec.name, "call": self._call_id}

    @property
    def _method_label(self) -> str:
        return f"{self._worker_cls_name}.{self._spec.name}"
