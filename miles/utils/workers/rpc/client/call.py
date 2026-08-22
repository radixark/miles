from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import httpx

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.client.misc import (
    NEVER_REACHED_SERVER_ERRORS,
    RETRY_INITIAL_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    RETRYABLE_ERRORS,
    RetryableResponseError,
    RpcTransport,
    RpcWorkerCallError,
)
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import (
    ACKNOWLEDGE_PATH,
    CALL_STATUS_PATH,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    SUBMIT_PATH,
    AcknowledgeRequest,
    AcknowledgeResponse,
    CallStatusResponse,
    SubmitRequest,
    SubmitResponse,
    compute_request_digest,
)
from miles.utils.workers.worker_handle import WorkerUnreachableError

logger = logging.getLogger(__name__)

SUBMIT_RETRY_WINDOW_SECONDS = 60.0
SUBMIT_ATTEMPT_TIMEOUT_SECONDS = 10.0
ACK_RETRY_WINDOW_SECONDS = 0.5
ACK_ATTEMPT_TIMEOUT_SECONDS = 0.1
ACK_RETRY_INITIAL_DELAY_SECONDS = 0.05
ACK_RETRY_MAX_DELAY_SECONDS = 0.1
POLL_SLACK_SECONDS = 5.0


class _CallStillPendingError(Exception):
    pass


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
        await self.submit()
        outcome, outcome_boot_uuid = await self._poll_until_done()
        elapsed = time.monotonic() - started_at

        ok = outcome.status != "failed"
        log_structured(logger.debug, op="call", phase="end", ok=ok, **self._log_fields, elapsed_s=round(elapsed, 3))
        if not ok:
            await self._acknowledge(expected_boot_uuid=outcome_boot_uuid)
            raise RpcWorkerCallError(f"{self._method_label} failed remotely:\n{outcome.error}")
        result = self._spec.serializer.decode_result(outcome.result)
        await self._acknowledge(expected_boot_uuid=outcome_boot_uuid)
        return result

    async def submit(self) -> None:
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

    async def _poll_until_done(self) -> tuple[CallStatusResponse, str]:
        try:
            return await retry_until_deadline(
                self._poll_once,
                total_seconds=self._call_timeout_seconds,
                retry_on=_CallStillPendingError,
                initial_delay=0.0,
                log_fields={**self._log_fields, "op": "poll"},
            )
        except _CallStillPendingError:
            log_structured(
                logger.warning,
                op="poll",
                phase="still_pending",
                **self._log_fields,
                timeout_s=self._call_timeout_seconds,
            )
            raise TimeoutError(
                f"{self._method_label} (call id {self._call_id}) still pending after {self._call_timeout_seconds}s"
            ) from None

    async def _poll_once(self, remaining: float) -> tuple[CallStatusResponse, str]:
        poll_seconds = min(DEFAULT_POLL_TIMEOUT_SECONDS, remaining)
        server_seconds = poll_seconds - min(POLL_SLACK_SECONDS, poll_seconds / 2)
        try:
            outcome, outcome_boot_uuid = await self._transport.request_with_boot_uuid(
                "GET",
                CALL_STATUS_PATH.format(call_id=self._call_id),
                seconds=poll_seconds,
                response_model=CallStatusResponse,
                params={"timeout": server_seconds},
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise _CallStillPendingError(f"long poll timed out after {poll_seconds:.1f}s") from e
        except (httpx.TransportError, RetryableResponseError) as e:
            await asyncio.sleep(RETRY_INITIAL_DELAY_SECONDS)
            raise _CallStillPendingError(f"poll attempt failed: {e!r}") from e

        if outcome.status == "pending":
            raise _CallStillPendingError("call still pending")
        if outcome_boot_uuid is None:
            raise WorkerUnreachableError(f"{self._method_label} terminal response is missing its server boot uuid")
        return outcome, outcome_boot_uuid

    async def _acknowledge(self, *, expected_boot_uuid: str) -> None:
        request = AcknowledgeRequest(
            request_digest=compute_request_digest(method_name=self._spec.name, query=self._query).hex()
        )
        task = asyncio.create_task(
            self._acknowledge_with_retries(request=request, expected_boot_uuid=expected_boot_uuid)
        )
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                log_structured(logger.warning, op="ack", phase="cancelled", **self._log_fields)
                if task.done():
                    return

    async def _acknowledge_with_retries(self, *, request: AcknowledgeRequest, expected_boot_uuid: str) -> None:
        try:
            await retry_until_deadline(
                lambda remaining: self._transport.request(
                    "POST",
                    ACKNOWLEDGE_PATH.format(call_id=self._call_id),
                    seconds=min(ACK_ATTEMPT_TIMEOUT_SECONDS, remaining),
                    response_model=AcknowledgeResponse,
                    expected_boot_uuid=expected_boot_uuid,
                    json=request.model_dump(),
                ),
                total_seconds=ACK_RETRY_WINDOW_SECONDS,
                attempt_seconds=ACK_ATTEMPT_TIMEOUT_SECONDS,
                retry_on=RETRYABLE_ERRORS,
                initial_delay=ACK_RETRY_INITIAL_DELAY_SECONDS,
                max_delay=ACK_RETRY_MAX_DELAY_SECONDS,
                log_fields={**self._log_fields, "op": "ack"},
            )
        except Exception:
            log_structured(logger.warning, op="ack", phase="failed", **self._log_fields, exc_info=True)

    @property
    def _log_fields(self) -> dict[str, Any]:
        return {"tag": "rpc", "worker": self._worker_cls_name, "method": self._spec.name, "call": self._call_id}

    @property
    def _method_label(self) -> str:
        return f"{self._worker_cls_name}.{self._spec.name}"
