from __future__ import annotations

import functools
import logging
import uuid
from typing import NoReturn

from fastapi import HTTPException
from pydantic import ValidationError

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.common.protocol import (
    MAX_POLL_TIMEOUT_SECONDS,
    CallStatusResponse,
    InFlightResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.rpc.server.executor import RpcCallExecutor
from miles.utils.workers.rpc.server.store import CallStore, DuplicateCallError

logger = logging.getLogger(__name__)


class RpcServer:
    def __init__(self, *, worker: object) -> None:
        self.boot_uuid = uuid.uuid4().hex
        self._specs = collect_rpc_method_specs(type(worker))
        self._store = CallStore()
        self._executor = RpcCallExecutor(worker=worker, specs=self._specs)
        log_structured(
            logger.info,
            tag="rpc",
            op="server",
            phase="boot",
            worker=type(worker).__name__,
            boot_uuid=self.boot_uuid,
            methods=len(self._specs),
            groups=self._executor.concurrency_groups,
        )

    def submit_call(self, *, method_name: str, request: SubmitRequest) -> SubmitResponse:
        def reject(*, status_code: int, reason: str, detail: str) -> NoReturn:
            log_structured(
                logger.warning,
                tag="rpc",
                op="submit",
                phase="reject",
                reason=reason,
                method=method_name,
                call=request.call_id,
                error=detail,
            )
            raise HTTPException(status_code=status_code, detail=detail)

        spec = self._specs.get(method_name)
        if spec is None:
            reject(status_code=404, reason="unknown_method", detail=f"unknown rpc method {method_name!r}")

        try:
            kwargs = spec.serializer.decode_query(request.query)
        except ValidationError as e:
            reject(status_code=400, reason="invalid_query", detail=str(e))

        try:
            self._store.begin(call_id=request.call_id)
        except DuplicateCallError as e:
            reject(status_code=409, reason="duplicate_call", detail=str(e))

        self._executor.start(
            spec=spec,
            kwargs=kwargs,
            call_id=request.call_id,
            finish=functools.partial(self._store.finish, call_id=request.call_id),
        )

        return SubmitResponse()

    def in_flight_calls(self) -> InFlightResponse:
        return InFlightResponse(call_ids=self._store.in_flight_call_ids())

    async def query_call(self, *, call_id: str, timeout: float) -> CallStatusResponse:
        if not self._store.contains(call_id):
            log_structured(logger.warning, tag="rpc", op="poll", phase="reject", reason="unknown_call", call=call_id)
            raise HTTPException(status_code=404, detail=f"unknown call id {call_id!r}")

        outcome = await self._store.wait(call_id=call_id, timeout=min(timeout, MAX_POLL_TIMEOUT_SECONDS))
        if outcome is None:
            return CallStatusResponse(status="pending")
        return CallStatusResponse(status=outcome.status, result=outcome.result, error=outcome.error)
