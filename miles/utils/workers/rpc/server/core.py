from __future__ import annotations

import functools
import logging
from typing import NoReturn

from fastapi import HTTPException
from pydantic import ValidationError

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.common.protocol import CallStatusResponse, SubmitRequest, SubmitResponse
from miles.utils.workers.rpc.server.executor import RpcCallExecutor
from miles.utils.workers.rpc.server.store import CallStore

logger = logging.getLogger(__name__)


class RpcServer:
    def __init__(self, *, worker: object) -> None:
        self._specs = collect_rpc_method_specs(type(worker))
        self._store = CallStore()
        self._executor = RpcCallExecutor(worker=worker)
        log_structured(
            logger.info,
            tag="rpc",
            op="server",
            phase="boot",
            worker=type(worker).__name__,
            methods=len(self._specs),
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

        self._store.begin(call_id=request.call_id)

        self._executor.start(
            spec=spec,
            kwargs=kwargs,
            call_id=request.call_id,
            finish=functools.partial(self._store.finish, call_id=request.call_id),
        )

        return SubmitResponse()

    async def query_call(self, *, call_id: str, timeout: float) -> CallStatusResponse:
        if not self._store.contains(call_id):
            log_structured(logger.warning, tag="rpc", op="poll", phase="reject", reason="unknown_call", call=call_id)
            raise HTTPException(status_code=404, detail=f"unknown call id {call_id!r}")

        outcome = await self._store.wait(call_id=call_id, timeout=timeout)
        if outcome is None:
            return CallStatusResponse(status="pending")
        return CallStatusResponse(status=outcome.status, result=outcome.result, error=outcome.error)
