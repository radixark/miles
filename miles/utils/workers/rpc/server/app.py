from __future__ import annotations

import logging

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from miles.utils.workers.rpc.common.protocol import (
    CALL_STATUS_PATH,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    HEALTH_PATH,
    SUBMIT_PATH,
    CallStatusResponse,
    HealthResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.rpc.server.core import RpcServer

logger = logging.getLogger(__name__)


def create_rpc_app(worker: object) -> FastAPI:
    server = RpcServer(worker=worker)

    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def handle_malformed_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get(HEALTH_PATH)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post(SUBMIT_PATH)
    async def submit_call(method_name: str, request: SubmitRequest) -> SubmitResponse:
        return server.submit_call(method_name=method_name, request=request)

    @app.get(CALL_STATUS_PATH)
    async def query_call(
        call_id: str, timeout: float = Query(default=DEFAULT_POLL_TIMEOUT_SECONDS, ge=0.0)
    ) -> CallStatusResponse:
        return await server.query_call(call_id=call_id, timeout=timeout)

    return app
