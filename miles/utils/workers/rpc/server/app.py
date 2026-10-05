from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    BOOT_UUID_MISMATCH_STATUS,
    CALL_STATUS_PATH,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    EXPECTED_BOOT_UUID_HEADER,
    HEALTH_PATH,
    IN_FLIGHT_PATH,
    SUBMIT_PATH,
    CallStatusResponse,
    HealthResponse,
    InFlightResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.rpc.server.core import RpcServer

logger = logging.getLogger(__name__)


def create_rpc_app(worker: object) -> FastAPI:
    server = RpcServer(worker=worker)

    app = FastAPI()

    @app.middleware("http")
    async def boot_uuid_guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        expected = request.headers.get(EXPECTED_BOOT_UUID_HEADER)
        if expected is not None and expected != server.boot_uuid:
            log_structured(
                logger.warning,
                tag="rpc",
                op="server",
                phase="reject",
                reason="boot_uuid_mismatch",
                expected=expected,
                actual=server.boot_uuid,
            )
            response: Response = JSONResponse(
                status_code=BOOT_UUID_MISMATCH_STATUS,
                content={"detail": f"boot uuid mismatch: client expected {expected}, server is {server.boot_uuid}"},
            )
        else:
            try:
                response = await call_next(request)
            except Exception:
                log_structured(
                    logger.error, tag="rpc", op="server", phase="unhandled_error", path=request.url.path, exc_info=True
                )
                response = JSONResponse(status_code=500, content={"detail": "unhandled rpc server error"})

        response.headers[BOOT_UUID_HEADER] = server.boot_uuid
        return response

    @app.exception_handler(RequestValidationError)
    async def handle_malformed_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get(HEALTH_PATH)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get(IN_FLIGHT_PATH)
    async def in_flight_calls() -> InFlightResponse:
        return server.in_flight_calls()

    @app.post(SUBMIT_PATH)
    async def submit_call(method_name: str, request: SubmitRequest) -> SubmitResponse:
        return server.submit_call(method_name=method_name, request=request)

    @app.get(CALL_STATUS_PATH)
    async def query_call(
        call_id: str, timeout: float = Query(default=DEFAULT_POLL_TIMEOUT_SECONDS, ge=0.0)
    ) -> CallStatusResponse:
        return await server.query_call(call_id=call_id, timeout=timeout)

    return app
