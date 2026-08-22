from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.protocol import (
    ACKNOWLEDGE_PATH,
    BOOT_UUID_HEADER,
    BOOT_UUID_MISMATCH_STATUS,
    CALL_STATUS_PATH,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    EXPECTED_BOOT_UUID_HEADER,
    HEALTH_PATH,
    IN_FLIGHT_PATH,
    MAX_AGGREGATE_REQUEST_BODY_BYTES,
    MAX_CONTROL_AGGREGATE_REQUEST_BODY_BYTES,
    SUBMIT_PATH,
    AcknowledgeRequest,
    AcknowledgeResponse,
    CallStatusResponse,
    HealthResponse,
    InFlightResponse,
    SubmitRequest,
    SubmitResponse,
    is_rpc_control_request,
)
from miles.utils.workers.rpc.server.core import RpcServer

logger = logging.getLogger(__name__)

MAX_DATA_IN_FLIGHT_REQUESTS = 4096
MAX_CONTROL_IN_FLIGHT_REQUESTS = 4096
MAX_DATA_IN_FLIGHT_REJECTIONS = 256
MAX_CONTROL_IN_FLIGHT_REJECTIONS = 256


class _RequestBodyLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        boot_uuid: str,
        max_data_aggregate_bytes: int = MAX_AGGREGATE_REQUEST_BODY_BYTES,
        max_control_aggregate_bytes: int = MAX_CONTROL_AGGREGATE_REQUEST_BODY_BYTES,
        max_data_in_flight_requests: int = MAX_DATA_IN_FLIGHT_REQUESTS,
        max_control_in_flight_requests: int = MAX_CONTROL_IN_FLIGHT_REQUESTS,
        max_data_in_flight_rejections: int = MAX_DATA_IN_FLIGHT_REJECTIONS,
        max_control_in_flight_rejections: int = MAX_CONTROL_IN_FLIGHT_REJECTIONS,
        control_paths: frozenset[str] = frozenset(),
    ) -> None:
        self._app = app
        self._boot_uuid = boot_uuid
        self._max_data_aggregate_bytes = max_data_aggregate_bytes
        self._max_control_aggregate_bytes = max_control_aggregate_bytes
        self._max_data_in_flight_requests = max_data_in_flight_requests
        self._max_control_in_flight_requests = max_control_in_flight_requests
        self._max_data_in_flight_rejections = max_data_in_flight_rejections
        self._max_control_in_flight_rejections = max_control_in_flight_rejections
        self._control_paths = control_paths
        self._data_aggregate_bytes = 0
        self._control_aggregate_bytes = 0
        self._data_in_flight_requests = 0
        self._control_in_flight_requests = 0
        self._data_in_flight_rejections = 0
        self._control_in_flight_rejections = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        control_plane = is_rpc_control_request(
            method=scope.get("method", ""),
            path=path,
            dynamic_paths=self._control_paths,
        )
        if not self._reserve_request(control_plane=control_plane):
            if not self._reserve_rejection(control_plane=control_plane):
                return
            try:
                await self._reject(
                    scope=scope,
                    receive=receive,
                    send=send,
                    status_code=503,
                    detail="rpc request ingress capacity is full",
                )
            finally:
                self._release_rejection(control_plane=control_plane)
            return

        reserved_bytes = 0
        try:
            body = bytearray()
            body_bytes = 0
            more_body = True
            while more_body:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    del message
                    continue
                chunk = message.get("body", b"")
                chunk_bytes = len(chunk)
                if chunk_bytes and not self._reserve(control_plane=control_plane, body_bytes=chunk_bytes):
                    del message, chunk
                    await self._reject(
                        scope=scope,
                        receive=receive,
                        send=send,
                        status_code=503,
                        detail="rpc request ingress capacity is full",
                    )
                    return
                if chunk_bytes:
                    reserved_bytes += chunk_bytes
                    body.extend(chunk)
                body_bytes += chunk_bytes
                more_body = message.get("more_body", False)
                del message, chunk

            if body:
                if not self._reserve(control_plane=control_plane, body_bytes=body_bytes):
                    await self._reject(
                        scope=scope,
                        receive=receive,
                        send=send,
                        status_code=503,
                        detail="rpc request ingress capacity is full",
                    )
                    return
                try:
                    replay_body = bytes(body)
                finally:
                    del body
                    self._release(control_plane=control_plane, body_bytes=body_bytes)
            else:
                replay_body = b""
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if delivered:
                    return {"type": "http.disconnect"}
                delivered = True
                return {"type": "http.request", "body": replay_body, "more_body": False}

            await self._app(scope, replay, send)
        finally:
            self._release(control_plane=control_plane, body_bytes=reserved_bytes)
            self._release_request(control_plane=control_plane)

    def _reserve(self, *, control_plane: bool, body_bytes: int) -> bool:
        aggregate_bytes = self._control_aggregate_bytes if control_plane else self._data_aggregate_bytes
        max_aggregate_bytes = self._max_control_aggregate_bytes if control_plane else self._max_data_aggregate_bytes
        if aggregate_bytes + body_bytes > max_aggregate_bytes:
            return False
        if control_plane:
            self._control_aggregate_bytes += body_bytes
        else:
            self._data_aggregate_bytes += body_bytes
        return True

    def _reserve_request(self, *, control_plane: bool) -> bool:
        in_flight = self._control_in_flight_requests if control_plane else self._data_in_flight_requests
        maximum = self._max_control_in_flight_requests if control_plane else self._max_data_in_flight_requests
        if in_flight >= maximum:
            return False
        if control_plane:
            self._control_in_flight_requests += 1
        else:
            self._data_in_flight_requests += 1
        return True

    def _release_request(self, *, control_plane: bool) -> None:
        if control_plane:
            self._control_in_flight_requests -= 1
        else:
            self._data_in_flight_requests -= 1

    def _reserve_rejection(self, *, control_plane: bool) -> bool:
        in_flight = self._control_in_flight_rejections if control_plane else self._data_in_flight_rejections
        maximum = self._max_control_in_flight_rejections if control_plane else self._max_data_in_flight_rejections
        if in_flight >= maximum:
            return False
        if control_plane:
            self._control_in_flight_rejections += 1
        else:
            self._data_in_flight_rejections += 1
        return True

    def _release_rejection(self, *, control_plane: bool) -> None:
        if control_plane:
            self._control_in_flight_rejections -= 1
        else:
            self._data_in_flight_rejections -= 1

    def _release(self, *, control_plane: bool, body_bytes: int) -> None:
        if control_plane:
            self._control_aggregate_bytes -= body_bytes
        else:
            self._data_aggregate_bytes -= body_bytes

    async def _reject(
        self,
        *,
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        detail: str,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail},
            headers={BOOT_UUID_HEADER: self._boot_uuid},
        )
        await response(scope, receive, send)


def create_rpc_app(worker: object) -> FastAPI:
    server = RpcServer(worker=worker)

    app = FastAPI()
    app.state.rpc_server = server
    app.state.rpc_control_paths = server.control_paths
    app.add_middleware(
        _RequestBodyLimitMiddleware,
        boot_uuid=server.boot_uuid,
        control_paths=server.control_paths,
    )

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

    @app.post(ACKNOWLEDGE_PATH)
    async def acknowledge_call(call_id: str, request: AcknowledgeRequest) -> AcknowledgeResponse:
        return server.acknowledge_call(call_id=call_id, request=request)

    return app
