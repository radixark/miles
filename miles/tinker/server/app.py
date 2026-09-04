"""HTTP skin over TinkerService: the endpoints the Tinker SDK calls.

All wire translation happens here (encoding.py for JSON, proto_codec.py for
protobuf); the service only ever sees decoded commands and returns internal
results. Auth is a bearer API key used as the tenant identity; authorization
(ownership of models, promises, checkpoints) is enforced in the service.
"""

import logging

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from miles.tinker.core.promise import FAILED, PENDING
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.server.encoding import decode_command, decode_sample_request, render_result
from miles.tinker.server.proto_codec import (
    PROTO_CONTENT_TYPE,
    PROTO_ENCODERS,
    decode_forward_backward_request,
    maybe_decompress,
)

logger = logging.getLogger(__name__)

COMMAND_ROUTES = {
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_state",
    "/api/v1/load_weights": "load_state",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
}


def _tenant(authorization: str | None) -> str:
    if not authorization:
        return "anonymous"
    return authorization.removeprefix("Bearer ").strip()


def build_app(service: TinkerService) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(UserInputError)
    async def _user_error(request: Request, error: UserInputError):
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.exception_handler(OwnershipError)
    async def _ownership_error(request: Request, error: OwnershipError):
        return JSONResponse(status_code=403, content={"error": str(error)})

    @app.get("/api/v1/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    @app.get("/api/v1/client/config")
    async def client_config():
        return {}

    @app.post("/api/v1/client/dynamic_config")
    @app.get("/api/v1/client/dynamic_config")
    async def client_dynamic_config():
        return {}

    @app.post("/api/v1/telemetry")
    async def telemetry():
        return {"status": "accepted"}

    @app.post("/api/v1/create_session")
    async def create_session(request: Request, authorization: str | None = Header(default=None)):
        payload = await request.json()
        session_id = service.create_session(_tenant(authorization), payload)
        return {"type": "create_session", "session_id": session_id}

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(request: Request):
        payload = await request.json()
        service.heartbeat(payload["session_id"])
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/get_server_capabilities")
    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities():
        return {"supported_models": [{"model_name": service.config.base_model, "trainable": True, "sampleable": True}]}

    @app.post("/api/v1/create_model")
    async def create_model(request: Request, authorization: str | None = Header(default=None)):
        payload = await request.json()
        request_id, model_id = service.create_model(_tenant(authorization), payload)
        return {"request_id": request_id, "model_id": model_id}

    @app.post("/api/v1/get_info")
    async def get_info(request: Request, authorization: str | None = Header(default=None)):
        payload = await request.json()
        record = service.get_model(_tenant(authorization), payload["model_id"])
        return {
            "type": "get_info",
            "model_id": record.model_id,
            "model_data": {"model_name": record.base_model, "lora_rank": record.lora_rank},
        }

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request, authorization: str | None = Header(default=None)):
        if PROTO_CONTENT_TYPE in request.headers.get("content-type", ""):
            body = maybe_decompress(await request.body(), request.headers.get("content-encoding"))
            kind, payload = decode_forward_backward_request(body)
        else:
            kind, payload = decode_command("forward_backward", await request.json())
        request_id = service.submit(_tenant(authorization), kind, payload)
        return {"request_id": request_id, "model_id": payload["model_id"]}

    for route, route_kind in COMMAND_ROUTES.items():

        def _make(route_kind: str):
            async def command(request: Request, authorization: str | None = Header(default=None)):
                kind, payload = decode_command(route_kind, await request.json())
                request_id = service.submit(_tenant(authorization), kind, payload)
                return {"request_id": request_id, "model_id": payload["model_id"]}

            return command

        app.post(route)(_make(route_kind))

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request, authorization: str | None = Header(default=None)):
        payload = await request.json()
        promise = service.retrieve(_tenant(authorization), payload["request_id"])
        if promise is None:
            return JSONResponse(status_code=410, content={"error": "unknown or expired promise"})
        if promise.state == PENDING:
            return {"type": "try_again", "queue_state": "active"}
        if promise.state == FAILED:
            return {"error": promise.error, "category": promise.error_category}
        encoder = PROTO_ENCODERS.get(promise.result["kind"])
        if encoder is not None and PROTO_CONTENT_TYPE in request.headers.get("accept", ""):
            return Response(content=encoder(promise.result), media_type=PROTO_CONTENT_TYPE)
        return render_result(promise.result)

    @app.post("/api/v1/cancel_future")
    async def cancel_future(request: Request, authorization: str | None = Header(default=None)):
        payload = await request.json()
        service.cancel(_tenant(authorization), payload["request_id"])
        return {"status": "ok"}

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: Request, authorization: str | None = Header(default=None)):
        payload = await request.json()
        sampling_session_id = service.create_sampling_session(_tenant(authorization), payload)
        return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}

    @app.post("/api/v1/asample")
    async def asample(request: Request, authorization: str | None = Header(default=None)):
        payload = decode_sample_request(await request.json())
        request_id, sequence_ids = service.submit_sample(_tenant(authorization), payload)
        return {"request_id": request_id, "sample_sequence_ids": sequence_ids}

    return app
