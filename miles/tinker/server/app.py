"""Tinker SDK endpoints with JSON/protobuf translation and tenant authentication."""

import asyncio
import json
import os
from contextlib import suppress
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from miles.tinker.core.future import FAILED, PENDING
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import NotFoundError, OwnershipError, UserInputError
from miles.tinker.server.encoding import (
    decode_command,
    decode_sample_request,
    render_result,
    validate_create_model,
    validate_create_sampling_session,
    validate_sdk_version,
)
from miles.tinker.server.proto_codec import (
    PROTO_CONTENT_TYPE,
    PROTO_ENCODERS,
    decode_forward_backward_request,
    maybe_decompress,
)

RETRIEVE_LONG_POLL_S = 30.0

COMMAND_ROUTES = {
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_state",
    "/api/v1/load_weights": "load_state",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
    "/api/v1/save_weights_external": "save_weights_external",
}
_DOWNLOAD_CHUNK_BYTES = 1 << 20


def _open_with_size(path: str):
    handle = open(path, "rb")
    return handle, os.fstat(handle.fileno()).st_size


def _tenant(request: Request) -> str:
    """Require X-API-Key or a bearer token to establish the tenant identity."""
    key = (
        request.headers.get("x-api-key")
        or (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
    )
    if not key:
        raise UserInputError("missing API key: send it in the X-API-Key header")
    return key


class ProfileTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    target: Literal["trainer", "inference"]


class StartProfileRequest(ProfileTarget):
    skip_first: int = Field(default=0, ge=0)
    wait: int = Field(default=0, ge=0)
    warmup: int = Field(default=0, ge=0)
    active: int = Field(default=1, ge=1)
    repeat: int = Field(default=1, ge=0)
    max_duration_sec: int = Field(default=7200, ge=1, le=7200)
    activities: list[Literal["CPU", "CUDA"]] = Field(default_factory=lambda: ["CPU", "CUDA"], min_length=1)
    ranks: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)
    record_shapes: bool = False
    with_stack: bool = False
    profile_memory: bool = False
    profile_modules: bool = False


def build_app(service: TinkerService) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(UserInputError)
    async def _user_error(request: Request, error: UserInputError):
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.exception_handler(OwnershipError)
    async def _ownership_error(request: Request, error: OwnershipError):
        return JSONResponse(status_code=403, content={"error": str(error)})

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, error: NotFoundError):
        return JSONResponse(status_code=404, content={"error": str(error)})

    @app.post("/api/v1/start_profiling")
    async def start_profiling(request: Request, payload: StartProfileRequest):
        options = payload.model_dump(exclude={"target", "model_id"})
        if payload.profile_modules and (not payload.profile_memory or payload.with_stack):
            raise UserInputError("Module tracking requires profile_memory=true and with_stack=false")
        if payload.target == "inference":
            if payload.max_duration_sec != 7200 or (
                payload.skip_first,
                payload.wait,
                payload.warmup,
                payload.repeat,
            ) != (0, 0, 0, 1):
                raise UserInputError(
                    "SGLang supports active steps only; schedules and duration limits are trainer-only"
                )
            options["num_steps"] = options.pop("active")
            for key in ("skip_first", "wait", "warmup", "repeat", "max_duration_sec"):
                options.pop(key)
            if payload.profile_modules or payload.ranks:
                raise UserInputError("Inference does not support module tracking or rank selection")
            if payload.profile_memory:
                options["activities"].append("MEM")
            options = {
                key: value
                for key, value in options.items()
                if key not in {"ranks", "profile_memory", "profile_modules"}
            }
        return await service.start_profile(_tenant(request), payload.model_id, payload.target, options)

    @app.post("/api/v1/stop_profiling")
    async def stop_profiling(request: Request, payload: ProfileTarget):
        return await service.stop_profile(_tenant(request), payload.model_id, payload.target)

    @app.get("/api/v1/profiling_status")
    async def profiling_status(request: Request, model_id: str, target: Literal["trainer", "inference"]):
        return await service.profiling_status(_tenant(request), model_id, target)

    @app.get("/api/v1/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    async def client_config(request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            raise UserInputError("client/config needs a JSON body carrying sdk_version") from None
        validate_sdk_version(payload)
        return {}

    @app.post("/api/v1/client/dynamic_config")
    @app.get("/api/v1/client/dynamic_config")
    async def client_dynamic_config():
        return {}

    @app.post("/api/v1/telemetry")
    async def telemetry():
        return {"status": "accepted"}

    @app.post("/api/v1/create_session")
    async def create_session(request: Request):
        session_id = service.create_session(_tenant(request))
        return {"type": "create_session", "session_id": session_id}

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(request: Request):
        tenant = _tenant(request)
        payload = await request.json()
        if not service.heartbeat(tenant, payload["session_id"]):
            return JSONResponse(status_code=410, content={"error": "unknown or expired session"})
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/sessions/{session_id}/finish")
    async def finish_session(session_id: str, request: Request):
        payload = await request.json()
        reason = payload["reason"]["type"]
        if reason not in ("success", "errored", "interrupted"):
            raise UserInputError(f"unknown session finish reason {reason!r}")
        if not await service.finish_session(_tenant(request), session_id, reason, payload["detail"]):
            return JSONResponse(status_code=410, content={"error": "session lease expired before it finished"})
        return Response(status_code=204)

    @app.post("/api/v1/get_server_capabilities")
    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities():
        return {"supported_models": [{"model_name": service.config.base_model, "trainable": True, "sampleable": True}]}

    @app.post("/api/v1/create_model")
    async def create_model(request: Request):
        payload = await request.json()
        validate_create_model(payload)
        request_id, model_id = service.create_model(_tenant(request), payload)
        return {"request_id": request_id, "model_id": model_id}

    @app.post("/api/v1/get_info")
    async def get_info(request: Request):
        payload = await request.json()
        record = service.get_model(_tenant(request), payload["model_id"])
        return {
            "type": "get_info",
            "model_id": record.model_id,
            "model_name": record.base_model,
            "is_lora": True,
            "lora_rank": record.lora_rank,
            "model_data": {"model_name": record.base_model},
        }

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request):
        if PROTO_CONTENT_TYPE in request.headers.get("content-type", ""):
            body = maybe_decompress(await request.body(), request.headers.get("content-encoding"))
            op, payload = decode_forward_backward_request(body)
        else:
            op, payload = decode_command("forward_backward", await request.json())
        request_id = service.submit(_tenant(request), op, payload)
        return {"request_id": request_id, "model_id": payload["model_id"]}

    for route, route_op in COMMAND_ROUTES.items():

        def _command_handler(route_op: str):
            async def command(request: Request):
                op, payload = decode_command(route_op, await request.json())
                request_id = service.submit(_tenant(request), op, payload)
                return {"request_id": request_id, "model_id": payload["model_id"]}

            return command

        app.post(route)(_command_handler(route_op))

    @app.post("/api/v1/weights_info")
    async def weights_info(request: Request):
        payload = await request.json()
        return service.weights_info(_tenant(request), payload["tinker_path"])

    # checkpoint_id is "<kind>/<name>"
    @app.get("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/external_weights_urls")
    async def external_weights_urls(model_id: str, checkpoint_id: str, request: Request):
        public_url = service.config.public_url
        base_url = public_url.rstrip("/") + "/" if public_url is not None else str(request.base_url)
        return await service.external_weights_urls(_tenant(request), model_id, checkpoint_id, base_url)

    @app.put("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/ttl")
    async def set_checkpoint_ttl(model_id: str, checkpoint_id: str, request: Request):
        payload = await request.json()
        await service.set_checkpoint_ttl(_tenant(request), model_id, checkpoint_id, payload["ttl_seconds"])
        return None

    @app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}")
    async def delete_checkpoint(model_id: str, checkpoint_id: str, request: Request):
        await service.delete_checkpoint(_tenant(request), model_id, checkpoint_id)
        return None

    @app.get("/api/v1/external_weights/{token}/{filename}")
    async def download_external_weights(token: str, filename: str):
        """The URL is the credential (the SDK downloads without one). The file is opened before the headers go
        out, so a concurrent delete yields the complete file or a 404, never a truncated one."""
        path = await service.download_path(token, filename)
        try:
            # shared storage: keep the event loop serving
            handle, size = await asyncio.to_thread(_open_with_size, path)
        except FileNotFoundError:
            raise NotFoundError(f"{filename} was deleted") from None

        def chunks():
            with handle:
                while chunk := handle.read(_DOWNLOAD_CHUNK_BYTES):
                    yield chunk

        return StreamingResponse(
            chunks(), media_type="application/octet-stream", headers={"Content-Length": str(size)}
        )

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request):
        payload = await request.json()
        future = service.retrieve_future(_tenant(request), payload["request_id"])
        if future is None:
            return JSONResponse(status_code=410, content={"error": "unknown or expired request"})
        if future.state == PENDING:
            # long-poll: the SDK retries try_again with no backoff, so answer on
            # settlement instead of turning every pending future into a busy-poll
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(future.settled.wait(), timeout=RETRIEVE_LONG_POLL_S)
        if future.state == PENDING:
            return {"type": "try_again", "queue_state": "active"}
        if future.state == FAILED:
            return {"error": future.error, "category": future.error_category}
        encoder = PROTO_ENCODERS.get(future.result["op"])
        if encoder is not None and PROTO_CONTENT_TYPE in request.headers.get("accept", ""):
            return Response(content=encoder(future.result), media_type=PROTO_CONTENT_TYPE)
        return render_result(future.result)

    @app.post("/api/v1/cancel_future")
    async def cancel_future(request: Request):
        payload = await request.json()
        service.cancel(_tenant(request), payload["request_id"])
        return {"status": "ok"}

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: Request):
        payload = await request.json()
        validate_create_sampling_session(payload)
        sampling_session_id = service.create_sampling_session(_tenant(request), payload)
        return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}

    @app.get("/api/v1/samplers/{sampling_session_id}")
    async def get_sampler(sampling_session_id: str, request: Request):
        return service.get_sampler(_tenant(request), sampling_session_id)

    @app.post("/api/v1/asample")
    async def asample(request: Request):
        payload = decode_sample_request(await request.json())
        request_id, sequence_ids = service.submit_sample(_tenant(request), payload)
        return {"request_id": request_id, "sample_sequence_ids": sequence_ids}

    return app
