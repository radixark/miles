"""Mount the sampling-scoped tinker SDK routes onto the existing control-plane FastAPI app."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from miles.tinker.service import TinkerService

CLIENT_CONFIG = {
    "pjwt_auth_enabled": False,
    "parallel_fwdbwd_chunks": False,
    "proto_compress_fwdbwd": False,
    "create_model_via_load_weights": True,
    "sample_max_concurrent_requests": 64,
}


def mount_tinker_routes(app, service: TinkerService) -> None:
    @app.get("/api/v1/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    async def client_config(request: Request):
        return CLIENT_CONFIG

    @app.post("/api/v1/client/dynamic_config")
    async def client_dynamic_config(request: Request):
        return {"sample_cancel_enabled": False}

    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities():
        return {"supported_models": [{"model_name": service.args.hf_checkpoint}]}

    @app.post("/api/v1/create_session")
    async def create_session(request: Request):
        return {"type": "create_session", "session_id": service.create_session()}

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(request: Request):
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/telemetry")
    async def telemetry(request: Request):
        return {"status": "accepted"}

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: Request):
        status, body = service.create_sampling_session(await request.json())
        return JSONResponse(body, status_code=status)

    @app.get("/api/v1/samplers/{sampler_id}")
    async def get_sampler(sampler_id: str):
        status, body = service.get_sampler(sampler_id)
        return JSONResponse(body, status_code=status)

    @app.post("/api/v1/asample")
    async def asample(request: Request):
        status, body = service.submit_sample(await request.json())
        return JSONResponse(body, status_code=status)

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request):
        body = await request.json()
        kind, payload = await service.retrieve(body.get("request_id"))
        if kind == "proto":
            return Response(content=payload, media_type="application/x-protobuf")
        if kind == "json":
            return JSONResponse(payload)
        status, error_body = payload
        return JSONResponse(error_body, status_code=status)
