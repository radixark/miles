"""FastAPI surface compatible with the official Tinker Python SDK."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer
from miles.ray.tinker.protocol import (
    ClientConfigRequest,
    CreateModelRequest,
    CreateSamplingSessionRequest,
    CreateSessionRequest,
    ForwardBackwardRequest,
    ForwardRequest,
    FutureRetrieveRequest,
    GetInfoRequest,
    LoadWeightsRequest,
    OptimStepRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    TelemetryRequest,
    TinkerError,
    UnloadModelRequest,
    WeightsInfoRequest,
)


class TinkerHTTPServer(MultiLoRAHTTPServer):
    """Tinker SDK wire API hosted by the existing controller actor."""

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Miles Tinker API", version="1")

        @app.exception_handler(TinkerError)
        async def tinker_error_handler(request: Request, exc: TinkerError):
            return JSONResponse(
                {"error": str(exc), "category": exc.category},
                status_code=400 if exc.category == "user" else 500,
            )

        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            return JSONResponse({"error": str(exc), "category": "user"}, status_code=400)

        api_key = getattr(self.backend.args, "tinker_api_key", None)
        if api_key:

            @app.middleware("http")
            async def authenticate(request: Request, call_next):
                if request.url.path in {"/api/v1/healthz", "/health"}:
                    return await call_next(request)
                provided = request.headers.get("x-api-key")
                if provided is None:
                    authorization = request.headers.get("authorization", "")
                    if authorization.startswith("Bearer "):
                        provided = authorization.removeprefix("Bearer ")
                if provided is None or not hmac.compare_digest(provided, api_key):
                    return JSONResponse(
                        {"error": "invalid API key", "category": "user"},
                        status_code=401,
                    )
                return await call_next(request)

        return app

    def add_routes(self, app: FastAPI) -> None:
        app.get("/health")(self.health)
        app.get("/api/v1/healthz")(self.healthz)
        app.get("/api/v1/get_server_capabilities")(self.get_server_capabilities)
        app.post("/api/v1/client/config")(self.client_config)
        app.post("/api/v1/create_session")(self.create_session)
        app.post("/api/v1/session_heartbeat")(self.session_heartbeat)
        app.post("/api/v1/create_model")(self.create_model)
        app.post("/api/v1/get_info")(self.get_info)
        app.post("/api/v1/unload_model")(self.unload_model)
        app.post("/api/v1/forward")(self.forward)
        app.post("/api/v1/forward_backward")(self.forward_backward)
        app.post("/api/v1/optim_step")(self.optim_step)
        app.post("/api/v1/save_weights")(self.save_weights)
        app.post("/api/v1/load_weights")(self.load_weights)
        app.post("/api/v1/save_weights_for_sampler")(self.save_weights_for_sampler)
        app.post("/api/v1/create_sampling_session")(self.create_sampling_session)
        app.get("/api/v1/samplers/{sampling_session_id}")(self.get_sampler)
        app.post("/api/v1/asample")(self.asample)
        app.post("/api/v1/retrieve_future")(self.retrieve_future)
        app.post("/api/v1/weights_info")(self.weights_info)
        app.post("/api/v1/telemetry")(self.telemetry)

    async def health(self) -> dict[str, str]:
        return {"status": "healthy"}

    async def healthz(self) -> dict[str, str]:
        if not getattr(self.backend, "ready", True):
            raise HTTPException(status_code=503, detail="trainer is initializing")
        return {"status": "ok"}

    async def get_server_capabilities(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "supported_models": [
                {
                    "model_name": self.backend.model_name,
                    "max_context_length": getattr(self.backend.args, "seq_length", None),
                }
            ]
        }

    async def client_config(self, request: ClientConfigRequest) -> dict[str, Any]:
        return self.backend.client_config()

    async def create_session(self, request: CreateSessionRequest) -> dict[str, Any]:
        return self.backend.create_session(request)

    async def session_heartbeat(self, request: SessionHeartbeatRequest) -> dict[str, str]:
        return self.backend.session_heartbeat(request.session_id)

    async def create_model(self, request: CreateModelRequest) -> dict[str, Any]:
        return (await self.backend.create_model(request)).model_dump(mode="json", exclude_none=True)

    async def get_info(self, request: GetInfoRequest) -> dict[str, Any]:
        return self.backend.get_info(request.model_id)

    async def unload_model(self, request: UnloadModelRequest) -> dict[str, Any]:
        return (await self.backend.unload_model(request.model_id)).model_dump(mode="json", exclude_none=True)

    async def forward(self, request: ForwardRequest) -> dict[str, Any]:
        return (await self.backend.forward(request)).model_dump(mode="json", exclude_none=True)

    async def forward_backward(self, request: ForwardBackwardRequest) -> dict[str, Any]:
        return (await self.backend.forward_backward(request)).model_dump(mode="json", exclude_none=True)

    async def optim_step(self, request: OptimStepRequest) -> dict[str, Any]:
        return (await self.backend.optim_step(request)).model_dump(mode="json", exclude_none=True)

    async def save_weights(self, request: SaveWeightsRequest) -> dict[str, Any]:
        return (await self.backend.save_weights(request)).model_dump(mode="json", exclude_none=True)

    async def load_weights(self, request: LoadWeightsRequest) -> dict[str, Any]:
        return (await self.backend.load_weights(request)).model_dump(mode="json", exclude_none=True)

    async def save_weights_for_sampler(self, request: SaveWeightsForSamplerRequest) -> dict[str, Any]:
        return (await self.backend.save_weights_for_sampler(request)).model_dump(mode="json", exclude_none=True)

    async def create_sampling_session(self, request: CreateSamplingSessionRequest) -> dict[str, str]:
        return await self.backend.create_sampling_session(request)

    async def get_sampler(self, sampling_session_id: str) -> dict[str, Any]:
        return self.backend.get_sampler(sampling_session_id)

    async def asample(self, request: SampleRequest) -> dict[str, Any]:
        return (await self.backend.sample(request)).model_dump(mode="json", exclude_none=True)

    async def retrieve_future(self, request: FutureRetrieveRequest) -> dict[str, Any]:
        return self.backend.retrieve_future(request.request_id)

    async def weights_info(self, request: WeightsInfoRequest) -> dict[str, Any]:
        return self.backend.weights_info(request.tinker_path)

    async def telemetry(self, request: TelemetryRequest) -> dict[str, str]:
        return {"status": "accepted"}
