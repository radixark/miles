"""Tinker-compatible FastAPI control plane backed by Miles primitives.

The wire surface follows SkyRL's Tinker server so an unmodified Tinker SDK can
drive Miles. Execution is deliberately behind a narrow protocol: the Ray actor
implementation can remain collective while API/session state stays local.
"""

from __future__ import annotations

import inspect
import random
from typing import Any, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class TinkerPrimitiveBackend(Protocol):
    async def create_model(self, model_id: str, lora_config: dict[str, Any], model_role: str) -> dict[str, Any]: ...

    async def forward_backward(self, model_id: str, batch: dict[str, Any]) -> dict[str, Any]: ...

    async def forward(self, model_id: str, batch: dict[str, Any]) -> dict[str, Any]: ...

    async def optim_step(self, model_id: str, adam_params: dict[str, float]) -> dict[str, Any]: ...

    async def save_sampler(self, model_id: str, checkpoint_id: str) -> dict[str, Any]: ...

    async def save_checkpoint(self, model_id: str, checkpoint_id: str) -> dict[str, Any]: ...

    async def load_checkpoint(self, model_id: str, path: str) -> dict[str, Any]: ...

    async def sample(self, model_id: str | None, request: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_model(self, model_id: str) -> None: ...


class LoRAConfig(BaseModel):
    rank: int = Field(gt=0)
    seed: int | None = None


class CreateSessionRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] | None = None
    sdk_version: str | None = None


class CreateModelRequest(BaseModel):
    session_id: str
    base_model: str
    lora_config: LoRAConfig
    model_role: str = "policy"


class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None


class ModelRequest(BaseModel):
    model_id: str


class ForwardBackwardRequest(ModelRequest):
    forward_backward_input: dict[str, Any]


class ForwardRequest(ModelRequest):
    forward_input: dict[str, Any]


class OptimStepRequest(ModelRequest):
    adam_params: dict[str, float]


class RetrieveFutureRequest(BaseModel):
    request_id: str


async def _call(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def create_app(backend: TinkerPrimitiveBackend, base_model: str, *, max_lora_rank: int = 32) -> FastAPI:
    """Create the SDK-compatible API around a persistent Miles actor facade."""
    app = FastAPI(title="Miles Tinker API")
    sessions: set[str] = set()
    models: dict[str, dict[str, Any]] = {}
    sampling_sessions: dict[str, dict[str, Any]] = {}
    futures: dict[str, Any] = {}

    async def completed(result: Any) -> str:
        request_id = str(len(futures) + 1)
        futures[request_id] = result
        return request_id

    def require_model(model_id: str) -> dict[str, Any]:
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        return models[model_id]

    @app.get("/api/v1/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    async def client_config() -> dict[str, bool]:
        return {"pjwt_auth_enabled": False}

    @app.post("/api/v1/create_session")
    async def create_session(request: CreateSessionRequest) -> dict[str, str]:
        session_id = f"session_{uuid4().hex[:8]}"
        sessions.add(session_id)
        return {"session_id": session_id}

    @app.post("/api/v1/session_heartbeat")
    async def heartbeat(request: dict[str, Any]) -> dict[str, str]:
        if request.get("session_id") not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok"}

    @app.post("/api/v1/telemetry")
    async def telemetry(request: dict[str, Any]) -> dict[str, str]:
        return {"status": "accepted"}

    @app.get("/api/v1/get_server_capabilities")
    async def capabilities() -> dict[str, Any]:
        return {"supported_models": [{"model_name": base_model}]}

    @app.post("/api/v1/create_model")
    async def create_model(request: CreateModelRequest) -> dict[str, Any]:
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        if request.base_model != base_model:
            raise HTTPException(status_code=400, detail=f"Server is pinned to {base_model}")
        if request.lora_config.rank > max_lora_rank:
            raise HTTPException(status_code=400, detail=f"LoRA rank exceeds allocated maximum {max_lora_rank}")
        model_id = f"model_{uuid4().hex[:8]}"
        lora = request.lora_config.model_dump()
        lora["seed"] = lora["seed"] if lora["seed"] is not None else random.randint(0, 2**31 - 1)
        result = await _call(backend.create_model(model_id, lora, request.model_role))
        future_id = await completed(result)
        models[model_id] = {"base_model": base_model, "lora_config": request.lora_config.model_dump()}
        return {
            "model_id": model_id,
            "base_model": base_model,
            "lora_config": request.lora_config.model_dump(),
            "status": "created",
            "request_id": future_id,
        }

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: CreateSamplingSessionRequest) -> dict[str, str]:
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        if (request.base_model is None) == (request.model_path is None):
            raise HTTPException(status_code=400, detail="Exactly one of base_model or model_path must be provided")
        if request.base_model is not None and request.base_model != base_model:
            raise HTTPException(status_code=400, detail=f"Server is pinned to {base_model}")

        sampling_session_id = f"sampling_{uuid4().hex[:8]}"
        sampling_sessions[sampling_session_id] = {
            "model_id": None,
            "base_model": request.base_model or base_model,
            "model_path": request.model_path,
            "sampling_session_seq_id": request.sampling_session_seq_id,
        }
        return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}

    @app.get("/api/v1/samplers/{sampler_id}")
    async def get_sampler(sampler_id: str) -> dict[str, Any]:
        sampling_session = sampling_sessions.get(sampler_id)
        if sampling_session is None:
            raise HTTPException(status_code=404, detail="Sampler not found")
        return {
            "sampler_id": sampler_id,
            "base_model": sampling_session["base_model"],
            "model_path": sampling_session.get("model_path"),
        }

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: ForwardBackwardRequest) -> dict[str, str]:
        require_model(request.model_id)
        result = await _call(backend.forward_backward(request.model_id, request.forward_backward_input))
        future_id = await completed(result)
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/forward")
    async def forward(request: ForwardRequest) -> dict[str, str]:
        require_model(request.model_id)
        result = await _call(backend.forward(request.model_id, request.forward_input))
        future_id = await completed(result)
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/optim_step")
    async def optim_step(request: OptimStepRequest) -> dict[str, str]:
        require_model(request.model_id)
        result = await _call(backend.optim_step(request.model_id, request.adam_params))
        future_id = await completed(result)
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(request: dict[str, Any]) -> dict[str, str]:
        model_id = request["model_id"]
        require_model(model_id)
        checkpoint_id = request.get("path") or f"ss{request['sampling_session_seq_id']}_seq{request['seq_id']}"
        result = {**await _call(backend.save_sampler(model_id, checkpoint_id)), "type": "save_weights_for_sampler"}
        if request.get("sampling_session_seq_id") is not None:
            sampling_session_id = f"sampling_{uuid4().hex[:8]}"
            sampling_sessions[sampling_session_id] = {
                "model_id": model_id,
                "model_path": f"tinker://{model_id}/sampler_weights/{checkpoint_id}",
            }
            result = {**result, "sampling_session_id": sampling_session_id}
        future_id = await completed(result)
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/save_weights")
    async def save_weights(request: dict[str, Any]) -> dict[str, str]:
        model_id = request["model_id"]
        require_model(model_id)
        checkpoint_id = request["path"]
        result = await _call(backend.save_checkpoint(model_id, checkpoint_id))
        result = {**result, "path": f"tinker://{model_id}/weights/{checkpoint_id}", "type": "save_weights"}
        future_id = await completed(result)
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/load_weights")
    async def load_weights(request: dict[str, Any]) -> dict[str, str]:
        model_id = request["model_id"]
        require_model(model_id)
        result = await _call(backend.load_checkpoint(model_id, request["path"]))
        future_id = await completed({**result, "type": "load_weights"})
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/get_info")
    async def get_info(request: ModelRequest) -> dict[str, Any]:
        metadata = require_model(request.model_id)
        return {
            "model_id": request.model_id,
            "status": "created",
            "model_data": {
                "base_model": metadata["base_model"],
                "model_name": metadata["base_model"],
                "lora_config": metadata["lora_config"],
            },
        }

    @app.post("/api/v1/asample")
    async def asample(request: dict[str, Any]) -> dict[str, str]:
        model_id = None
        if sampling_session_id := request.get("sampling_session_id"):
            session = sampling_sessions.get(sampling_session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Sampling session not found")
            model_id = session["model_id"]
        result = await _call(backend.sample(model_id, request))
        future_id = await completed(result)
        return {"future_id": future_id, "status": "pending", "request_id": future_id}

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: RetrieveFutureRequest) -> Any:
        if request.request_id not in futures:
            raise HTTPException(status_code=404, detail="Future not found")
        return futures[request.request_id]

    @app.post("/api/v1/unload_model")
    async def unload_model(request: ModelRequest) -> dict[str, str]:
        require_model(request.model_id)
        await _call(backend.delete_model(request.model_id))
        models.pop(request.model_id)
        future_id = await completed({"model_id": request.model_id, "status": "unloaded", "type": "unload_model"})
        return {"request_id": future_id, "model_id": request.model_id}

    return app
