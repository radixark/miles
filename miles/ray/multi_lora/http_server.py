"""Multi-LoRA control-plane HTTP API over a MultiLoRABackend.

Subclass via ``--multi-lora-http-server-path`` (override add_routes /
create_app)."""

import asyncio
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from miles.ray.multi_lora.registry import AdapterState
from miles.utils.adapter_config import AdapterRunConfig, parse_adapter_run_yaml


class RegisterAdapterRequest(BaseModel):
    """Exactly one of ``config`` (inline) or ``yaml_path`` must be set."""

    name: str
    config: AdapterRunConfig | None = None
    yaml_path: str | None = None


_NAMES_QUERY = Query(default_factory=list)


class MultiLoRAHTTPServer:
    """Control-plane API over a MultiLoRABackend. Subclass via
    --multi-lora-http-server-path (add_routes / create_app)."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        self.backend = backend
        self.host = host
        self.api_port = api_port
        self.api_server: uvicorn.Server | None = None
        self.api_task: asyncio.Task | None = None

    @property
    def actual_api_port(self) -> int:
        if self.api_server is not None and self.api_server.started:
            return self.api_server.servers[0].sockets[0].getsockname()[1]
        return self.api_port

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Miles Multi-LoRA Controller")

        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            return JSONResponse({"detail": str(exc)}, status_code=400)

        @app.exception_handler(RuntimeError)
        async def runtime_error_handler(request: Request, exc: RuntimeError):
            status = 409 if "No free adapter slots" in str(exc) else 500
            return JSONResponse({"detail": str(exc)}, status_code=status)

        return app

    def add_routes(self, app: FastAPI) -> None:
        app.get("/health")(self.health)
        app.get("/adapter_runs")(self.list_adapters)
        app.get("/adapter_runs/state")(self.adapter_states)  # before /adapter_runs/{name}
        app.get("/adapter_runs/{name}")(self.get_adapter)
        app.post("/adapter_runs")(self.register_adapter)
        app.delete("/adapter_runs/{name}")(self.deregister_adapter)

    async def start(self) -> None:
        app = self.create_app()
        self.add_routes(app)
        config = uvicorn.Config(app, host=self.host, port=self.api_port, log_level="warning", access_log=False)
        self.api_server = uvicorn.Server(config)
        self.api_task = asyncio.create_task(self.api_server.serve())
        while not self.api_server.started:
            if self.api_task.done():
                self.api_task.result()
                raise RuntimeError("uvicorn exited before startup completed")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self.api_server is not None:
            self.api_server.should_exit = True
            await self.api_task
        self.api_server = self.api_task = None

    async def health(self) -> dict:
        return {"status": "healthy"}

    def adapter_statuses(self) -> list[dict]:
        registry = self.backend.registry
        statuses = []
        for record in registry.records.values():
            flat = asdict(registry.view(record))
            flat |= flat.pop("config")
            flat["save"] = str(flat["save"])
            flat["state"] = record.state
            if record.state is AdapterState.COMPLETED:
                flat["version"] = None
            statuses.append(flat)
        return statuses

    async def list_adapters(self) -> dict:
        return {"adapters": self.adapter_statuses()}

    async def adapter_states(self, names: list[str] = _NAMES_QUERY) -> dict:
        return {"states": {name: self.backend.registry.adapter_state(name) for name in names}}

    async def get_adapter(self, name: str) -> dict:
        for status in self.adapter_statuses():
            if status["name"] == name:
                return status
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not registered")

    async def register_adapter(self, request: RegisterAdapterRequest) -> dict:
        if (request.config is None) == (request.yaml_path is None):
            raise HTTPException(status_code=400, detail="Exactly one of 'config' or 'yaml_path' must be set")
        if request.yaml_path is not None:
            config = parse_adapter_run_yaml(Path(request.yaml_path))
        else:
            config = request.config
        return await self.backend.register(request.name, config)

    async def deregister_adapter(self, name: str) -> dict:
        state = self.backend.registry.adapter_state(name)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Adapter '{name}' not registered")
        await self.backend.deregister(name)
        return {"status": "ok", "name": name}
