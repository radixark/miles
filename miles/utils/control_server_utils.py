from __future__ import annotations

import abc
import asyncio
import logging
import threading
from typing import Literal

import ray
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


def start_control_server(actor_model: object, rollout_manager: object, port: int) -> None:
    registry = _SubsystemRegistry()

    registry.register(_TrainingSubsystemHandle(node_ids=actor_model.get_node_ids()))

    cell_infos = ray.get(rollout_manager.list_cells.remote())
    for cell_info in cell_infos:
        registry.register(
            _RolloutSubsystemHandle(
                rollout_manager=rollout_manager,
                cell_id=cell_info["cell_id"],
            )
        )

    _start_control_server_raw(registry=registry, port=port)


def _start_control_server_raw(registry: _SubsystemRegistry, port: int) -> None:
    app = _create_control_app(registry)

    def _run() -> None:
        uvicorn.run(app, host="0.0.0.0", port=port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Control server started on port %d", port)


class _SubsystemInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subsystem_id: str
    subsystem_type: Literal["training", "rollout"]
    status: Literal["running", "stopped", "pending", "failed"]
    node_ids: list[str]


class _StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = 30


class _OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


def _create_control_app(registry: _SubsystemRegistry) -> FastAPI:
    app = FastAPI()

    @app.get("/subsystems")
    async def get_subsystems() -> list[_SubsystemInfo]:
        handles = registry.get_all()

        async def _fetch(handle: _SubsystemHandle) -> _SubsystemInfo:
            status, node_ids = await asyncio.gather(handle.get_status(), handle.get_node_ids())
            return _SubsystemInfo(
                subsystem_id=handle.subsystem_id,
                subsystem_type=handle.subsystem_type,
                status=status,
                node_ids=node_ids,
            )

        return list(await asyncio.gather(*(_fetch(h) for h in handles)))

    def _get_handle(subsystem_id: str) -> _SubsystemHandle:
        try:
            return registry.get(subsystem_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Subsystem '{subsystem_id}' not found") from None

    async def _call_handle(subsystem_id: str, action: str, coro) -> _OkResponse:
        try:
            await coro
        except NotImplementedError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception:
            logger.error("Failed to %s subsystem %s", action, subsystem_id, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to {action} subsystem '{subsystem_id}'") from None
        return _OkResponse()

    @app.post("/subsystems/{subsystem_id}/stop")
    async def stop_subsystem(subsystem_id: str, body: _StopRequest | None = None) -> _OkResponse:
        if body is None:
            body = _StopRequest()
        handle = _get_handle(subsystem_id)
        return await _call_handle(subsystem_id, "stop", handle.stop(timeout_seconds=body.timeout_seconds))

    @app.post("/subsystems/{subsystem_id}/start")
    async def start_subsystem(subsystem_id: str) -> _OkResponse:
        handle = _get_handle(subsystem_id)
        return await _call_handle(subsystem_id, "start", handle.start())

    return app


class _SubsystemRegistry:
    def __init__(self) -> None:
        self._handles: dict[str, _SubsystemHandle] = {}

    def register(self, handle: _SubsystemHandle) -> None:
        if handle.subsystem_id in self._handles:
            raise ValueError(f"Subsystem '{handle.subsystem_id}' is already registered")
        self._handles[handle.subsystem_id] = handle

    def get_all(self) -> list[_SubsystemHandle]:
        return list(self._handles.values())

    def get(self, subsystem_id: str) -> _SubsystemHandle:
        try:
            return self._handles[subsystem_id]
        except KeyError as e:
            raise KeyError(f"Subsystem '{subsystem_id}' not found") from e


class _SubsystemHandle(abc.ABC):
    @property
    @abc.abstractmethod
    def subsystem_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def subsystem_type(self) -> str: ...

    @abc.abstractmethod
    async def stop(self, timeout_seconds: int) -> None: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def get_status(self) -> str: ...

    @abc.abstractmethod
    async def get_node_ids(self) -> list[str]: ...


class _TrainingSubsystemHandle(_SubsystemHandle):
    def __init__(self, node_ids: list[str]) -> None:
        self._node_ids = node_ids

    @property
    def subsystem_id(self) -> str:
        return "training"

    @property
    def subsystem_type(self) -> str:
        return "training"

    async def stop(self, timeout_seconds: int) -> None:
        raise NotImplementedError(
            "Training subsystem lifecycle is managed by the platform (kill/relaunch pod), not via control server API"
        )

    async def start(self) -> None:
        raise NotImplementedError(
            "Training subsystem lifecycle is managed by the platform (kill/relaunch pod), not via control server API"
        )

    async def get_status(self) -> str:
        return "running"

    async def get_node_ids(self) -> list[str]:
        return self._node_ids


class _RolloutSubsystemHandle(_SubsystemHandle):
    def __init__(self, rollout_manager: object, cell_id: str) -> None:
        self._rollout_manager = rollout_manager
        self._cell_id = cell_id

    @property
    def subsystem_id(self) -> str:
        return self._cell_id

    @property
    def subsystem_type(self) -> str:
        return "rollout"

    async def stop(self, timeout_seconds: int) -> None:
        await self._rollout_manager.stop_cell.remote(self._cell_id, timeout_seconds)

    async def start(self) -> None:
        await self._rollout_manager.start_cell.remote(self._cell_id)

    async def get_status(self) -> str:
        return await self._rollout_manager.get_cell_status.remote(self._cell_id)

    async def get_node_ids(self) -> list[str]:
        return await self._rollout_manager.get_cell_node_ids.remote(self._cell_id)
