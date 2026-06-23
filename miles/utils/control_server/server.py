from __future__ import annotations

import asyncio
import logging
import threading

import ray
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from miles.ray.train.group import RayTrainGroup
from miles.utils.control_server.handles import _ActorCellHandle, _CellHandle, _RolloutCellHandle
from miles.utils.control_server.models import Cell, CellList, CellPatch, FaultInjection, K8sStatus, _OkResponse
from miles.utils.control_server.registry import _CellRegistry

logger = logging.getLogger(__name__)


# -------------------------- entrypoint ------------------------------


def start_control_server(
    *,
    actor_model: RayTrainGroup,
    rollout_manager: object,
    port: int,
    ft_components: list[str],
) -> None:
    registry = _CellRegistry()

    if "train" in ft_components:
        for i in range(len(actor_model._cells)):
            registry.register(_ActorCellHandle(group=actor_model, cell_index=i))

    if "rollout" in ft_components:
        # TODO the code will NOT work before implementing rollout ft
        num_rollout_cells = ray.get(rollout_manager.get_cell_count.remote())
        for i in range(num_rollout_cells):
            registry.register(
                _RolloutCellHandle(
                    rollout_manager=rollout_manager,
                    cell_index=i,
                )
            )

    _start_control_server_raw(registry=registry, port=port)


def _start_control_server_raw(registry: _CellRegistry, port: int) -> None:
    app = _create_control_app(registry)

    def _run() -> None:
        uvicorn.run(app, host="0.0.0.0", port=port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Control server started on port %d", port)


# -------------------------- main app ------------------------------


def _create_control_app(registry: _CellRegistry) -> FastAPI:
    app = FastAPI()

    # -------------------------- exceptions ------------------------------

    @app.exception_handler(_K8sError)
    async def _handle_k8s_error(request: Request, exc: _K8sError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=K8sStatus(message=exc.message, reason=exc.reason, code=exc.status_code).model_dump(),
        )

    # -------------------------- APIs ------------------------------

    @app.get("/api/v1/health")
    async def health() -> _OkResponse:
        return _OkResponse()

    @app.get("/api/v1/cells")
    async def get_cells() -> CellList:
        handles = registry.get_all()
        cells = list(await asyncio.gather(*(h.get_cell() for h in handles)))
        return CellList(items=cells)

    @app.get("/api/v1/cells/{name}")
    async def get_cell(name: str) -> Cell:
        handle = _get_handle(name)
        return await handle.get_cell()

    @app.patch("/api/v1/cells/{name}")
    async def patch_cell(name: str, body: CellPatch) -> Cell:
        handle = _get_handle(name)

        if body.spec is not None and body.spec.suspend is not None:
            try:
                if body.spec.suspend:
                    await handle.suspend()
                else:
                    await handle.resume()
            except Exception as err:
                logger.error("Failed to patch cell %s", name, exc_info=True)
                raise _K8sError(
                    status_code=500, reason="InternalError", message=f"Failed to patch cell '{name}'"
                ) from err

        return await handle.get_cell()

    @app.post("/api/v1/cells/{name}/inject-fault")
    async def inject_fault(name: str, body: FaultInjection) -> _OkResponse:
        handle = _get_handle(name)
        try:
            await handle.inject_fault(mode=body.mode, sub_index=body.sub_index)
        except NotImplementedError as err:
            raise _K8sError(
                status_code=400,
                reason="BadRequest",
                message=str(err),
            ) from err
        except Exception as err:
            logger.error("Failed to inject fault into cell %s", name, exc_info=True)
            raise _K8sError(
                status_code=500,
                reason="InternalError",
                message=f"Failed to inject fault into cell '{name}'",
            ) from err
        return _OkResponse()

    # -------------------------- utils ------------------------------

    def _get_handle(name: str) -> _CellHandle:
        try:
            return registry.get(name)
        except KeyError:
            raise _K8sError(status_code=404, reason="NotFound", message=f"Cell '{name}' not found") from None

    return app


# -------------------------- exception ------------------------------


class _K8sError(Exception):
    def __init__(self, *, status_code: int, reason: str, message: str) -> None:
        self.status_code = status_code
        self.reason = reason
        self.message = message
