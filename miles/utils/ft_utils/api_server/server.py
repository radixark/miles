from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from miles.ray.specs.inference import compute_engine_pool_ids
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell, CellList, CellPatch, FaultInjection, K8sStatus, _OkResponse
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.workers.cell_operations.base import BaseCellOperations, CellGenerationMismatchError
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)

_API_SERVER_STARTUP_TIMEOUT_SECONDS = 30.0
_THREAD_READY_POLL_INTERVAL_SECONDS = 0.05


# -------------------------- entrypoint ------------------------------


def start_api_server(
    *,
    args,
    trainer_models: dict[str, BaseWorkerHandle],
    inference_controller: BaseWorkerHandle | None,
    port: int,
    ft_components: list[str],
    cell_operations: BaseCellOperations,
) -> None:
    handlers: list[_CellHandler] = []

    if "train" in ft_components:
        handlers.append(
            _CellHandler(
                cell_type="actor",
                operations=cell_operations,
                controllers=list(trainer_models.values()),
                pool_ids=[compute_trainer_pool_id(trainer_id) for trainer_id in trainer_models],
            )
        )

    if "rollout" in ft_components:
        assert inference_controller is not None, (
            "rollout cells are suspended and resumed through the inference controller, so a deployment that runs "
            "none of its own cannot answer for them"
        )
        handlers.append(
            _CellHandler(
                cell_type="rollout",
                operations=cell_operations,
                controllers=[inference_controller],
                pool_ids=compute_engine_pool_ids(args),
            )
        )

    _start_api_server_raw(registry=_CellRegistry(handlers), port=port)


def _start_api_server_raw(registry: _CellRegistry, port: int) -> uvicorn.Server:
    app = _create_api_app(registry)

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port))
    _start_and_wait_thread(
        target=server.run,
        is_ready=lambda: server.started,
        description=f"Api server on port {port}",
        timeout_seconds=_API_SERVER_STARTUP_TIMEOUT_SECONDS,
    )
    return server


# -------------------------- main app ------------------------------


def _create_api_app(registry: _CellRegistry) -> FastAPI:
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
        return CellList(items=await registry.list_cells())

    @app.get("/api/v1/cells/{name}")
    async def get_cell(name: str) -> Cell:
        handler = await _resolve(name)
        return await handler.get_cell(name)

    @app.patch("/api/v1/cells/{name}")
    async def patch_cell(name: str, body: CellPatch) -> Cell:
        handler = await _resolve(name)

        if body.spec is not None and body.spec.suspend is not None:
            try:
                if body.spec.suspend:
                    await handler.suspend(name)
                else:
                    await handler.resume(name)
            except Exception as err:
                logger.error("Failed to patch cell %s", name, exc_info=True)
                raise _K8sError(
                    status_code=500, reason="InternalError", message=f"Failed to patch cell '{name}'"
                ) from err

        return await handler.get_cell(name)

    @app.post("/api/v1/cells/{name}/inject-fault")
    async def inject_fault(name: str, body: FaultInjection) -> _OkResponse:
        handler = await _resolve(name)
        try:
            await handler.inject_fault(
                name,
                expected_workers_hash=body.workers_hash,
                mode=body.mode,
                sub_index=body.sub_index,
            )
        except CellGenerationMismatchError as err:
            raise _K8sError(
                status_code=409,
                reason="Conflict",
                message=str(err),
            ) from err
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

    async def _resolve(name: str) -> _CellHandler:
        try:
            return await registry.resolve(name)
        except KeyError:
            raise _K8sError(status_code=404, reason="NotFound", message=f"Cell '{name}' not found") from None

    return app


# -------------------------- exception ------------------------------


class _K8sError(Exception):
    def __init__(self, *, status_code: int, reason: str, message: str) -> None:
        self.status_code = status_code
        self.reason = reason
        self.message = message


# -------------------------- thread startup ------------------------------


def _start_and_wait_thread(
    *,
    target: Callable[[], None],
    is_ready: Callable[[], bool],
    description: str,
    timeout_seconds: float,
) -> threading.Thread:
    error: list[BaseException] = []

    def _run() -> None:
        try:
            target()
        except BaseException as err:  # noqa: BLE001 - re-raised on the caller thread below
            logger.error("%s died", description, exc_info=True)
            error.append(err)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while not is_ready():
        if error:
            raise RuntimeError(f"{description} failed during startup") from error[0]
        if not thread.is_alive():
            raise RuntimeError(f"{description} exited during startup")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{description} did not finish startup within {timeout_seconds}s")
        time.sleep(_THREAD_READY_POLL_INTERVAL_SECONDS)

    logger.info("%s started", description)
    return thread
