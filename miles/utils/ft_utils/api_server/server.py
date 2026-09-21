from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import (
    Cell,
    CellList,
    CellPatch,
    K8sStatus,
    _OkResponse,
)
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookConflictError
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.cell_operations.base import BaseCellOperations, StaleFaultTargetError
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
    host: str = "127.0.0.1",
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
                category=None,
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
                pool_ids=None,
                category=POOL_CATEGORY_INFERENCE_ENGINE,
            )
        )

    _start_api_server_raw(registry=_CellRegistry(handlers), host=host, port=port)


def _start_api_server_raw(*, registry: _CellRegistry, port: int, host: str) -> uvicorn.Server:
    app = _create_api_app(registry)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
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

    @app.get("/api/v1/cells/{name}/fault-target")
    async def get_fault_target(name: str, rank: int = 0) -> ObservedFaultHookTarget:
        handler = await _resolve(name)
        with _translate_fault_errors(name, action="Fault target observation"):
            return await handler.observe_fault_target(name, rank=rank)

    @app.post("/api/v1/cells/{name}/fault-hook")
    async def control_fault_hook(name: str, body: FaultHookCommand) -> FaultHookRecord:
        target = body.request.target
        if not isinstance(target, ObservedFaultHookTarget):
            raise _K8sError(status_code=400, reason="BadRequest", message="Fault hook must name an observed target")
        if target.cell_id != name:
            raise _K8sError(status_code=400, reason="BadRequest", message="Fault target does not match route")
        handler = await _resolve(name)
        with _translate_fault_errors(name, action="Fault hook"):
            return await handler.control_fault_hook(body)

    # -------------------------- utils ------------------------------

    @contextmanager
    def _translate_fault_errors(name: str, *, action: str) -> Iterator[None]:
        try:
            yield
        except StaleFaultTargetError as err:
            raise _K8sError(status_code=412, reason="PreconditionFailed", message=str(err)) from err
        except NotImplementedError as err:
            raise _K8sError(status_code=400, reason="BadRequest", message=str(err)) from err
        except FaultHookConflictError as err:
            raise _K8sError(status_code=409, reason="Conflict", message=str(err)) from err
        except (TimeoutError, asyncio.TimeoutError) as err:
            raise _K8sError(status_code=504, reason="Timeout", message=f"{action} outcome is unknown") from err
        except Exception as err:
            logger.error("%s failed in cell %s", action, name, exc_info=True)
            raise _K8sError(status_code=500, reason="InternalError", message=f"{action} outcome is unknown") from err

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
