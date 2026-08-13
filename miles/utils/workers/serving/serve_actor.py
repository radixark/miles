from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

import uvicorn

from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils.fault_injector import inject_fault as _inject_fault
from miles.utils.workers.rpc.server.app import create_rpc_app

logger = logging.getLogger(__name__)

SERVE_HOST = "0.0.0.0"

_SERVER_THREAD_NAME = "miles-rpc-server"


class ServeActor(NodeProbeMixin):
    def __init__(self, *, build_worker: Callable[[], Any]) -> None:
        self._worker = build_worker()
        self._server_thread: threading.Thread | None = None

    def start_rpc_server(self, *, port: int) -> None:
        assert self._server_thread is None, "the rpc server of this actor is already running"

        app = create_rpc_app(self._worker)
        logger.info(f"ServeActor serves {type(self._worker).__name__} on {SERVE_HOST}:{port}")
        self._server_thread = threading.Thread(
            target=serve_until_stopped,
            kwargs=dict(app=app, port=port),
            name=_SERVER_THREAD_NAME,
            daemon=True,
        )
        self._server_thread.start()

    def inject_fault(self, mode: str) -> None:
        _inject_fault(mode=mode)


def serve_until_stopped(*, app: Any, port: int) -> None:
    try:
        uvicorn.run(app, host=SERVE_HOST, port=port)
        logger.error(f"The rpc server on port {port} stopped serving, so the process it lives in exits with it")
    except BaseException:
        logger.error(f"The rpc server on port {port} crashed, so the process it lives in exits with it", exc_info=True)

    os._exit(1)
