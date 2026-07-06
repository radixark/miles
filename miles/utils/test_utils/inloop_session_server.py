"""In-thread session server for tests: the production router app over in-loop workers.

Serves the SAME client-facing app production runs (`build_router_app`) on a real
port, but wires the N `SessionWorker` shards as in-loop IPC handlers on the server
thread's event loop instead of spawning OS processes — real socketpair framing and
worker dispatch, none of the spawn cost. Process lifecycle (spawn, readiness,
fail-fast, teardown) is the supervisor's job and is covered by its own tests.
"""

import asyncio
import socket
import threading
import time

import uvicorn

from miles.rollout.session.core import build_session_core
from miles.rollout.session.ipc import open_unix_channel
from miles.rollout.session.router import build_router_app
from miles.rollout.session.worker import ProxyBackend, SessionWorker


class InloopSessionServer:
    """Router app + `n_workers` in-loop worker shards, served from one thread."""

    def __init__(self, args, backend_url: str, host: str, port: int, n_workers: int = 1):
        self.args = args
        self.backend_url = backend_url
        self.host = host
        self.port = port
        self.n_workers = n_workers
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self._thread.start()
        self._wait_for_port_open()

    async def _serve(self) -> None:
        # Channels (and the workers behind them) must live on the serving loop,
        # so all wiring happens here, before the port opens. Both channel ends are
        # kept referenced: asyncio holds tasks weakly, so an unreferenced channel
        # (with its reader/writer tasks) would be garbage-collected mid-serve.
        channels, worker_channels, backends = [], [], []
        for _ in range(self.n_workers):
            worker_end, router_end = socket.socketpair()
            backend = ProxyBackend(self.backend_url, timeout=getattr(self.args, "miles_router_timeout", 600.0))
            backends.append(backend)
            worker = SessionWorker(build_session_core(backend, self.args))
            worker_channels.append(await open_unix_channel(worker_end, request_handler=worker.handle))
            channels.append(await open_unix_channel(router_end))
        app = build_router_app(channels, getattr(self.args, "session_server_instance_id", None))
        self._server = uvicorn.Server(uvicorn.Config(app, host=self.host, port=self.port, log_level="warning"))
        try:
            await self._server.serve()
        finally:
            for channel in worker_channels + channels:
                await channel.aclose()
            for backend in backends:
                await backend.aclose()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _wait_for_port_open(self) -> None:
        for _ in range(100):
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError("InloopSessionServer thread died during startup")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex((self.host, self.port)) == 0:
                    return
            time.sleep(0.1)
        raise RuntimeError(f"Failed to start server on {self.url}")
