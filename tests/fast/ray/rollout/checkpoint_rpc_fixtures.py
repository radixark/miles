from __future__ import annotations

import asyncio
import socket
import threading
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn

from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.server.app import create_rpc_app


@dataclass
class CheckpointRpc:
    worker: CheckpointWorker
    client: httpx.AsyncClient
    handle: RpcWorkerHandle


@asynccontextmanager
async def serve_checkpoint_worker(directory: Path) -> AsyncIterator[CheckpointRpc]:
    worker = CheckpointWorker(directory)
    app = create_rpc_app(worker)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with httpx.AsyncClient(base_url=url, timeout=2.0, trust_env=False) as client:
                handle = RpcWorkerHandle(CheckpointWorker, server_url=url, http_client=client)
                await handle.wait_ready(timeout=5.0)
                await handle.init()
                yield CheckpointRpc(worker=worker, client=client, handle=handle)
        finally:
            worker.data_source.release.set()
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=15.0)


class CheckpointWorker(RolloutExecutor):
    def __init__(self, directory: Path) -> None:
        self.args = Namespace(
            save=str(directory),
            load=str(directory),
            save_debug_event_data=None,
            ci_test=False,
            ci_inject_missing_prefetched_batch_bug=False,
        )
        self.use_legacy_rollout_v1 = True
        self.data_source = _BlockingDataSource()
        self._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=self.args)
        self.produced = threading.Event()
        self.cancelled = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.produce: asyncio.Event | None = None

    async def init(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.produce = asyncio.Event()
        self._producer = asyncio.create_task(self._produce())

    async def _produce(self) -> None:
        try:
            await self.produce.wait()
            self.data_source.tokens.append(2)
            self.produced.set()
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


class _BlockingDataSource:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.tokens = [1]

    def save(self, directory: Path) -> None:
        before = list(self.tokens)
        self._block()
        assert self.tokens == before
        save_simple_checkpoint(directory=directory, data=self.tokens)

    def load(self, directory: Path) -> None:
        self._block()
        self.tokens = load_simple_checkpoint(directory=directory)

    def _block(self) -> None:
        self.entered.set()
        assert self.release.wait(timeout=10.0), "checkpoint release timed out"
