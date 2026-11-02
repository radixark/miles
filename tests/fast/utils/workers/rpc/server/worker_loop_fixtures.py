import asyncio
import threading
from multiprocessing.connection import Connection

from miles.utils.workers.rpc.common.metadata import DedicatedRpcLoopWorker
from miles.utils.workers.rpc.common.protocol import SubmitRequest
from miles.utils.workers.rpc.server import core
from miles.utils.workers.rpc.server.core import RpcServer
from miles.utils.workers.rpc.server.worker_loop import _DedicatedWorkerLoop


def _worker_loop_failure_probe(connection: Connection, mode: str) -> None:
    async def run() -> None:
        worker_loop = _DedicatedWorkerLoop()
        connection.send("ready")
        if mode == "stop":
            worker_loop._loop.call_soon_threadsafe(worker_loop._loop.stop)
            await asyncio.Event().wait()
        elif mode == "shutdown":

            async def fail_on_cancel() -> None:
                try:
                    await asyncio.Event().wait()
                finally:
                    raise SystemExit(7)

            async def start_background_task() -> None:
                asyncio.create_task(fail_on_cancel())
                await asyncio.sleep(0)

            await worker_loop.call(method=start_background_task, kwargs={}, is_async=True)
            await worker_loop.close()
        else:

            async def fail() -> None:
                if mode == "system_exit":
                    raise SystemExit(7)
                raise KeyboardInterrupt

            await worker_loop.call(method=fail, kwargs={}, is_async=True)

    asyncio.run(run())


def _blocked_checkpoint_probe(connection: Connection, mode: str) -> None:
    async def run() -> None:
        core._SHUTDOWN_TIMEOUT_SECONDS = 0.2
        worker = _BlockedCheckpointWorker()
        server = RpcServer(worker=worker)
        server.submit_call(method_name="save", request=SubmitRequest(call_id="checkpoint", query={}))
        assert await asyncio.to_thread(worker.started.wait, 5)
        connection.send("ready")
        await server.close()

    asyncio.run(run())


class _BlockedCheckpointWorker(DedicatedRpcLoopWorker):
    def __init__(self) -> None:
        self.started = threading.Event()

    async def save(self) -> None:
        self.started.set()
        threading.Event().wait(timeout=60)
