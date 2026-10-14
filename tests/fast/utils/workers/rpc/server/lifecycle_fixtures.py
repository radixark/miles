import asyncio
import multiprocessing
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection

from miles.utils.workers.rpc.common.protocol import SubmitRequest
from miles.utils.workers.rpc.server import core
from miles.utils.workers.rpc.server.core import RpcServer


def _run_process_probe(*, target: Callable[[Connection, str], None], mode: str) -> int:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, mode))
    process.start()
    sender.close()
    try:
        assert receiver.poll(60), "The child did not reach the fault injection"
        assert receiver.recv() == "ready"
        process.join(timeout=10)
        assert not process.is_alive(), "The worker process did not terminate"
        assert process.exitcode is not None
        return process.exitcode
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        receiver.close()
        process.close()


def _shutdown_probe(connection: Connection, mode: str) -> None:
    async def run() -> None:
        core._SHUTDOWN_TIMEOUT_SECONDS = 0.2
        worker = _ShutdownWorker()
        server = RpcServer(worker=worker)
        if mode != "normal":
            method = "block_sync" if mode == "sync" else "ignore_cancellation"
            server.submit_call(method_name=method, request=SubmitRequest(call_id="blocked", query={}))
            assert await asyncio.to_thread(worker.started.wait, 5)
        connection.send("ready")
        await server.close()
        await asyncio.sleep(0.4)

    asyncio.run(run())


class _ShutdownWorker:
    def __init__(self) -> None:
        self.started = threading.Event()

    def block_sync(self) -> None:
        self.started.set()
        threading.Event().wait(timeout=60)

    async def ignore_cancellation(self) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()
