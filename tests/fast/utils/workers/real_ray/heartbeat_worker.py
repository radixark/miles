import threading
from pathlib import Path

from miles.utils.workers.rpc.common.metadata import rpc


class _HeartbeatWorker:
    def __init__(self) -> None:
        self._release = threading.Event()

    def train(self, *, started_path: str) -> None:
        Path(started_path).touch()
        assert self._release.wait(timeout=15), "Training was not released by the heartbeat"

    @rpc(concurrency_group="heartbeat_status")
    def heartbeat(self) -> str:
        self._release.set()
        return "alive"
