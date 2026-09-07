import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future

logger = logging.getLogger(__name__)

_SHUTDOWN_GRACE_SECONDS = 5.0


class _CellWriteExecutor:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, fn: Callable, *args: object) -> Future:
        assert not self._closed, f"[P2P-Shared] the write executor of cell {self.cell_id} is already closed"
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"p2p-write-{self.cell_id}", daemon=True)
            self._thread.start()

        future: Future = Future()
        self._queue.put((future, fn, args))
        return future

    def close(self, timeout: float = _SHUTDOWN_GRACE_SECONDS) -> bool:
        self._closed = True
        if self._thread is None:
            return True

        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.error(f"[P2P-Shared] the write thread of cell {self.cell_id} is still inside a native transfer")
            return False
        self._thread = None
        return True

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # noqa: BLE001
                future.set_exception(error)
