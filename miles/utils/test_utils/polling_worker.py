import threading
from collections.abc import Callable

RunUntilStoppedFn = Callable[[threading.Event], None]
TickFn = Callable[[], None]


class PollingWorker:
    def __init__(self, *, name: str, run: RunUntilStoppedFn) -> None:
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=run, args=(self._stop_event,), daemon=True, name=name)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, *, timeout_seconds: float) -> None:
        self._thread.join(timeout=timeout_seconds)

    def stop_and_join(self, *, timeout_seconds: float) -> None:
        self.stop()
        self.join(timeout_seconds=timeout_seconds)

    def assert_not_running(self, *, message: str) -> None:
        assert not self.is_running, message


def poll_until_stopped(stop_event: threading.Event, *, tick: TickFn, poll_interval_seconds: float) -> None:
    while not stop_event.is_set():
        tick()
        stop_event.wait(timeout=poll_interval_seconds)
