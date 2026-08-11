from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0


@contextmanager
def polling_in_background(step: Callable[[], None], *, description: str, join_timeout: float) -> Iterator[None]:
    stop = threading.Event()
    thread = threading.Thread(target=_poll_until_stopped, args=(step, description, stop), daemon=True)
    thread.start()

    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=join_timeout)


def _poll_until_stopped(step: Callable[[], None], description: str, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            step()
        except Exception:
            logger.warning(f"Could not {description}; retrying", exc_info=True)
        stop.wait(POLL_INTERVAL_SECONDS)
