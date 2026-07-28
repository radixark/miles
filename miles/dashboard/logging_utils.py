"""Logging helpers shared by the dashboard's background collectors."""

from __future__ import annotations

import logging
import time


class RateLimitedWarner:
    """Warn at most once per ``interval_seconds``.

    Background loops (scraper ticks, NVML sampling) hit the same failure
    thousands of times during an outage; observability must neither spam the
    training logs nor die silently, so the first warning goes through and
    repeats are suppressed for the window.
    """

    def __init__(self, logger: logging.Logger, *, interval_seconds: float = 300.0):
        self._logger = logger
        self.interval_seconds = interval_seconds
        self._last_warn = float("-inf")

    def warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warn >= self.interval_seconds:
            self._last_warn = now
            self._logger.warning("%s (further warnings suppressed for %.0fs)", message, self.interval_seconds)

    def reset_window_for_test(self) -> None:
        self._last_warn = float("-inf")
