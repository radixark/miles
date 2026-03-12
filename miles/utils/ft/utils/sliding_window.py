"""Sliding-window event counters and throttles."""

from __future__ import annotations

import time


class SlidingWindowCounter:
    """Generic sliding-window event counter with one-shot threshold notification.

    Records timestamped events and prunes those outside the window.
    ``should_notify`` returns True exactly once when the threshold is first
    reached; it resets after the window clears below threshold.
    """

    def __init__(self, *, window_seconds: float, threshold: int) -> None:
        self._window_seconds = window_seconds
        self._threshold = threshold
        self._events: list[tuple[float, str]] = []
        self._notified = False

    def record(self, label: str = "", *, _now: float | None = None) -> None:
        now = _now if _now is not None else time.monotonic()
        self._events.append((now, label))
        self._prune(now)

    @property
    def count(self) -> int:
        self._prune(time.monotonic())
        return len(self._events)

    @property
    def threshold_reached(self) -> bool:
        return self.count >= self._threshold

    @property
    def should_notify(self) -> bool:
        """True once when threshold is first reached; resets after window clears."""
        if self.threshold_reached and not self._notified:
            self._notified = True
            return True
        return False

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for _, label in self._events:
            counts[label] = counts.get(label, 0) + 1
        parts = [f"{label}={count}" for label, count in sorted(counts.items())]
        return f"{len(self._events)} events in {self._window_seconds}s window: {', '.join(parts)}"

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        self._events = [(t, l) for t, l in self._events if t >= cutoff]
        if len(self._events) < self._threshold:
            self._notified = False


class SlidingWindowThrottle:
    """Tracks event frequency and throttles when a limit is exceeded.

    Thin wrapper over ``SlidingWindowCounter`` that preserves the
    ``record()`` / ``is_throttled()`` API.  Unlike ``should_notify``,
    ``is_throttled`` returns True on *every* call while the threshold
    is reached (not one-shot).
    """

    def __init__(self, window_minutes: float, max_count: int) -> None:
        self._counter = SlidingWindowCounter(
            window_seconds=window_minutes * 60,
            threshold=max_count,
        )

    def record(self) -> None:
        self._counter.record()

    def is_throttled(self) -> bool:
        return self._counter.threshold_reached
