"""Tests for SlidingWindowCounter and SlidingWindowThrottle."""

from __future__ import annotations

import time

from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle


def _now() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# SlidingWindowCounter
# ---------------------------------------------------------------------------


class TestSlidingWindowCounter:
    def test_empty_counter(self) -> None:
        counter = SlidingWindowCounter(window_seconds=60, threshold=3)
        assert counter.count == 0
        assert not counter.threshold_reached
        assert not counter.should_notify

    def test_below_threshold_not_reached(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=3)
        counter.record("a", _now=t)
        counter.record("b", _now=t + 1.0)
        assert counter.count == 2
        assert not counter.threshold_reached

    def test_at_threshold_reached(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=3)
        counter.record("a", _now=t)
        counter.record("a", _now=t + 1.0)
        counter.record("b", _now=t + 2.0)
        assert counter.threshold_reached

    def test_above_threshold_reached(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=3)
        counter.record("a", _now=t)
        counter.record("a", _now=t + 1.0)
        counter.record("b", _now=t + 2.0)
        counter.record("c", _now=t + 3.0)
        assert counter.threshold_reached

    def test_should_notify_fires_once(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=2)
        counter.record("a", _now=t)
        counter.record("b", _now=t + 1.0)
        assert counter.should_notify
        assert not counter.should_notify
        assert not counter.should_notify

    def test_window_expiry_resets(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=10, threshold=2)
        counter.record("a", _now=t)
        counter.record("a", _now=t + 1.0)
        assert counter.should_notify

        counter.record("a", _now=t + 20.0)
        assert not counter.should_notify

        counter.record("a", _now=t + 21.0)
        assert counter.should_notify

    def test_record_with_label(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=5)
        counter.record("DetA", _now=t)
        counter.record("DetB", _now=t + 1.0)
        counter.record("DetA", _now=t + 2.0)
        summary = counter.summary()
        assert "DetA=2" in summary
        assert "DetB=1" in summary

    def test_summary_format(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=5)
        counter.record("X", _now=t)
        counter.record("Y", _now=t + 1.0)
        summary = counter.summary()
        assert summary.startswith("2 events in 60")
        assert "X=1" in summary
        assert "Y=1" in summary

    def test_prune_does_not_lose_recent(self) -> None:
        """Events exactly at the window boundary are kept."""
        t = _now()
        counter = SlidingWindowCounter(window_seconds=10, threshold=2)
        counter.record("a", _now=t)
        counter.record("b", _now=t + 10.0)
        assert counter.threshold_reached

    def test_well_above_threshold_still_reached(self) -> None:
        t = _now()
        counter = SlidingWindowCounter(window_seconds=60, threshold=2)
        counter.record(_now=t)
        counter.record(_now=t + 1.0)
        counter.record(_now=t + 2.0)
        counter.record(_now=t + 3.0)
        assert counter.threshold_reached


# ---------------------------------------------------------------------------
# SlidingWindowThrottle
# ---------------------------------------------------------------------------


class TestSlidingWindowThrottle:
    def test_not_throttled_below_max_count(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=3)
        throttle.record()
        throttle.record()
        assert not throttle.is_throttled()

    def test_throttled_at_max_count(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=3)
        throttle.record()
        throttle.record()
        throttle.record()
        assert throttle.is_throttled()

    def test_empty_history_not_throttled(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=1)
        assert not throttle.is_throttled()

    def test_throttle_is_not_one_shot(self) -> None:
        """Unlike should_notify, is_throttled returns True on every call."""
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=2)
        throttle.record()
        throttle.record()
        assert throttle.is_throttled()
        assert throttle.is_throttled()
        assert throttle.is_throttled()
