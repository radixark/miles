"""Tests for detector crash tracking via SlidingWindowCounter and run_detectors."""

from __future__ import annotations

from unittest.mock import MagicMock

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
import pytest

from miles.utils.ft.controller.state_machines.main.utils import handle_notify_human, run_detectors
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.platform import JobStatus
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter


def _make_detector_context() -> DetectorContext:
    return DetectorContext(
        metric_store=MagicMock(),
        mini_wandb=MagicMock(),
        rank_placement={0: "node-0"},
        job_status=JobStatus.RUNNING,
    )


class _PassingDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision.no_fault(reason="all good")


class _CrashingDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        raise RuntimeError("detector internal error")


class _RecoveryDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-0"],
            reason="found bad node",
            trigger=TriggerType.HARDWARE,
        )


class TestDetectorCrashTracking:
    def test_below_threshold_does_not_notify(self) -> None:
        tracker = SlidingWindowCounter(window_seconds=60, threshold=3)
        tracker.record("DetA", _now=0.0)
        tracker.record("DetB", _now=1.0)
        assert not tracker.should_notify

    def test_reaching_threshold_notifies_once(self) -> None:
        tracker = SlidingWindowCounter(window_seconds=60, threshold=3)
        tracker.record("DetA", _now=0.0)
        tracker.record("DetA", _now=1.0)
        tracker.record("DetB", _now=2.0)
        assert tracker.should_notify
        assert not tracker.should_notify

    def test_window_expiry_resets_notification(self) -> None:
        tracker = SlidingWindowCounter(window_seconds=10, threshold=2)
        tracker.record("DetA", _now=0.0)
        tracker.record("DetA", _now=1.0)
        assert tracker.should_notify

        tracker.record("DetA", _now=20.0)
        assert not tracker.should_notify

        tracker.record("DetA", _now=21.0)
        assert tracker.should_notify

    def test_summary_includes_crash_counts(self) -> None:
        tracker = SlidingWindowCounter(window_seconds=60, threshold=5)
        tracker.record("DetA", _now=0.0)
        tracker.record("DetB", _now=1.0)
        tracker.record("DetA", _now=2.0)
        summary = tracker.summary()
        assert "DetA=2" in summary
        assert "DetB=1" in summary


class TestRunDetectorsCrashHandling:
    def test_crashing_detector_is_skipped_returns_no_fault(self) -> None:
        """A single crash is silently skipped — main loop continues."""
        ctx = _make_detector_context()
        decision = run_detectors(detectors=[_CrashingDetector()], ctx=ctx)
        assert decision.action == ActionType.NONE

    def test_crash_records_in_tracker(self) -> None:
        ctx = _make_detector_context()
        tracker = SlidingWindowCounter(window_seconds=60, threshold=2)
        run_detectors(detectors=[_CrashingDetector()], ctx=ctx, crash_tracker=tracker)
        assert not tracker.should_notify

        run_detectors(detectors=[_CrashingDetector()], ctx=ctx, crash_tracker=tracker)
        assert tracker.should_notify

    def test_crash_before_recovery_still_returns_recovery(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_CrashingDetector(), _RecoveryDetector()],
            ctx=ctx,
        )
        assert decision.action == ActionType.ENTER_RECOVERY

    def test_all_crash_returns_no_fault(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_CrashingDetector(), _CrashingDetector()],
            ctx=ctx,
        )
        assert decision.action == ActionType.NONE

    def test_all_passing_returns_no_fault(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_PassingDetector(), _PassingDetector()],
            ctx=ctx,
        )
        assert decision.action == ActionType.NONE


class TestHandleNotifyHuman:
    @pytest.mark.anyio
    async def test_sends_notification(self) -> None:
        from tests.fast.utils.ft.conftest import FakeNotifier

        notifier = FakeNotifier()
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="something bad happened",
            trigger=TriggerType.MISC,
        )

        await handle_notify_human(decision=decision, notifier=notifier)

        assert len(notifier.calls) == 1
        title, content, _ = notifier.calls[0]
        assert title == "Fault Alert"
        assert "something bad happened" in content

    @pytest.mark.anyio
    async def test_none_notifier_does_not_crash(self) -> None:
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="should not crash",
            trigger=TriggerType.MISC,
        )

        await handle_notify_human(decision=decision, notifier=None)
