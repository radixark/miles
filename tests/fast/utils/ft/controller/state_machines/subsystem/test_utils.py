"""Tests for detector crash tracking via SlidingWindowCounter and run_detectors."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.state_machines.subsystem.models import RecoveringSt
from miles.utils.ft.controller.state_machines.subsystem.utils import (
    collect_evictable_bad_nodes,
    handle_notify_human,
    run_detectors,
)
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter


def _make_detector_context() -> DetectorContext:
    return DetectorContext(
        metric_store=MagicMock(),
        active_node_ids={"node-0"},
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
        t = time.monotonic()
        tracker = SlidingWindowCounter(window_seconds=60, threshold=3)
        tracker.record("DetA", _now=t)
        tracker.record("DetB", _now=t + 1.0)
        assert not tracker.should_notify

    def test_reaching_threshold_notifies_once(self) -> None:
        t = time.monotonic()
        tracker = SlidingWindowCounter(window_seconds=60, threshold=3)
        tracker.record("DetA", _now=t)
        tracker.record("DetA", _now=t + 1.0)
        tracker.record("DetB", _now=t + 2.0)
        assert tracker.should_notify
        assert not tracker.should_notify

    def test_window_expiry_resets_notification(self) -> None:
        t = time.monotonic()
        tracker = SlidingWindowCounter(window_seconds=10, threshold=2)
        tracker.record("DetA", _now=t)
        tracker.record("DetA", _now=t + 1.0)
        assert tracker.should_notify

        tracker.record("DetA", _now=t + 20.0)
        assert not tracker.should_notify

        tracker.record("DetA", _now=t + 21.0)
        assert tracker.should_notify

    def test_summary_includes_crash_counts(self) -> None:
        t = time.monotonic()
        tracker = SlidingWindowCounter(window_seconds=60, threshold=5)
        tracker.record("DetA", _now=t)
        tracker.record("DetB", _now=t + 1.0)
        tracker.record("DetA", _now=t + 2.0)
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


# ---------------------------------------------------------------------------
# P1 item 13: collect_evictable_bad_nodes()
# ---------------------------------------------------------------------------


class _NotifyHumanDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="notify human",
            trigger=TriggerType.MISC,
        )


class _RecoveryDetectorWithNodes(BaseFaultDetector):
    def __init__(self, bad_node_ids: list[str]) -> None:
        self._bad_node_ids = bad_node_ids

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=self._bad_node_ids,
            reason="found bad nodes",
            trigger=TriggerType.HARDWARE,
        )


class TestCollectEvictableBadNodes:
    def test_returns_empty_when_context_is_none(self) -> None:
        result = collect_evictable_bad_nodes(
            detectors=[_RecoveryDetector()],
            tick_detector_context=None,
        )
        assert result == set()

    def test_accumulates_bad_nodes_across_multiple_enter_recovery(self) -> None:
        ctx = DetectorContext(
            metric_store=MagicMock(),
            active_node_ids={"node-0", "node-1", "node-2"},
            job_status=JobStatus.RUNNING,
        )
        result = collect_evictable_bad_nodes(
            detectors=[
                _RecoveryDetectorWithNodes(["node-0"]),
                _RecoveryDetectorWithNodes(["node-1", "node-2"]),
            ],
            tick_detector_context=ctx,
        )
        assert result == {"node-0", "node-1", "node-2"}

    def test_ignores_notify_human_decisions(self) -> None:
        ctx = _make_detector_context()
        result = collect_evictable_bad_nodes(
            detectors=[_NotifyHumanDetector(), _RecoveryDetectorWithNodes(["node-0"])],
            tick_detector_context=ctx,
        )
        assert result == {"node-0"}

    def test_all_passing_returns_empty(self) -> None:
        ctx = _make_detector_context()
        result = collect_evictable_bad_nodes(
            detectors=[_PassingDetector()],
            tick_detector_context=ctx,
        )
        assert result == set()

    def test_enter_recovery_without_bad_node_ids_ignored(self) -> None:
        """ENTER_RECOVERY with no bad_node_ids should not contribute to result."""
        ctx = _make_detector_context()

        class _NoNodeRecovery(BaseFaultDetector):
            def _evaluate_raw(self, ctx_: DetectorContext) -> Decision:
                return Decision(
                    action=ActionType.ENTER_RECOVERY,
                    reason="recovery without specific nodes",
                    trigger=TriggerType.CRASH,
                )

        result = collect_evictable_bad_nodes(
            detectors=[_NoNodeRecovery()],
            tick_detector_context=ctx,
        )
        assert result == set()


class TestKnownBadNodeIds:
    def test_known_bad_node_ids_available_during_diagnostics(self) -> None:
        """2.4: bad nodes must remain visible even when inner state is StopTimeDiagnosticsSt."""
        from miles.utils.ft.controller.state_machines.recovery.models import StopTimeDiagnosticsSt

        state = RecoveringSt(
            recovery=StopTimeDiagnosticsSt(),
            trigger="crash",
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=["node-0", "node-1"],
        )
        assert state.known_bad_node_ids == ["node-0", "node-1"]

    def test_known_bad_node_ids_defaults_to_empty(self) -> None:
        from miles.utils.ft.controller.state_machines.recovery.models import NotifyHumansSt

        state = RecoveringSt(
            recovery=NotifyHumansSt(state_before="test"),
            trigger="crash",
            recovery_start_time=datetime.now(timezone.utc),
        )
        assert state.known_bad_node_ids == []
