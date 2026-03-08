from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.controller.actions import handle_notify_human
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.recovery.utils import SlidingWindowThrottle
from miles.utils.ft.controller.recovery.recovery_stepper.handlers import RecoveryContext
from miles.utils.ft.controller.recovery.recovery_stepper.states import (
    EvictingAndRestarting,
    RealtimeChecks,
    RecoveryState,
)
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.platform import JobStatus, NotificationProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crash tracker
# ---------------------------------------------------------------------------


class DetectorCrashTracker:
    """Sliding-window tracker for detector crashes.

    Records crashes and fires a one-shot notification when >=threshold
    crashes occur within window_seconds.  Resets after the window clears.
    """

    def __init__(self, *, window_seconds: float = 1800, threshold: int = 5) -> None:
        self._window_seconds = window_seconds
        self._threshold = threshold
        self._crashes: list[tuple[float, str]] = []
        self._notified = False

    def record(self, detector_name: str, *, _now: float | None = None) -> None:
        """Record a detector crash."""
        import time

        now = _now if _now is not None else time.monotonic()
        self._crashes.append((now, detector_name))
        self._prune(now)

    @property
    def should_notify(self) -> bool:
        """True once when threshold is first reached; resets after window clears."""
        if len(self._crashes) >= self._threshold and not self._notified:
            self._notified = True
            return True
        return False

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for _, name in self._crashes:
            counts[name] = counts.get(name, 0) + 1
        parts = [f"{name}={count}" for name, count in sorted(counts.items())]
        return f"{len(self._crashes)} detector crashes in {self._window_seconds}s window: {', '.join(parts)}"

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        self._crashes = [(t, n) for t, n in self._crashes if t >= cutoff]
        if len(self._crashes) < self._threshold:
            self._notified = False


# ---------------------------------------------------------------------------
# Fat context
# ---------------------------------------------------------------------------


class MainContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # per-tick
    job_status: JobStatus
    tick_count: int
    should_run_detectors: bool
    detector_context: DetectorContext | None

    # deps
    notifier: NotificationProtocol | None
    detectors: list[BaseFaultDetector]
    cooldown: SlidingWindowThrottle
    detector_crash_tracker: DetectorCrashTracker
    recovery_stepper: Callable[..., Awaitable[RecoveryState | None]]
    recovery_context_factory: Callable[[TriggerType, datetime], RecoveryContext]
    on_recovery_duration: Callable[[float], None] | None
    max_simultaneous_bad_nodes: int


# ---------------------------------------------------------------------------
# Shared functions
# ---------------------------------------------------------------------------


def get_known_bad_nodes(recovery_state: RecoveryState) -> list[str]:
    if isinstance(recovery_state, EvictingAndRestarting):
        return recovery_state.restart.bad_node_ids
    if isinstance(recovery_state, RealtimeChecks):
        return recovery_state.pre_identified_bad_nodes
    return []


async def notify_too_many_bad_nodes(
    *,
    bad_node_count: int,
    max_simultaneous_bad_nodes: int,
    trigger: TriggerType | None,
    context_str: str,
    notifier: NotificationProtocol | None,
) -> None:
    logger.warning(
        "too_many_bad_nodes count=%d threshold=%d context=%s, likely false positive",
        bad_node_count,
        max_simultaneous_bad_nodes,
        context_str,
    )
    await handle_notify_human(
        decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason=(
                f"{context_str}: {bad_node_count} bad nodes "
                f"(>= {max_simultaneous_bad_nodes}), likely false positive"
            ),
            trigger=trigger,
        ),
        notifier=notifier,
    )


def run_detectors(
    detectors: list[BaseFaultDetector],
    ctx: DetectorContext,
    crash_tracker: DetectorCrashTracker | None = None,
) -> Decision:
    for decision in _run_detectors_raw(detectors=detectors, ctx=ctx, crash_tracker=crash_tracker):
        if decision.action != ActionType.NONE:
            return decision
    return Decision.no_fault(reason="all detectors passed")


def collect_evictable_bad_nodes(
    detectors: list[BaseFaultDetector],
    tick_detector_context: DetectorContext | None,
    crash_tracker: DetectorCrashTracker | None = None,
) -> set[str]:
    if tick_detector_context is None:
        return set()
    bad_nodes: set[str] = set()
    for decision in _run_detectors_raw(detectors=detectors, ctx=tick_detector_context, crash_tracker=crash_tracker):
        if decision.action == ActionType.ENTER_RECOVERY and decision.bad_node_ids:
            bad_nodes.update(decision.bad_node_ids)
    return bad_nodes


def _run_detectors_raw(
    *,
    detectors: list[BaseFaultDetector],
    ctx: DetectorContext,
    crash_tracker: DetectorCrashTracker | None = None,
) -> Iterator[Decision]:
    for detector in detectors:
        try:
            yield detector.evaluate(ctx)
        except Exception:
            logger.error(
                "detector_evaluate_failed detector=%s",
                type(detector).__name__,
                exc_info=True,
            )
            if crash_tracker is not None:
                crash_tracker.record(type(detector).__name__)
