from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from miles.utils.ft.adapters.types import NotifierProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter

logger = logging.getLogger(__name__)


async def handle_notify_human(
    decision: Decision,
    notifier: NotifierProtocol | None,
) -> None:
    logger.warning(
        "decision_notify_human reason=%s",
        decision.reason,
    )
    await safe_notify(
        notifier,
        title="Fault Alert",
        content=decision.reason,
    )


async def notify_too_many_bad_nodes(
    *,
    bad_node_count: int,
    max_simultaneous_bad_nodes: int,
    trigger: TriggerType | None,
    context_str: str,
    notifier: NotifierProtocol | None,
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


@dataclass
class DetectorResult:
    recovery_decision: Decision | None = None
    notify_decisions: list[Decision] = field(default_factory=list)


def run_detectors(
    detectors: list[BaseFaultDetector],
    ctx: DetectorContext,
    crash_tracker: SlidingWindowCounter | None = None,
) -> DetectorResult:
    """Run all detectors and merge decisions by priority.

    ENTER_RECOVERY decisions are merged: bad_node_ids union, highest-priority
    trigger wins (TriggerType enum order), reasons concatenated.
    NOTIFY_HUMAN decisions are collected as side effects for the caller to
    send (with dedup support from issue 18).
    """
    recovery_decisions: list[Decision] = []
    notify_decisions: list[Decision] = []

    for decision in _run_detectors_raw(detectors=detectors, ctx=ctx, crash_tracker=crash_tracker):
        if decision.action == ActionType.ENTER_RECOVERY:
            recovery_decisions.append(decision)
        elif decision.action == ActionType.NOTIFY_HUMAN:
            notify_decisions.append(decision)

    merged_recovery = _merge_recovery_decisions(recovery_decisions)
    return DetectorResult(
        recovery_decision=merged_recovery,
        notify_decisions=notify_decisions,
    )


def collect_evictable_bad_nodes(
    detectors: list[BaseFaultDetector],
    tick_detector_context: DetectorContext | None,
    crash_tracker: SlidingWindowCounter | None = None,
) -> set[str]:
    if tick_detector_context is None:
        return set()
    bad_nodes: set[str] = set()
    for decision in _run_detectors_raw(detectors=detectors, ctx=tick_detector_context, crash_tracker=crash_tracker):
        if decision.action == ActionType.ENTER_RECOVERY and decision.bad_node_ids:
            bad_nodes.update(decision.bad_node_ids)
    return bad_nodes


def _merge_recovery_decisions(decisions: list[Decision]) -> Decision | None:
    """Merge multiple ENTER_RECOVERY decisions into one.

    - bad_node_ids: sorted union
    - trigger: highest priority (smallest TriggerType enum index)
    - reason: sorted and joined (stable across ticks)
    """
    if not decisions:
        return None

    if len(decisions) == 1:
        return decisions[0]

    all_bad_nodes: set[str] = set()
    reasons: list[str] = []
    best_trigger: TriggerType | None = None
    trigger_members = list(TriggerType)

    for decision in decisions:
        all_bad_nodes.update(decision.bad_node_ids)
        reasons.append(decision.reason)
        if decision.trigger is not None:
            if best_trigger is None:
                best_trigger = decision.trigger
            elif trigger_members.index(decision.trigger) < trigger_members.index(best_trigger):
                best_trigger = decision.trigger

    return Decision(
        action=ActionType.ENTER_RECOVERY,
        bad_node_ids=sorted(all_bad_nodes),
        reason="; ".join(sorted(reasons)),
        trigger=best_trigger,
    )


def _run_detectors_raw(
    *,
    detectors: list[BaseFaultDetector],
    ctx: DetectorContext,
    crash_tracker: SlidingWindowCounter | None = None,
) -> Iterator[Decision]:
    for detector in detectors:
        try:
            yield detector.evaluate(ctx)
        except Exception:
            # Intentionally skip this detector (no Decision yielded) — a single
            # crash should not block other detectors or trigger recovery.
            # Observability: logger.error (per-crash traceback) + crash_tracker
            # accumulation → one-shot NOTIFY_HUMAN when threshold is reached.
            logger.error(
                "detector_evaluate_failed detector=%s",
                type(detector).__name__,
                exc_info=True,
            )
            if crash_tracker is not None:
                crash_tracker.record(type(detector).__name__)
