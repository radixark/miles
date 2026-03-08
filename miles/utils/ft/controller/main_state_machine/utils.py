from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime

from pydantic import ConfigDict

from miles.utils.ft.controller.actions import handle_notify_human
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
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


def run_detectors(detectors: list[BaseFaultDetector], ctx: DetectorContext) -> Decision:
    for decision in _run_detectors_raw(detectors=detectors, ctx=ctx):
        if decision.action != ActionType.NONE:
            return decision
    return Decision.no_fault(reason="all detectors passed")


def collect_critical_bad_nodes(
    detectors: list[BaseFaultDetector],
    tick_detector_context: DetectorContext | None,
) -> set[str]:
    if tick_detector_context is None:
        return set()
    bad_nodes: set[str] = set()
    for decision in _run_detectors_raw(
        detectors=detectors,
        ctx=tick_detector_context,
        critical_only=True,
    ):
        if decision.action == ActionType.ENTER_RECOVERY and decision.bad_node_ids:
            bad_nodes.update(decision.bad_node_ids)
    return bad_nodes


def _run_detectors_raw(
    *,
    detectors: list[BaseFaultDetector],
    ctx: DetectorContext,
    critical_only: bool = False,
) -> Iterator[Decision]:
    for detector in detectors:
        if critical_only and not detector.is_critical:
            continue
        try:
            yield detector.evaluate(ctx)
        except Exception:
            logger.error(
                "detector_evaluate_failed detector=%s",
                type(detector).__name__,
                exc_info=True,
            )
