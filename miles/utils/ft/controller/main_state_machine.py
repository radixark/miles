from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ConfigDict

from miles.utils.ft.controller.actions import PlatformDeps, handle_notify_human
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.controller.recovery.recovery_stepper import (
    EvictingAndRestarting,
    DirectlyRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryContext,
    RecoveryDone,
    RecoveryStepper,
    RecoveryState,
)
from miles.utils.ft.controller.recovery.restart_stepper import RestartStepper
from miles.utils.ft.utils.state_machine import StateMachineStepper
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.platform import JobStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State classes
# ---------------------------------------------------------------------------


class MainState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class DetectingAnomaly(MainState):
    pass


class Recovering(MainState):
    recovery: RecoveryState
    trigger: TriggerType
    recovery_start_time: datetime


# ---------------------------------------------------------------------------
# Per-tick context (passed to step() by FtController each tick)
# ---------------------------------------------------------------------------


@dataclass
class TickContext:
    job_status: JobStatus
    tick_count: int
    should_run_detectors: bool
    detector_context: DetectorContext | None


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


class MainStepper(StateMachineStepper[MainState, TickContext]):
    def __init__(
        self,
        *,
        platform_deps: PlatformDeps,
        recovery_stepper: RecoveryStepper,
        detectors: list[BaseFaultDetector],
        cooldown: SlidingWindowThrottle,
        on_recovery_duration: Callable[[float], None] | None = None,
        max_simultaneous_bad_nodes: int = 3,
    ) -> None:
        self._platform_deps = platform_deps
        self._recovery_stepper = recovery_stepper
        self._detectors = detectors
        self._cooldown = cooldown
        self._on_recovery_duration = on_recovery_duration
        self._max_simultaneous_bad_nodes = max_simultaneous_bad_nodes
        super().__init__()

    def _build_handlers(self) -> dict[type, Callable[[MainState, TickContext], Awaitable[MainState | None]]]:
        return {
            DetectingAnomaly: self._handle_detecting_anomaly,
            Recovering: self._handle_recovering,
        }

    # -- DetectingAnomaly -------------------------------------------------

    async def _handle_detecting_anomaly(self, state: DetectingAnomaly, context: TickContext) -> MainState | None:
        if not context.should_run_detectors or context.detector_context is None:
            return None

        decision = self._run_detectors(context.detector_context)
        if decision.action == ActionType.NONE:
            return None

        if decision.action == ActionType.NOTIFY_HUMAN:
            await handle_notify_human(decision=decision, notifier=self._platform_deps.notifier)
            return None

        if len(decision.bad_node_ids) >= self._max_simultaneous_bad_nodes:
            logger.warning(
                "too_many_bad_nodes count=%d threshold=%d, likely false positive",
                len(decision.bad_node_ids),
                self._max_simultaneous_bad_nodes,
            )
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=(
                        f"Detector reported {len(decision.bad_node_ids)} bad nodes "
                        f"(>= {self._max_simultaneous_bad_nodes}), likely false positive"
                    ),
                    trigger=decision.trigger,
                ),
                notifier=self._platform_deps.notifier,
            )
            return None

        if decision.trigger is None:
            raise ValueError(f"Decision with action={decision.action} has no trigger")
        self._cooldown.record(decision.trigger)
        if self._cooldown.is_throttled(decision.trigger):
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=f"Recovery cooldown throttled for {decision.trigger}",
                    trigger=decision.trigger,
                ),
                notifier=self._platform_deps.notifier,
            )
            return None

        now = datetime.now(timezone.utc)
        initial_recovery = RealtimeChecks(pre_identified_bad_nodes=decision.bad_node_ids)
        return Recovering(
            recovery=initial_recovery,
            trigger=decision.trigger,
            recovery_start_time=now,
        )

    # -- Recovering -------------------------------------------------------

    async def _handle_recovering(self, state: Recovering, context: TickContext) -> MainState | None:
        new_bad_nodes = self._collect_critical_bad_nodes(context)
        if len(new_bad_nodes) >= self._max_simultaneous_bad_nodes:
            logger.warning(
                "too_many_dynamic_bad_nodes count=%d threshold=%d, likely false positive",
                len(new_bad_nodes),
                self._max_simultaneous_bad_nodes,
            )
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=(
                        f"Critical detectors reported {len(new_bad_nodes)} new bad nodes "
                        f"during recovery (>= {self._max_simultaneous_bad_nodes}), likely false positive"
                    ),
                    trigger=state.trigger,
                ),
                notifier=self._platform_deps.notifier,
            )
            return DetectingAnomaly()

        if new_bad_nodes:
            all_bad = list(set(get_known_bad_nodes(state.recovery)) | set(new_bad_nodes))
            now = datetime.now(timezone.utc)
            return Recovering(
                recovery=RealtimeChecks(pre_identified_bad_nodes=all_bad),
                trigger=state.trigger,
                recovery_start_time=now,
            )

        try:
            recovery_ctx = RecoveryContext(
                trigger=state.trigger,
                recovery_start_time=state.recovery_start_time,
            )
            new_recovery = await self._recovery_stepper(state.recovery, recovery_ctx)
        except Exception:
            logger.error("Recovery stepper raised exception", exc_info=True)
            new_recovery = NotifyHumans(state_before=type(state.recovery).__name__)

        if new_recovery is None:
            return None
        if isinstance(new_recovery, RecoveryDone):
            self._report_recovery_duration(state)
            return DetectingAnomaly()
        return Recovering(
            recovery=new_recovery,
            trigger=state.trigger,
            recovery_start_time=state.recovery_start_time,
        )

    # -- helpers ----------------------------------------------------------

    def _report_recovery_duration(self, state: Recovering) -> None:
        if self._on_recovery_duration is not None:
            duration = (datetime.now(timezone.utc) - state.recovery_start_time).total_seconds()
            self._on_recovery_duration(duration)

    def _run_detectors(self, ctx: DetectorContext) -> Decision:
        for decision in self._run_detectors_raw(ctx):
            if decision.action != ActionType.NONE:
                return decision
        return Decision.no_fault(reason="all detectors passed")

    def _collect_critical_bad_nodes(self, ctx: TickContext) -> set[str]:
        if ctx.detector_context is None:
            return set()
        bad_nodes: set[str] = set()
        for decision in self._run_detectors_raw(ctx.detector_context, critical_only=True):
            if decision.action == ActionType.ENTER_RECOVERY and decision.bad_node_ids:
                bad_nodes.update(decision.bad_node_ids)
        return bad_nodes

    def _run_detectors_raw(
        self,
        ctx: DetectorContext,
        *,
        critical_only: bool = False,
    ) -> Iterator[Decision]:
        for detector in self._detectors:
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


def get_known_bad_nodes(recovery_state: RecoveryState) -> list[str]:
    if isinstance(recovery_state, (EvictingAndRestarting, DirectlyRestarting)):
        return recovery_state.restart.bad_node_ids
    if isinstance(recovery_state, RealtimeChecks):
        return recovery_state.pre_identified_bad_nodes
    return []
