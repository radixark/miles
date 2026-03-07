from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
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
    RecoveryDone,
    RecoveryStepper,
    RecoveryState,
)
from miles.utils.ft.controller.recovery.restart_stepper import RestartStepper
from miles.utils.ft.controller.state_machine import StateMachineStepper
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision
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
    trigger: str
    recovery_start_time: datetime


# ---------------------------------------------------------------------------
# Per-tick context (set by FtController before each step)
# ---------------------------------------------------------------------------


@dataclass
class _TickContext:
    job_status: JobStatus
    tick_count: int
    should_run_detectors: bool
    detector_context: DetectorContext | None


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


class MainStepper(StateMachineStepper[MainState]):
    def __init__(
        self,
        *,
        platform_deps: PlatformDeps,
        restart_stepper: RestartStepper,
        recovery_stepper: RecoveryStepper,
        detectors: list[BaseFaultDetector],
        cooldown: SlidingWindowThrottle,
        controller_exporter: object | None = None,
        on_recovery_duration: Callable[[float], None] | None = None,
    ) -> None:
        self._platform_deps = platform_deps
        self._restart_stepper = restart_stepper
        self._recovery_stepper = recovery_stepper
        self._detectors = detectors
        self._cooldown = cooldown
        self._controller_exporter = controller_exporter
        self._on_recovery_duration = on_recovery_duration

        self._tick_context: _TickContext | None = None
        super().__init__()

    def set_tick_context(
        self,
        *,
        job_status: JobStatus,
        tick_count: int,
        should_run_detectors: bool,
        detector_context: DetectorContext | None,
    ) -> None:
        self._tick_context = _TickContext(
            job_status=job_status,
            tick_count=tick_count,
            should_run_detectors=should_run_detectors,
            detector_context=detector_context,
        )

    def _build_handlers(self) -> dict:
        return {
            DetectingAnomaly: self._handle_detecting_anomaly,
            Recovering: self._handle_recovering,
        }

    # -- DetectingAnomaly -------------------------------------------------

    async def _handle_detecting_anomaly(self, state: DetectingAnomaly) -> MainState | None:
        ctx = self._tick_context
        if ctx is None or not ctx.should_run_detectors or ctx.detector_context is None:
            return None

        decision = self._run_detectors(ctx.detector_context)
        if decision.action == ActionType.NONE:
            return None

        if decision.action == ActionType.NOTIFY_HUMAN:
            await handle_notify_human(decision=decision, notifier=self._platform_deps.notifier)
            return None

        assert decision.trigger is not None
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
            trigger=decision.trigger.value,
            recovery_start_time=now,
        )

    # -- Recovering -------------------------------------------------------

    async def _handle_recovering(self, state: Recovering) -> MainState | None:
        ctx = self._tick_context
        if ctx is not None:
            new_bad_nodes = self._collect_critical_bad_nodes(ctx)
            if new_bad_nodes:
                all_bad = list(set(self._get_known_bad_nodes(state.recovery)) | set(new_bad_nodes))
                now = datetime.now(timezone.utc)
                return Recovering(
                    recovery=RealtimeChecks(pre_identified_bad_nodes=all_bad),
                    trigger=state.trigger,
                    recovery_start_time=now,
                )

        from miles.utils.ft.models.fault import TriggerType
        trigger = TriggerType(state.trigger)

        try:
            new_recovery = await self._recovery_stepper.step_with_context(
                state.recovery,
                trigger=trigger,
                recovery_start_time=state.recovery_start_time,
            )
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

    def _collect_critical_bad_nodes(self, ctx: _TickContext) -> set[str]:
        if ctx.detector_context is None:
            return set()
        bad_nodes: set[str] = set()
        for decision in self._run_detectors_raw(ctx.detector_context, critical_only=True):
            if decision.action in (ActionType.MARK_BAD_AND_RESTART, ActionType.ENTER_RECOVERY) and decision.bad_node_ids:
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

    @staticmethod
    def _get_known_bad_nodes(recovery_state: RecoveryState) -> list[str]:
        if isinstance(recovery_state, (EvictingAndRestarting, DirectlyRestarting)):
            return recovery_state.restart.bad_node_ids
        if isinstance(recovery_state, RealtimeChecks):
            return recovery_state.pre_identified_bad_nodes
        return []
