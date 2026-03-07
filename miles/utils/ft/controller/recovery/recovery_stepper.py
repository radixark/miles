from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ConfigDict

from miles.utils.ft.controller.recovery.alert_checker import AlertChecker
from miles.utils.ft.controller.recovery.helpers import safe_notify
from miles.utils.ft.controller.recovery.restart_stepper import (
    Evicting,
    RestartDone,
    RestartFailed,
    RestartStepper,
    RestartState,
    StoppingAndRestarting,
)
from miles.utils.ft.utils.state_machine import StateMachineStepper
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import TriggerType, unique_node_ids
from miles.utils.ft.protocols.platform import (
    DiagnosticOrchestratorProtocol,
    NotificationProtocol,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State classes
# ---------------------------------------------------------------------------


class RecoveryState(FtBaseModel):
    model_config = ConfigDict(frozen=True)


class RealtimeChecks(RecoveryState):
    pre_identified_bad_nodes: list[str] = []


class EvictingAndRestarting(RecoveryState):
    restart: RestartState
    succeed_next_state: RecoveryState
    failed_next_state: RecoveryState


class StopTimeDiagnostics(RecoveryState):
    pass


class NotifyHumans(RecoveryState):
    state_before: str


class RecoveryDone(RecoveryState):
    pass


# ---------------------------------------------------------------------------
# Per-call context (passed from MainStepper each tick)
# ---------------------------------------------------------------------------


@dataclass
class RecoveryContext:
    trigger: TriggerType
    recovery_start_time: datetime


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

RECOVERY_STATE_TO_INT: dict[type[RecoveryState], int] = {
    RealtimeChecks: 1,
    EvictingAndRestarting: 2,
    StopTimeDiagnostics: 4,
    NotifyHumans: 5,
    RecoveryDone: 6,
}

BAD_NODES_CONFIRMED_TYPES: frozenset[type[RecoveryState]] = frozenset({
    EvictingAndRestarting, NotifyHumans, RecoveryDone,
})


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


class RecoveryStepper(StateMachineStepper[RecoveryState, RecoveryContext]):
    def __init__(
        self,
        *,
        alert_checker: AlertChecker,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
        restart_stepper: RestartStepper,
        notifier: NotificationProtocol | None,
        timeout_seconds: int = 1800,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> None:
        self._alert_checker = alert_checker
        self._diagnostic_orchestrator = diagnostic_orchestrator
        self._restart_stepper = restart_stepper
        self._notifier = notifier
        self._timeout_seconds = timeout_seconds
        self._rank_pids_provider = rank_pids_provider
        super().__init__()

    def set_rank_pids_provider(self, provider: Callable[[str], dict[int, int]]) -> None:
        self._rank_pids_provider = provider

    def _build_handlers(self) -> dict:
        return {
            RealtimeChecks: self._handle_realtime_checks,
            EvictingAndRestarting: self._handle_evicting_and_restarting,
            StopTimeDiagnostics: self._handle_stop_time_diagnostics,
            NotifyHumans: self._handle_notify_humans,
        }

    async def __call__(self, state: RecoveryState, context: RecoveryContext) -> RecoveryState | None:
        elapsed = (datetime.now(timezone.utc) - context.recovery_start_time).total_seconds()
        if elapsed > self._timeout_seconds and not isinstance(state, (NotifyHumans, RecoveryDone)):
            return NotifyHumans(state_before=type(state).__name__)
        return await super().__call__(state, context)

    # -- handlers ---------------------------------------------------------

    async def _handle_realtime_checks(self, state: RealtimeChecks, context: RecoveryContext) -> RecoveryState:
        if state.pre_identified_bad_nodes:
            return EvictingAndRestarting(
                restart=Evicting(bad_node_ids=state.pre_identified_bad_nodes),
                succeed_next_state=RecoveryDone(),
                failed_next_state=StopTimeDiagnostics(),
            )

        node_faults = self._alert_checker.check_alerts()
        if not node_faults:
            logger.info("check_alerts_clean trigger=%s", context.trigger)
            return EvictingAndRestarting(
                restart=StoppingAndRestarting(bad_node_ids=[]),
                succeed_next_state=RecoveryDone(),
                failed_next_state=StopTimeDiagnostics(),
            )

        non_ephemeral = [f for f in node_faults if not f.ephemeral]
        if non_ephemeral:
            bad_ids = sorted(unique_node_ids(non_ephemeral))
            logger.info("check_alerts_found bad_nodes=%s", bad_ids)
            return EvictingAndRestarting(
                restart=Evicting(bad_node_ids=bad_ids),
                succeed_next_state=RecoveryDone(),
                failed_next_state=StopTimeDiagnostics(),
            )

        logger.info("check_alerts_ephemeral_only trigger=%s", context.trigger)
        return EvictingAndRestarting(
            restart=StoppingAndRestarting(bad_node_ids=[]),
            succeed_next_state=RecoveryDone(),
            failed_next_state=StopTimeDiagnostics(),
        )

    async def _handle_evicting_and_restarting(
        self, state: EvictingAndRestarting, _context: RecoveryContext,
    ) -> RecoveryState | None:
        new_restart = await self._restart_stepper(state.restart, None)
        if new_restart is None:
            return None
        if isinstance(new_restart, RestartDone):
            return state.succeed_next_state
        if isinstance(new_restart, RestartFailed):
            return state.failed_next_state
        return EvictingAndRestarting(
            restart=new_restart,
            succeed_next_state=state.succeed_next_state,
            failed_next_state=state.failed_next_state,
        )

    async def _handle_stop_time_diagnostics(
        self, state: StopTimeDiagnostics, context: RecoveryContext,
    ) -> RecoveryState:
        result = await self._diagnostic_orchestrator.run_diagnostic_pipeline(
            trigger_reason=context.trigger,
            rank_pids_provider=self._rank_pids_provider,
        )

        if result.bad_node_ids:
            logger.info("diagnosing_found_bad_nodes bad_nodes=%s", result.bad_node_ids)
            return EvictingAndRestarting(
                restart=Evicting(bad_node_ids=result.bad_node_ids),
                succeed_next_state=RecoveryDone(),
                failed_next_state=NotifyHumans(state_before="EvictingAndRestarting"),
            )

        logger.info("diagnosing_all_passed trigger=%s", context.trigger)
        return NotifyHumans(state_before="StopTimeDiagnostics")

    async def _handle_notify_humans(self, state: NotifyHumans, context: RecoveryContext) -> RecoveryState:
        message = (
            f"Recovery requires human intervention. "
            f"trigger={context.trigger} "
            f"state_before={state.state_before}"
        )
        logger.warning("recovery_notify reason=%s", message)
        await safe_notify(self._notifier, title="Recovery Alert", content=message)
        return RecoveryDone()
