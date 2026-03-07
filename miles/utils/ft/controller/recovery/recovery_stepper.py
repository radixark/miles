from __future__ import annotations

import logging
from collections.abc import Callable
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
from miles.utils.ft.controller.state_machine import StateMachineStepper
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
    is_final_attempt: bool = False


class DirectlyRestarting(RecoveryState):
    restart: RestartState


class StopTimeDiagnostics(RecoveryState):
    pass


class NotifyHumans(RecoveryState):
    state_before: str


class RecoveryDone(RecoveryState):
    pass


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

RECOVERY_STATE_TO_INT: dict[type[RecoveryState], int] = {
    RealtimeChecks: 1,
    EvictingAndRestarting: 2,
    DirectlyRestarting: 3,
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


class RecoveryStepper(StateMachineStepper[RecoveryState]):
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

        self._call_trigger: TriggerType = TriggerType.MISC
        self._call_recovery_start_time: datetime = datetime.now(timezone.utc)
        super().__init__()

    def set_rank_pids_provider(self, provider: Callable[[str], dict[int, int]]) -> None:
        self._rank_pids_provider = provider

    def _build_handlers(self) -> dict:
        return {
            RealtimeChecks: self._handle_realtime_checks,
            EvictingAndRestarting: self._handle_evicting_and_restarting,
            DirectlyRestarting: self._handle_directly_restarting,
            StopTimeDiagnostics: self._handle_stop_time_diagnostics,
            NotifyHumans: self._handle_notify_humans,
        }

    async def step_with_context(
        self,
        state: RecoveryState,
        *,
        trigger: TriggerType,
        recovery_start_time: datetime,
    ) -> RecoveryState | None:
        """Called by MainStepper each tick. trigger and recovery_start_time come from Recovering state."""
        self._call_trigger = trigger
        self._call_recovery_start_time = recovery_start_time

        elapsed = (datetime.now(timezone.utc) - recovery_start_time).total_seconds()
        if elapsed > self._timeout_seconds and not isinstance(state, (NotifyHumans, RecoveryDone)):
            return NotifyHumans(state_before=type(state).__name__)
        return await super().__call__(state)

    # -- handlers ---------------------------------------------------------

    async def _handle_realtime_checks(self, state: RealtimeChecks) -> RecoveryState:
        if state.pre_identified_bad_nodes:
            return EvictingAndRestarting(
                restart=Evicting(bad_node_ids=state.pre_identified_bad_nodes),
                is_final_attempt=False,
            )

        node_faults = self._alert_checker.check_alerts()
        if not node_faults:
            logger.info("check_alerts_clean trigger=%s", self._call_trigger)
            return DirectlyRestarting(restart=StoppingAndRestarting(bad_node_ids=[]))

        non_ephemeral = [f for f in node_faults if not f.ephemeral]
        if non_ephemeral:
            bad_ids = sorted(unique_node_ids(non_ephemeral))
            logger.info("check_alerts_found bad_nodes=%s", bad_ids)
            return EvictingAndRestarting(
                restart=Evicting(bad_node_ids=bad_ids),
                is_final_attempt=False,
            )

        logger.info("check_alerts_ephemeral_only trigger=%s", self._call_trigger)
        return DirectlyRestarting(restart=StoppingAndRestarting(bad_node_ids=[]))

    async def _handle_evicting_and_restarting(
        self, state: EvictingAndRestarting,
    ) -> RecoveryState | None:
        new_restart = await self._restart_stepper(state.restart)
        if new_restart is None:
            return None
        if isinstance(new_restart, RestartDone):
            return RecoveryDone()
        if isinstance(new_restart, RestartFailed):
            if not state.is_final_attempt:
                return StopTimeDiagnostics()
            return NotifyHumans(state_before="EvictingAndRestarting")
        return EvictingAndRestarting(
            restart=new_restart, is_final_attempt=state.is_final_attempt,
        )

    async def _handle_directly_restarting(
        self, state: DirectlyRestarting,
    ) -> RecoveryState | None:
        new_restart = await self._restart_stepper(state.restart)
        if new_restart is None:
            return None
        if isinstance(new_restart, RestartDone):
            return RecoveryDone()
        if isinstance(new_restart, RestartFailed):
            return StopTimeDiagnostics()
        return DirectlyRestarting(restart=new_restart)

    async def _handle_stop_time_diagnostics(
        self, state: StopTimeDiagnostics,
    ) -> RecoveryState:
        result = await self._diagnostic_orchestrator.run_diagnostic_pipeline(
            trigger_reason=self._call_trigger,
            rank_pids_provider=self._rank_pids_provider,
        )

        if result.bad_node_ids:
            logger.info("diagnosing_found_bad_nodes bad_nodes=%s", result.bad_node_ids)
            return EvictingAndRestarting(
                restart=Evicting(bad_node_ids=result.bad_node_ids),
                is_final_attempt=True,
            )

        logger.info("diagnosing_all_passed trigger=%s", self._call_trigger)
        return NotifyHumans(state_before="StopTimeDiagnostics")

    async def _handle_notify_humans(self, state: NotifyHumans) -> RecoveryState:
        message = (
            f"Recovery requires human intervention. "
            f"trigger={self._call_trigger} "
            f"state_before={state.state_before}"
        )
        logger.warning("recovery_notify reason=%s", message)
        await safe_notify(self._notifier, title="Recovery Alert", content=message)
        return RecoveryDone()
