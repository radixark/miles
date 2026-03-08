from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from pydantic import ConfigDict

from miles.utils.ft.controller.recovery.utils import safe_notify
from miles.utils.ft.controller.recovery.recovery_stepper.states import (
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    RecoveryState,
    StopTimeDiagnostics,
)
from miles.utils.ft.controller.recovery.restart_stepper.handlers import RestartContext
from miles.utils.ft.controller.recovery.restart_stepper.states import RestartDone, RestartFailed, RestartState
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.protocols.platform import DiagnosticOrchestratorProtocol, NotificationProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fat context
# ---------------------------------------------------------------------------


class RecoveryContext(FtBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # per-call
    trigger: TriggerType
    recovery_start_time: datetime

    # deps
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    restart_stepper: Callable[[RestartState, RestartContext], Awaitable[RestartState | None]]
    restart_context: RestartContext
    notifier: NotificationProtocol | None
    timeout_seconds: int
    rank_pids_provider: Callable[[str], dict[int, int]] | None


# ---------------------------------------------------------------------------
# Pre-dispatch: global timeout
# ---------------------------------------------------------------------------


async def recovery_timeout_check(
    state: RecoveryState,
    ctx: RecoveryContext,
) -> RecoveryState | None:
    elapsed = (datetime.now(timezone.utc) - ctx.recovery_start_time).total_seconds()
    if elapsed > ctx.timeout_seconds and not isinstance(state, (NotifyHumans, RecoveryDone)):
        return NotifyHumans(state_before=type(state).__name__)
    return None


# ---------------------------------------------------------------------------
# Handler classes
# ---------------------------------------------------------------------------


class RealtimeChecksHandler:
    async def step(self, state: RealtimeChecks, ctx: RecoveryContext) -> RecoveryState:
        if state.pre_identified_bad_nodes:
            return EvictingAndRestarting.evict_and_restart(
                bad_node_ids=state.pre_identified_bad_nodes,
            )

        logger.info("realtime_checks_clean trigger=%s", ctx.trigger)
        return EvictingAndRestarting.direct_restart()


class EvictingAndRestartingHandler:
    async def step(
        self,
        state: EvictingAndRestarting,
        ctx: RecoveryContext,
    ) -> RecoveryState | None:
        new_restart = await ctx.restart_stepper(state.restart, ctx.restart_context)
        if new_restart is None:
            return None
        if isinstance(new_restart, RestartDone):
            return RecoveryDone()
        if isinstance(new_restart, RestartFailed):
            return state.failed_next_state
        return EvictingAndRestarting(
            restart=new_restart,
            failed_next_state=state.failed_next_state,
        )


class StopTimeDiagnosticsHandler:
    async def step(
        self,
        state: StopTimeDiagnostics,
        ctx: RecoveryContext,
    ) -> RecoveryState:
        result = await ctx.diagnostic_orchestrator.run_diagnostic_pipeline(
            trigger_reason=ctx.trigger,
            rank_pids_provider=ctx.rank_pids_provider,
        )

        if result.bad_node_ids:
            logger.info("diagnosing_found_bad_nodes bad_nodes=%s", result.bad_node_ids)
            return EvictingAndRestarting.evict_and_restart_final(
                bad_node_ids=result.bad_node_ids,
            )

        logger.info("diagnosing_all_passed trigger=%s", ctx.trigger)
        return NotifyHumans(state_before="StopTimeDiagnostics")


class NotifyHumansHandler:
    async def step(self, state: NotifyHumans, ctx: RecoveryContext) -> RecoveryState:
        message = (
            f"Recovery requires human intervention. " f"trigger={ctx.trigger} " f"state_before={state.state_before}"
        )
        logger.warning("recovery_notify reason=%s", message)
        await safe_notify(ctx.notifier, title="Recovery Alert", content=message)
        return RecoveryDone()


class RecoveryDoneHandler:
    async def step(self, state: RecoveryDone, ctx: RecoveryContext) -> None:
        return None
