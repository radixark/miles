from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import ClusterExecutorProtocol
from miles.utils.ft.controller.diagnostics.executors import StackTraceClusterExecutor
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryContext,
    RecoveryDoneSt,
    RecoveryState,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.restart.models import RestartDoneSt, RestartFailedSt
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.controller.types import TriggerType
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


async def recovery_timeout_check(
    state: RecoveryState,
    ctx: RecoveryContext,
) -> RecoveryState | None:
    elapsed = (datetime.now(timezone.utc) - ctx.recovery_start_time).total_seconds()
    if elapsed > ctx.timeout_seconds and not isinstance(state, (NotifyHumansSt, RecoveryDoneSt)):
        return NotifyHumansSt(state_before=type(state).__name__)
    return None


# ---------------------------------------------------------------------------
# Handler classes
# ---------------------------------------------------------------------------


class RealtimeChecksHandler(StateHandler[RealtimeChecksSt, RecoveryContext]):
    async def step(self, state: RealtimeChecksSt, ctx: RecoveryContext) -> RecoveryState:
        if state.pre_identified_bad_nodes:
            return EvictingAndRestartingSt.evict_and_restart_next_stop_time_diag(
                bad_node_ids=state.pre_identified_bad_nodes,
            )

        logger.info("realtime_checks_clean trigger=%s", ctx.trigger)
        return EvictingAndRestartingSt.direct_restart()


class EvictingAndRestartingHandler(StateHandler[EvictingAndRestartingSt, RecoveryContext]):
    async def step(
        self,
        state: EvictingAndRestartingSt,
        ctx: RecoveryContext,
    ) -> RecoveryState | None:
        new_restart = None
        async for new_restart in ctx.restart_stepper(state.restart, ctx.restart_context):
            pass
        if new_restart is None:
            return None
        if isinstance(new_restart, RestartDoneSt):
            return RecoveryDoneSt()
        if isinstance(new_restart, RestartFailedSt):
            return state.failed_next_state
        return EvictingAndRestartingSt(
            restart=new_restart,
            failed_next_state=state.failed_next_state,
        )


class StopTimeDiagnosticsHandler(StateHandler[StopTimeDiagnosticsSt, RecoveryContext]):
    async def step(
        self,
        state: StopTimeDiagnosticsSt,
        ctx: RecoveryContext,
    ) -> RecoveryState:
        pre_executors: list[ClusterExecutorProtocol] = []
        if ctx.trigger == TriggerType.HANG and ctx.rank_pids_provider is not None:
            pre_executors.append(StackTraceClusterExecutor(rank_pids_provider=ctx.rank_pids_provider))

        result = await ctx.diagnostic_orchestrator.run_diagnostic_pipeline(
            pre_executors=pre_executors or None,
        )

        if result.bad_node_ids:
            logger.info("diagnosing_found_bad_nodes bad_nodes=%s", result.bad_node_ids)
            return EvictingAndRestartingSt.evict_and_restart_final(
                bad_node_ids=result.bad_node_ids,
            )

        logger.info("diagnosing_all_passed trigger=%s", ctx.trigger)
        return NotifyHumansSt(state_before="StopTimeDiagnostics")


class NotifyHumansHandler(StateHandler[NotifyHumansSt, RecoveryContext]):
    async def step(self, state: NotifyHumansSt, ctx: RecoveryContext) -> RecoveryState:
        message = (
            f"Recovery requires human intervention. " f"trigger={ctx.trigger} " f"state_before={state.state_before}"
        )
        logger.warning("recovery_notify reason=%s", message)
        await safe_notify(ctx.notifier, title="Recovery Alert", content=message)
        return RecoveryDoneSt()
