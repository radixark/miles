from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.controller.main_state_machine.helpers import (
    MainContext,
    collect_critical_bad_nodes,
    get_known_bad_nodes,
    notify_too_many_bad_nodes,
)
from miles.utils.ft.controller.main_state_machine.states import DetectingAnomaly, MainState, Recovering
from miles.utils.ft.controller.recovery.recovery_stepper.states import NotifyHumans, RealtimeChecks, RecoveryDone

logger = logging.getLogger(__name__)


class RecoveringHandler:
    async def step(self, state: Recovering, ctx: MainContext) -> MainState | None:
        ret = await self._check_new_bad_nodes(state=state, ctx=ctx)
        if ret is not None:
            return ret
        return await self._advance_recovery(state=state, ctx=ctx)

    async def _check_new_bad_nodes(
        self,
        *,
        state: Recovering,
        ctx: MainContext,
    ) -> MainState | None:
        curr_bad_nodes = collect_critical_bad_nodes(
            detectors=ctx.detectors,
            tick_detector_context=ctx.detector_context,
        )
        if len(curr_bad_nodes) >= ctx.max_simultaneous_bad_nodes:
            await notify_too_many_bad_nodes(
                bad_node_count=len(curr_bad_nodes),
                max_simultaneous_bad_nodes=ctx.max_simultaneous_bad_nodes,
                trigger=state.trigger,
                context_str="Critical detectors reported during recovery",
                notifier=ctx.notifier,
            )
            return None

        known_bad = set(get_known_bad_nodes(state.recovery))
        truly_new = curr_bad_nodes - known_bad
        if truly_new:
            all_bad = sorted(known_bad | curr_bad_nodes)
            return Recovering(
                recovery=RealtimeChecks(pre_identified_bad_nodes=all_bad),
                trigger=state.trigger,
                recovery_start_time=datetime.now(timezone.utc),
            )
        return None

    async def _advance_recovery(
        self,
        *,
        state: Recovering,
        ctx: MainContext,
    ) -> MainState | None:
        try:
            recovery_ctx = ctx.recovery_context_factory(
                state.trigger,
                state.recovery_start_time,
            )
            new_recovery = await ctx.recovery_stepper(state.recovery, recovery_ctx)
        except Exception:
            logger.error("Recovery stepper raised exception", exc_info=True)
            new_recovery = NotifyHumans(state_before=type(state.recovery).__name__)

        if new_recovery is None:
            return None
        if isinstance(new_recovery, RecoveryDone):
            self._report_recovery_duration(state=state, ctx=ctx)
            return DetectingAnomaly()
        return Recovering(
            recovery=new_recovery,
            trigger=state.trigger,
            recovery_start_time=state.recovery_start_time,
        )

    def _report_recovery_duration(self, *, state: Recovering, ctx: MainContext) -> None:
        if ctx.on_recovery_duration is not None:
            duration = (datetime.now(timezone.utc) - state.recovery_start_time).total_seconds()
            ctx.on_recovery_duration(duration)
