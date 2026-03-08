from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.controller.actions import handle_notify_human
from miles.utils.ft.controller.main_stepper.states import DetectingAnomaly, MainState, Recovering
from miles.utils.ft.controller.main_stepper.utils import (
    MainContext,
    collect_evictable_bad_nodes,
    get_known_bad_nodes,
    notify_too_many_bad_nodes,
    run_detectors,
)
from miles.utils.ft.controller.recovery.recovery_stepper.states import NotifyHumans, RealtimeChecks, RecoveryDone
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType

logger = logging.getLogger(__name__)


class DetectingAnomalyHandler:
    async def step(self, state: DetectingAnomaly, ctx: MainContext) -> MainState | None:
        decision = await self._get_actionable_decision(ctx=ctx)
        if decision is None:
            return None

        ctx.cooldown.record()
        if ctx.cooldown.is_throttled():
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=f"Recovery cooldown throttled for {decision.trigger}",
                    trigger=decision.trigger,
                ),
                notifier=ctx.notifier,
            )
            return None

        return Recovering(
            recovery=RealtimeChecks(pre_identified_bad_nodes=decision.bad_node_ids),
            trigger=decision.trigger,
            recovery_start_time=datetime.now(timezone.utc),
        )

    async def _get_actionable_decision(self, *, ctx: MainContext) -> Decision | None:
        """Run detectors and return a validated ENTER_RECOVERY decision, or None.

        Handles NONE, NOTIFY_HUMAN, and too-many-bad-nodes cases internally.
        """
        if not ctx.should_run_detectors or ctx.detector_context is None:
            return None

        tracker = ctx.detector_crash_tracker
        decision = run_detectors(detectors=ctx.detectors, ctx=ctx.detector_context, crash_tracker=tracker)

        if tracker.should_notify:
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=tracker.summary(),
                    trigger=TriggerType.MISC,
                ),
                notifier=ctx.notifier,
            )

        if decision.action == ActionType.NONE:
            return None

        if decision.action == ActionType.NOTIFY_HUMAN:
            await handle_notify_human(decision=decision, notifier=ctx.notifier)
            return None

        if len(decision.bad_node_ids) >= ctx.max_simultaneous_bad_nodes:
            await notify_too_many_bad_nodes(
                bad_node_count=len(decision.bad_node_ids),
                max_simultaneous_bad_nodes=ctx.max_simultaneous_bad_nodes,
                trigger=decision.trigger,
                context_str="Detector reported",
                notifier=ctx.notifier,
            )
            return None

        if decision.trigger is None:
            raise ValueError(f"Decision with action={decision.action} has no trigger")
        return decision


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
        new_bad_nodes = collect_evictable_bad_nodes(
            detectors=ctx.detectors,
            tick_detector_context=ctx.detector_context,
            crash_tracker=ctx.detector_crash_tracker,
        )
        if len(new_bad_nodes) >= ctx.max_simultaneous_bad_nodes:
            await notify_too_many_bad_nodes(
                bad_node_count=len(new_bad_nodes),
                max_simultaneous_bad_nodes=ctx.max_simultaneous_bad_nodes,
                trigger=state.trigger,
                context_str="Critical detectors reported during recovery",
                notifier=ctx.notifier,
            )
            return None

        known_bad = set(get_known_bad_nodes(state.recovery))
        truly_new = new_bad_nodes - known_bad
        if truly_new:
            all_bad = sorted(known_bad | new_bad_nodes)
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
