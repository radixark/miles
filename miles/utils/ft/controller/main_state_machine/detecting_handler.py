from __future__ import annotations

from datetime import datetime, timezone

from miles.utils.ft.controller.actions import handle_notify_human
from miles.utils.ft.controller.main_state_machine.states import DetectingAnomaly, MainState, Recovering
from miles.utils.ft.controller.main_state_machine.utils import MainContext, notify_too_many_bad_nodes, run_detectors
from miles.utils.ft.controller.recovery.recovery_stepper.states import RealtimeChecks
from miles.utils.ft.models.fault import ActionType, Decision


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

        decision = run_detectors(detectors=ctx.detectors, ctx=ctx.detector_context)
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
