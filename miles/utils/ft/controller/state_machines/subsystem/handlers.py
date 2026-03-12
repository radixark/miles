from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    SubsystemContext,
    SubsystemState,
    RecoveringSt,
)
from miles.utils.ft.controller.state_machines.subsystem.utils import (
    collect_evictable_bad_nodes,
    handle_notify_human,
    run_detectors,
)
from miles.utils.ft.controller.state_machines.recovery.models import (
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryDoneSt,
)
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


class DetectingAnomalyHandler(StateHandler[DetectingAnomalySt, SubsystemContext]):
    async def step(self, state: DetectingAnomalySt, ctx: SubsystemContext) -> SubsystemState | None:
        decision = await self._get_actionable_decision(ctx=ctx)
        if decision is None:
            return None

        if ctx.cooldown.is_throttled():
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    # NOTE: notify reason strings here are operator-facing hints only. Exact
                    # wording is intentionally non-contractual and should not drive refactors.
                    reason=f"Recovery cooldown throttled for {decision.trigger}",
                    trigger=decision.trigger,
                ),
                notifier=ctx.notifier,
            )
            return None

        ctx.cooldown.record()
        return RecoveringSt(
            recovery=RealtimeChecksSt(pre_identified_bad_nodes=decision.bad_node_ids),
            trigger=decision.trigger,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=decision.bad_node_ids,
        )

    async def _get_actionable_decision(self, *, ctx: SubsystemContext) -> Decision | None:
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
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=(
                        f"too_many_simultaneous_bad_nodes "
                        f"({len(decision.bad_node_ids)} >= {ctx.max_simultaneous_bad_nodes}), "
                        f"likely false positive"
                    ),
                    trigger=decision.trigger,
                ),
                notifier=ctx.notifier,
            )
            return None

        if decision.trigger is None:
            raise ValueError(f"Decision with action={decision.action} has no trigger")
        return decision


class RecoveringHandler(StateHandler[RecoveringSt, SubsystemContext]):
    async def step(self, state: RecoveringSt, ctx: SubsystemContext) -> SubsystemState | None:
        ret = await self._check_new_bad_nodes(state=state, ctx=ctx)
        if ret is not None:
            return ret
        return await self._advance_recovery(state=state, ctx=ctx)

    async def _check_new_bad_nodes(
        self,
        *,
        state: RecoveringSt,
        ctx: SubsystemContext,
    ) -> SubsystemState | None:
        new_bad_nodes = collect_evictable_bad_nodes(
            detectors=ctx.detectors,
            tick_detector_context=ctx.detector_context,
            crash_tracker=ctx.detector_crash_tracker,
        )
        if len(new_bad_nodes) >= ctx.max_simultaneous_bad_nodes:
            return RecoveringSt(
                recovery=NotifyHumansSt(
                    state_before=type(state.recovery).__name__,
                    reason="too_many_simultaneous_bad_nodes",
                ),
                trigger=state.trigger,
                recovery_start_time=state.recovery_start_time,
                known_bad_node_ids=state.known_bad_node_ids,
            )

        known_bad = set(state.known_bad_node_ids)
        truly_new = new_bad_nodes - known_bad
        if truly_new:
            # Restart recovery from RealtimeChecksSt with the combined bad-node
            # set. This intentionally discards in-progress diagnostics/eviction:
            # the new bad nodes change the fault scope, so previous partial work
            # may be based on incomplete information. Re-entering RealtimeChecks
            # ensures the full set goes through the evict→restart→monitor pipeline.
            all_bad = tuple(sorted(known_bad | new_bad_nodes))
            return RecoveringSt(
                recovery=RealtimeChecksSt(pre_identified_bad_nodes=all_bad),
                trigger=state.trigger,
                recovery_start_time=state.recovery_start_time,
                known_bad_node_ids=all_bad,
            )
        return None

    async def _advance_recovery(
        self,
        *,
        state: RecoveringSt,
        ctx: SubsystemContext,
    ) -> SubsystemState | None:
        recovery_ctx = ctx.recovery_context_factory(
            state.trigger,
            state.recovery_start_time,
        )
        new_recovery = None
        async for new_recovery in ctx.recovery_stepper(state.recovery, recovery_ctx):
            pass

        if new_recovery is None:
            return None
        if isinstance(new_recovery, RecoveryDoneSt):
            self._report_recovery_duration(state=state, ctx=ctx)
            return DetectingAnomalySt()
        return RecoveringSt(
            recovery=new_recovery,
            trigger=state.trigger,
            recovery_start_time=state.recovery_start_time,
            known_bad_node_ids=state.known_bad_node_ids,
        )

    def _report_recovery_duration(self, *, state: RecoveringSt, ctx: SubsystemContext) -> None:
        if ctx.on_recovery_duration is not None:
            duration = (datetime.now(timezone.utc) - state.recovery_start_time).total_seconds()
            ctx.on_recovery_duration(duration)


