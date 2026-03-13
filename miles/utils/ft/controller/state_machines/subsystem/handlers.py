from __future__ import annotations

import logging
from datetime import datetime, timezone

from miles.utils.ft.controller.state_machines.recovery.models import NotifyHumansSt, RealtimeChecksSt, RecoveryDoneSt
from miles.utils.ft.controller.state_machines.subsystem.models import (
    DetectingAnomalySt,
    RecoveringSt,
    SubsystemContext,
    SubsystemState,
)
from miles.utils.ft.controller.state_machines.subsystem.utils import (
    DetectorResult,
    collect_evictable_bad_nodes,
    handle_notify_human,
    run_detectors,
)
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


class DetectingAnomalyHandler(StateHandler[DetectingAnomalySt, SubsystemContext]):
    async def step(self, state: DetectingAnomalySt, ctx: SubsystemContext) -> SubsystemState | None:
        logger.debug("subsystem_sm: DetectingAnomalyHandler.step tick=%d", ctx.tick_count)
        decision = await self._get_actionable_decision(ctx=ctx)
        if decision is None:
            return None

        if ctx.cooldown.is_throttled():
            logger.warning(
                "subsystem_sm: recovery cooldown throttled for trigger=%s, skipping",
                decision.trigger,
            )
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
        logger.info(
            "subsystem_sm: state transition: old=DetectingAnomalySt, new=RecoveringSt, " "trigger=%s, bad_nodes=%s",
            decision.trigger,
            decision.bad_node_ids,
        )
        return RecoveringSt(
            recovery=RealtimeChecksSt(pre_identified_bad_nodes=decision.bad_node_ids),
            trigger=decision.trigger,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=decision.bad_node_ids,
        )

    async def _get_actionable_decision(self, *, ctx: SubsystemContext) -> Decision | None:
        """Run all detectors, handle NOTIFY_HUMAN as side effects, return merged ENTER_RECOVERY or None."""
        if not ctx.should_run_detectors or ctx.detector_context is None:
            logger.debug("subsystem_sm: _get_actionable_decision: detectors not running this tick")
            return None

        tracker = ctx.detector_crash_tracker
        result = run_detectors(detectors=ctx.detectors, ctx=ctx.detector_context, crash_tracker=tracker)

        if tracker.should_notify:
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=tracker.summary(),
                    trigger=TriggerType.MISC,
                ),
                notifier=ctx.notifier,
            )

        # Step 1: Handle NOTIFY_HUMAN decisions as side effects with dedup
        await self._handle_notify_decisions(result=result, ctx=ctx)

        # Step 2: Return merged ENTER_RECOVERY or None
        decision = result.recovery_decision
        if decision is None:
            logger.debug("subsystem_sm: no recovery decision after detector run")
            return None

        if len(decision.bad_node_ids) > ctx.max_simultaneous_bad_nodes:
            logger.warning(
                "subsystem_sm: too_many_simultaneous_bad_nodes: count=%d, max=%d, abandoning recovery",
                len(decision.bad_node_ids),
                ctx.max_simultaneous_bad_nodes,
            )
            await handle_notify_human(
                decision=Decision(
                    action=ActionType.NOTIFY_HUMAN,
                    reason=(
                        f"too_many_simultaneous_bad_nodes "
                        f"({len(decision.bad_node_ids)} > {ctx.max_simultaneous_bad_nodes}), "
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

    async def _handle_notify_decisions(
        self,
        *,
        result: DetectorResult,
        ctx: SubsystemContext,
    ) -> None:
        for decision in ctx.notify_deduplicator.check_batch(result.notify_decisions):
            await handle_notify_human(decision=decision, notifier=ctx.notifier)


class RecoveringHandler(StateHandler[RecoveringSt, SubsystemContext]):
    async def step(self, state: RecoveringSt, ctx: SubsystemContext) -> SubsystemState | None:
        logger.debug(
            "subsystem_sm: RecoveringHandler.step trigger=%s, recovery_phase=%s",
            state.trigger,
            type(state.recovery).__name__,
        )
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
        if len(new_bad_nodes) > ctx.max_simultaneous_bad_nodes:
            logger.warning(
                "subsystem_sm: too_many_new_bad_nodes during recovery: count=%d, max=%d",
                len(new_bad_nodes),
                ctx.max_simultaneous_bad_nodes,
            )
            known_bad = set(state.known_bad_node_ids)
            all_bad = tuple(sorted(known_bad | new_bad_nodes))
            return RecoveringSt(
                recovery=NotifyHumansSt(
                    state_before=type(state.recovery).__name__,
                    reason="too_many_simultaneous_bad_nodes",
                    bad_node_ids=all_bad,
                ),
                trigger=state.trigger,
                recovery_start_time=state.recovery_start_time,
                known_bad_node_ids=all_bad,
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
            logger.info(
                "subsystem_sm: new bad nodes discovered during recovery: new=%s, all=%s, restarting recovery",
                sorted(truly_new),
                all_bad,
            )
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
        async for _new_recovery in ctx.recovery_stepper(state.recovery, recovery_ctx):
            new_recovery = _new_recovery

        if new_recovery is None:
            logger.debug("subsystem_sm: _advance_recovery: recovery stepper returned None, no transition")
            return None
        if isinstance(new_recovery, RecoveryDoneSt):
            self._report_recovery_duration(state=state, ctx=ctx)
            logger.info(
                "subsystem_sm: state transition: old=RecoveringSt, new=DetectingAnomalySt, trigger=recovery_done"
            )
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
