from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.restart.models import (
    Evicting,
    MonitoringProgress,
    RestartContext,
    RestartDone,
    RestartFailed,
    RestartState,
    StoppingAndRestarting,
)
from miles.utils.ft.controller.state_machines.restart.utils import (
    get_already_bad_nodes,
    retry_mark_node_bad,
    stop_and_submit,
)
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)

_WANDB_ITERATION_METRIC = "iteration"
_PENDING_TIMEOUT_SECONDS: int = 300


def iteration_progress(state: MonitoringProgress, mini_wandb: MiniWandb) -> int:
    current_iteration = mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
    if current_iteration is None or not math.isfinite(current_iteration):
        return 0
    raw = int(current_iteration) - state.base_iteration
    if raw < 0:
        logger.warning(
            "iteration_progress_negative current=%d base=%d",
            int(current_iteration),
            state.base_iteration,
        )
        return 0
    return raw


# ---------------------------------------------------------------------------
# Handler classes
# ---------------------------------------------------------------------------


class EvictingHandler(StateHandler[Evicting, RestartContext]):
    async def step(self, state: Evicting, ctx: RestartContext) -> RestartState:
        try:
            already_bad = await get_already_bad_nodes(ctx.node_manager)
        except Exception:
            # mark_node_bad is idempotent, so re-marking is harmless
            logger.warning("get_already_bad_nodes_failed, proceeding with empty set", exc_info=True)
            already_bad = set()

        nodes_to_mark = [n for n in state.bad_node_ids if n not in already_bad]

        for node_id in nodes_to_mark:
            result = await retry_mark_node_bad(
                ctx.node_manager,
                node_id=node_id,
                reason="recovery eviction",
            )
            if not result.ok:
                return RestartFailed(bad_node_ids=state.bad_node_ids)

        await safe_notify(
            ctx.notifier,
            title="Nodes evicted",
            content=f"Evicted: {nodes_to_mark}",
            severity="warning",
        )

        return StoppingAndRestarting(bad_node_ids=state.bad_node_ids)


class StoppingAndRestartingHandler(StateHandler[StoppingAndRestarting, RestartContext]):
    async def step(
        self,
        state: StoppingAndRestarting,
        ctx: RestartContext,
    ) -> RestartState | None:
        if not state.submitted:
            return await self._submit(state=state, ctx=ctx)
        return await self._poll(state=state, ctx=ctx)

    async def _submit(
        self,
        *,
        state: StoppingAndRestarting,
        ctx: RestartContext,
    ) -> RestartState | None:
        success = await stop_and_submit(
            ctx.training_job,
            on_new_run=ctx.on_new_run,
        )
        if not success:
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        return StoppingAndRestarting(
            bad_node_ids=state.bad_node_ids,
            submitted=True,
            submit_time=datetime.now(timezone.utc),
        )

    async def _poll(
        self,
        *,
        state: StoppingAndRestarting,
        ctx: RestartContext,
    ) -> RestartState | None:
        status = await ctx.training_job.get_training_status()

        if status == JobStatus.RUNNING:
            current_iter = ctx.mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
            base = int(current_iter) if current_iter is not None and math.isfinite(current_iter) else 0
            return MonitoringProgress(
                bad_node_ids=state.bad_node_ids,
                start_time=datetime.now(timezone.utc),
                base_iteration=base,
            )

        if status == JobStatus.FAILED:
            logger.warning("restart_job_immediately_failed")
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        if state.submit_time is not None:
            elapsed = (datetime.now(timezone.utc) - state.submit_time).total_seconds()
            if elapsed > _PENDING_TIMEOUT_SECONDS:
                logger.warning("restart_pending_timeout elapsed=%.0f", elapsed)
                return RestartFailed(bad_node_ids=state.bad_node_ids)

        return None


class MonitoringProgressHandler(StateHandler[MonitoringProgress, RestartContext]):
    async def step(
        self,
        state: MonitoringProgress,
        ctx: RestartContext,
    ) -> RestartState | None:
        status = await ctx.training_job.get_training_status()

        if status == JobStatus.FAILED:
            logger.warning("monitoring_training_failed")
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        progress = iteration_progress(state=state, mini_wandb=ctx.mini_wandb)

        if status == JobStatus.RUNNING and progress >= ctx.monitoring_success_iterations:
            logger.info(
                "monitoring_success progress_iterations=%d threshold=%d",
                progress,
                ctx.monitoring_success_iterations,
            )
            return RestartDone(bad_node_ids=state.bad_node_ids)

        elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
        if elapsed > ctx.monitoring_timeout_seconds:
            logger.warning("monitoring_timeout elapsed=%.0f", elapsed)
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        return None
