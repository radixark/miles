from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.subsystem_hub import RestartMode
from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import MetricStore
from miles.utils.ft.controller.state_machines.restart.models import (
    EvictingSt,
    ExternalExecutionResult,
    MonitoringProgressSt,
    RestartContext,
    RestartDoneSt,
    RestartFailedSt,
    ExternalRestartingMainJobSt,
    RestartState,
    StoppingAndRestartingSt,
)
from miles.utils.ft.controller.state_machines.restart.utils import (
    retry_mark_node_bad,
    stop_and_submit,
)
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)

_WANDB_ITERATION_METRIC = "iteration"
_PENDING_TIMEOUT_SECONDS: int = 300


def iteration_progress(state: MonitoringProgressSt, mini_wandb: MiniWandb) -> int:
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


class EvictingHandler(StateHandler[EvictingSt, RestartContext]):
    async def step(self, state: EvictingSt, ctx: RestartContext) -> RestartState:
        for node_id in state.bad_node_ids:
            metadata = ctx.node_metadata.get(node_id)
            result = await retry_mark_node_bad(
                ctx.node_manager,
                node_id=node_id,
                reason="recovery eviction",
                node_metadata=metadata,
            )
            if not result.ok:
                return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        await safe_notify(
            ctx.notifier,
            title="Nodes evicted",
            content=f"Evicted: {state.bad_node_ids}",
            severity="warning",
        )

        return StoppingAndRestartingSt(bad_node_ids=state.bad_node_ids)


class StoppingAndRestartingHandler(StateHandler[StoppingAndRestartingSt, RestartContext]):
    async def step(
        self,
        state: StoppingAndRestartingSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        if not state.submitted:
            return await self._submit(state=state, ctx=ctx)
        return await self._poll(state=state, ctx=ctx)

    async def _submit(
        self,
        *,
        state: StoppingAndRestartingSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        if ctx.restart_mode == RestartMode.MAIN_JOB:
            return ExternalRestartingMainJobSt(bad_node_ids=state.bad_node_ids)

        success = await stop_and_submit(
            job=ctx.actuator,
            on_new_run=ctx.on_new_run,
        )
        if not success:
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        return StoppingAndRestartingSt(
            bad_node_ids=state.bad_node_ids,
            submitted=True,
            submit_time=datetime.now(timezone.utc),
        )

    async def _poll(
        self,
        *,
        state: StoppingAndRestartingSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        status = await ctx.actuator.get_status()

        if status == JobStatus.RUNNING:
            current_iter = ctx.metric_store.mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
            base = int(current_iter) if current_iter is not None and math.isfinite(current_iter) else 0
            return MonitoringProgressSt(
                bad_node_ids=state.bad_node_ids,
                start_time=datetime.now(timezone.utc),
                base_iteration=base,
            )

        if status == JobStatus.FAILED:
            logger.warning("restart_job_immediately_failed")
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        if state.submit_time is not None:
            elapsed = (datetime.now(timezone.utc) - state.submit_time).total_seconds()
            if elapsed > _PENDING_TIMEOUT_SECONDS:
                logger.warning("restart_pending_timeout elapsed=%.0f", elapsed)
                return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        return None


class MonitoringProgressHandler(StateHandler[MonitoringProgressSt, RestartContext]):
    async def step(
        self,
        state: MonitoringProgressSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        if isinstance(ctx.monitoring_config, MonitoringSustainedAliveConfig):
            return await self._step_sustained_alive(state=state, ctx=ctx)
        return await self._step_iteration_progress(state=state, ctx=ctx)

    async def _step_iteration_progress(
        self,
        *,
        state: MonitoringProgressSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        status = await ctx.actuator.get_status()

        if status == JobStatus.FAILED:
            logger.warning("monitoring_training_failed")
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        assert isinstance(ctx.monitoring_config, MonitoringIterationProgressConfig)
        config = ctx.monitoring_config

        progress = iteration_progress(state=state, mini_wandb=ctx.metric_store.mini_wandb)

        if status == JobStatus.RUNNING and progress >= config.success_iterations:
            logger.info(
                "monitoring_success progress_iterations=%d threshold=%d",
                progress,
                config.success_iterations,
            )
            return RestartDoneSt(bad_node_ids=state.bad_node_ids)

        elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
        if elapsed > config.timeout_seconds:
            logger.warning("monitoring_timeout elapsed=%.0f", elapsed)
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        return None

    async def _step_sustained_alive(
        self,
        *,
        state: MonitoringProgressSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        assert isinstance(ctx.monitoring_config, MonitoringSustainedAliveConfig)
        config = ctx.monitoring_config

        status = await ctx.actuator.get_status()

        if status == JobStatus.FAILED:
            logger.warning("sustained_alive_failed")
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        if status == JobStatus.RUNNING:
            elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
            if elapsed >= config.alive_duration_seconds:
                logger.info(
                    "sustained_alive_success elapsed=%.0f threshold=%d",
                    elapsed,
                    config.alive_duration_seconds,
                )
                return RestartDoneSt(bad_node_ids=state.bad_node_ids)

        elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
        if elapsed > config.timeout_seconds:
            logger.warning("sustained_alive_timeout elapsed=%.0f", elapsed)
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        return None


class ExternalRestartingMainJobHandler(StateHandler[ExternalRestartingMainJobSt, RestartContext]):
    async def step(
        self,
        state: ExternalRestartingMainJobSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        if state.external_execution_result is None:
            return None

        if state.external_execution_result == ExternalExecutionResult.SUCCEEDED:
            current_iter = ctx.metric_store.mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
            base = int(current_iter) if current_iter is not None and math.isfinite(current_iter) else 0
            return MonitoringProgressSt(
                bad_node_ids=state.bad_node_ids,
                start_time=datetime.now(timezone.utc),
                base_iteration=base,
            )

        # FAILED or TIMEOUT
        return RestartFailedSt(bad_node_ids=state.bad_node_ids)
