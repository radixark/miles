from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.restart.models import (
    EvictingSt,
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
    MonitoringIterationProgressConfig,
    MonitoringProgressSt,
    MonitoringRunningAfterDelayConfig,
    RestartContext,
    RestartDoneSt,
    RestartFailedSt,
    RestartState,
    StoppingAndRestartingSt,
)
from miles.utils.ft.controller.state_machines.restart.utils import retry_mark_node_bad, stop_and_submit
from miles.utils.ft.controller.state_machines.utils import safe_notify
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)

_WANDB_ITERATION_METRIC = "iteration"


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
        logger.debug("restart_sm: EvictingHandler.step bad_node_ids=%s", state.bad_node_ids)
        for node_id in state.bad_node_ids:
            metadata = ctx.node_metadata.get(node_id)
            result = await retry_mark_node_bad(
                ctx.node_manager,
                node_id=node_id,
                reason="recovery eviction",
                node_metadata=metadata,
            )
            if not result.ok:
                logger.warning("restart_sm: mark_node_bad failed for node=%s, transitioning to RestartFailedSt", node_id)
                return RestartFailedSt(bad_node_ids=state.bad_node_ids)
            if ctx.on_node_evicted is not None:
                ctx.on_node_evicted(node_id)

        await safe_notify(
            ctx.notifier,
            title="Nodes evicted",
            content=f"Evicted: {state.bad_node_ids}",
            severity="warning",
        )

        logger.info(
            "restart_sm: state transition: old=EvictingSt, new=StoppingAndRestartingSt, trigger=eviction_complete, "
            "bad_nodes=%s",
            state.bad_node_ids,
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
        if ctx.is_main_job_restart:
            logger.info(
                "restart_sm: state transition: old=StoppingAndRestartingSt, new=ExternalRestartingMainJobSt, "
                "trigger=main_job_restart_delegation"
            )
            return ExternalRestartingMainJobSt(bad_node_ids=state.bad_node_ids)

        success = await stop_and_submit(
            job=ctx.actuator,
            on_new_run=ctx.on_new_run,
            restart_lock=ctx.restart_lock,
        )
        if not success:
            logger.warning("restart_sm: stop_and_submit failed, transitioning to RestartFailedSt")
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        logger.info("restart_sm: stop_and_submit succeeded, job submitted")
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
            logger.info(
                "restart_sm: state transition: old=StoppingAndRestartingSt, new=MonitoringProgressSt, "
                "trigger=job_running, base_iteration=%d",
                base,
            )
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
            if elapsed > ctx.pending_timeout_seconds:
                logger.warning("restart_pending_timeout elapsed=%.0f timeout=%d", elapsed, ctx.pending_timeout_seconds)
                return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        logger.debug("restart_sm: _poll returning None, status=%s", status.value)
        return None


class MonitoringProgressHandler(StateHandler[MonitoringProgressSt, RestartContext]):
    async def step(
        self,
        state: MonitoringProgressSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        logger.debug(
            "restart_sm: MonitoringProgressHandler.step mode=%s, base_iteration=%d",
            ctx.monitoring_config.mode,
            state.base_iteration,
        )
        if isinstance(ctx.monitoring_config, MonitoringRunningAfterDelayConfig):
            return await self._step_running_after_delay(state=state, ctx=ctx)
        return await self._step_iteration_progress(state=state, ctx=ctx)

    async def _step_iteration_progress(
        self,
        *,
        state: MonitoringProgressSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        status = await ctx.actuator.get_status()
        now = datetime.now(timezone.utc)
        elapsed = (now - state.start_time).total_seconds()

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

        if elapsed > config.timeout_seconds:
            logger.warning("monitoring_timeout elapsed=%.0f", elapsed)
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        logger.debug("restart_sm: _step_iteration_progress: progress=%d, elapsed=%.0f, waiting", progress, elapsed)
        return None

    async def _step_running_after_delay(
        self,
        *,
        state: MonitoringProgressSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        assert isinstance(ctx.monitoring_config, MonitoringRunningAfterDelayConfig)
        config = ctx.monitoring_config

        status = await ctx.actuator.get_status()
        now = datetime.now(timezone.utc)
        elapsed = (now - state.start_time).total_seconds()

        if status == JobStatus.FAILED:
            logger.warning("running_after_delay_failed")
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        if status == JobStatus.RUNNING and elapsed >= config.alive_duration_seconds:
            logger.info(
                "running_after_delay_success elapsed=%.0f threshold=%d",
                elapsed,
                config.alive_duration_seconds,
            )
            return RestartDoneSt(bad_node_ids=state.bad_node_ids)

        if elapsed > config.timeout_seconds:
            logger.warning("running_after_delay_timeout elapsed=%.0f", elapsed)
            return RestartFailedSt(bad_node_ids=state.bad_node_ids)

        logger.debug("restart_sm: _step_running_after_delay: elapsed=%.0f, status=%s, waiting", elapsed, status.value)
        return None


class ExternalRestartingMainJobHandler(StateHandler[ExternalRestartingMainJobSt, RestartContext]):
    async def step(
        self,
        state: ExternalRestartingMainJobSt,
        ctx: RestartContext,
    ) -> RestartState | None:
        if state.external_execution_result is None:
            logger.debug("restart_sm: ExternalRestartingMainJobHandler waiting for external result")
            return None

        if state.external_execution_result == ExternalExecutionResult.SUCCEEDED:
            current_iter = ctx.metric_store.mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
            base = int(current_iter) if current_iter is not None and math.isfinite(current_iter) else 0
            logger.info(
                "restart_sm: state transition: old=ExternalRestartingMainJobSt, new=MonitoringProgressSt, "
                "trigger=external_succeeded, base_iteration=%d",
                base,
            )
            return MonitoringProgressSt(
                bad_node_ids=state.bad_node_ids,
                start_time=datetime.now(timezone.utc),
                base_iteration=base,
            )

        # FAILED or TIMEOUT
        logger.warning(
            "restart_sm: external restart result=%s, transitioning to RestartFailedSt",
            state.external_execution_result.value,
        )
        return RestartFailedSt(bad_node_ids=state.bad_node_ids)
