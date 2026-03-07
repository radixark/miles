from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.helpers import (
    get_already_bad_nodes,
    retry_mark_node_bad,
    safe_notify,
    stop_and_submit,
)
from miles.utils.ft.controller.state_machine import StateMachineStepper
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.protocols.platform import (
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_WANDB_ITERATION_METRIC = "iteration"
_PENDING_TIMEOUT_SECONDS: int = 300


# ---------------------------------------------------------------------------
# State classes
# ---------------------------------------------------------------------------


class RestartState(FtBaseModel):
    model_config = ConfigDict(frozen=True)
    bad_node_ids: list[str] = Field(default_factory=list)


class Evicting(RestartState):
    pass


class StoppingAndRestarting(RestartState):
    submitted: bool = False
    submit_time: datetime | None = None


class MonitoringProgress(RestartState):
    start_time: datetime
    base_iteration: int


class RestartDone(RestartState):
    pass


class RestartFailed(RestartState):
    pass


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


class RestartStepper(StateMachineStepper[RestartState]):
    def __init__(
        self,
        *,
        node_manager: NodeManagerProtocol,
        training_job: TrainingJobProtocol,
        mini_wandb: MiniWandb,
        notifier: NotificationProtocol | None,
        on_new_run: Callable[[str], None] | None,
        monitoring_success_iterations: int,
        monitoring_timeout_seconds: int,
    ) -> None:
        self._node_manager = node_manager
        self._training_job = training_job
        self._mini_wandb = mini_wandb
        self._notifier = notifier
        self._on_new_run = on_new_run
        self._monitoring_success_iterations = monitoring_success_iterations
        self._monitoring_timeout_seconds = monitoring_timeout_seconds
        super().__init__()

    def set_on_new_run(self, callback: Callable[[str], None]) -> None:
        self._on_new_run = callback

    def _build_handlers(self) -> dict[type, Callable[[RestartState], Awaitable[RestartState | None]]]:
        return {
            Evicting: self._handle_evicting,
            StoppingAndRestarting: self._handle_stopping_and_restarting,
            MonitoringProgress: self._handle_monitoring_progress,
        }

    async def _handle_evicting(self, state: Evicting) -> RestartState:
        already_bad = await get_already_bad_nodes(self._node_manager)
        nodes_to_mark = [n for n in state.bad_node_ids if n not in already_bad]

        for node_id in nodes_to_mark:
            result = await retry_mark_node_bad(
                self._node_manager, node_id=node_id, reason="recovery eviction",
            )
            if not result.ok:
                return RestartFailed(bad_node_ids=state.bad_node_ids)

        await safe_notify(
            self._notifier,
            title="Nodes evicted",
            content=f"Evicted: {nodes_to_mark}",
            severity="warning",
        )
        return StoppingAndRestarting(bad_node_ids=state.bad_node_ids)

    async def _handle_stopping_and_restarting(
        self, state: StoppingAndRestarting,
    ) -> RestartState | None:
        if not state.submitted:
            return await self._submit(state)
        return await self._poll(state)

    async def _submit(self, state: StoppingAndRestarting) -> RestartState | None:
        excluded = state.bad_node_ids or None
        success = await stop_and_submit(
            self._training_job,
            excluded_node_ids=excluded,
            on_new_run=self._on_new_run,
        )
        if not success:
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        return StoppingAndRestarting(
            bad_node_ids=state.bad_node_ids,
            submitted=True,
            submit_time=datetime.now(timezone.utc),
        )

    async def _poll(self, state: StoppingAndRestarting) -> RestartState | None:
        status = await self._training_job.get_training_status()

        if status == JobStatus.RUNNING:
            iteration = self._mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
            base = (
                int(iteration) if iteration is not None and math.isfinite(iteration) else 0
            )
            return MonitoringProgress(
                bad_node_ids=state.bad_node_ids,
                start_time=datetime.now(timezone.utc),
                base_iteration=base,
            )

        if status == JobStatus.FAILED:
            logger.warning("restart_job_immediately_failed bad_node_ids=%s", state.bad_node_ids)
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        if state.submit_time is not None:
            elapsed = (datetime.now(timezone.utc) - state.submit_time).total_seconds()
            if elapsed > _PENDING_TIMEOUT_SECONDS:
                logger.warning("restart_pending_timeout elapsed=%.0f", elapsed)
                return RestartFailed(bad_node_ids=state.bad_node_ids)

        return None

    async def _handle_monitoring_progress(
        self, state: MonitoringProgress,
    ) -> RestartState | None:
        status = await self._training_job.get_training_status()

        if status == JobStatus.FAILED:
            logger.warning("monitoring_training_failed bad_node_ids=%s", state.bad_node_ids)
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        progress = self._iteration_progress(state)

        if status == JobStatus.RUNNING and progress >= self._monitoring_success_iterations:
            logger.info(
                "monitoring_success progress_iterations=%d threshold=%d",
                progress, self._monitoring_success_iterations,
            )
            return RestartDone(bad_node_ids=state.bad_node_ids)

        elapsed = (datetime.now(timezone.utc) - state.start_time).total_seconds()
        if elapsed > self._monitoring_timeout_seconds:
            logger.warning("monitoring_timeout elapsed=%.0f", elapsed)
            return RestartFailed(bad_node_ids=state.bad_node_ids)

        return None

    def _iteration_progress(self, state: MonitoringProgress) -> int:
        current_iteration = self._mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
        if current_iteration is None or not math.isfinite(current_iteration):
            return 0
        raw = int(current_iteration) - state.base_iteration
        if raw < 0:
            logger.warning(
                "iteration_progress_negative current=%d base=%d",
                int(current_iteration), state.base_iteration,
            )
            return 0
        return raw
