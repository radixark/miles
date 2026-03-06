from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import datetime, timezone

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.helpers import (
    evict_and_notify,
    safe_notify,
    stop_and_submit,
)
from miles.utils.ft.controller.recovery.alert_checker import AlertChecker
from miles.utils.ft.controller.recovery.context import (
    PENDING_TIMEOUT_SECONDS,
    RecoveryContext,
)
from miles.utils.ft.models.fault import ActionType
from miles.utils.ft.models.recovery import RecoveryPhase
from miles.utils.ft.protocols.platform import (
    DiagnosticOrchestratorProtocol,
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_WANDB_ITERATION_METRIC = "iteration"


# -------------------------------------------------------------------
# CHECK_ALERTS
# -------------------------------------------------------------------


async def step_check_alerts(
    ctx: RecoveryContext,
    alert_checker: AlertChecker,
) -> RecoveryPhase:
    bad_node_ids, reasons = alert_checker.check_alerts()

    if bad_node_ids:
        ctx.bad_node_ids = bad_node_ids
        logger.info("check_alerts_found bad_nodes=%s reasons=%s", ctx.bad_node_ids, reasons)
        return RecoveryPhase.EVICT_AND_RESTART

    logger.info("check_alerts_clean trigger=%s", ctx.trigger)
    return RecoveryPhase.REATTEMPTING


# -------------------------------------------------------------------
# REATTEMPTING
# -------------------------------------------------------------------


async def step_reattempting(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
    on_new_run: Callable[[str], None] | None = None,
) -> RecoveryPhase | None:
    if not ctx.reattempt_submitted:
        return await _reattempt_submit(ctx, training_job, mini_wandb, on_new_run=on_new_run)
    return await _reattempt_poll(ctx, training_job, mini_wandb)


# -------------------------------------------------------------------
# MONITORING
# -------------------------------------------------------------------


async def step_monitoring(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase | None:
    status = await training_job.get_training_status()
    progress = _iteration_progress(ctx, mini_wandb)

    if status == JobStatus.FAILED:
        logger.warning(
            "monitoring_training_failed progress_iterations=%d trigger=%s",
            progress, ctx.trigger,
        )
        return RecoveryPhase.DIAGNOSING

    if status == JobStatus.RUNNING and progress >= ctx.monitoring_success_iterations:
        logger.info(
            "monitoring_success progress_iterations=%d threshold=%d",
            progress, ctx.monitoring_success_iterations,
        )
        return RecoveryPhase.DONE

    if ctx.reattempt_start_time is not None:
        elapsed = (datetime.now(timezone.utc) - ctx.reattempt_start_time).total_seconds()
        if elapsed > ctx.monitoring_timeout_seconds:
            logger.warning("monitoring_timeout elapsed=%.0f trigger=%s", elapsed, ctx.trigger)
            return RecoveryPhase.DIAGNOSING

    return None


# -------------------------------------------------------------------
# DIAGNOSING
# -------------------------------------------------------------------


async def step_diagnosing(
    ctx: RecoveryContext,
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
    rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
) -> RecoveryPhase:
    decision = await diagnostic_orchestrator.run_diagnostic_pipeline(
        trigger_reason=ctx.trigger,
        rank_pids_provider=rank_pids_provider,
    )

    if decision.action == ActionType.MARK_BAD_AND_RESTART:
        ctx.bad_node_ids = list(decision.bad_node_ids)
        logger.info("diagnosing_found_bad_nodes bad_nodes=%s", ctx.bad_node_ids)
        return RecoveryPhase.EVICT_AND_RESTART

    logger.info("diagnosing_all_passed trigger=%s", ctx.trigger)
    return RecoveryPhase.NOTIFY


# -------------------------------------------------------------------
# EVICT_AND_RESTART
# -------------------------------------------------------------------


async def step_evict_and_restart(
    ctx: RecoveryContext,
    node_manager: NodeManagerProtocol,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
    notifier: NotificationProtocol | None = None,
    on_new_run: Callable[[str], None] | None = None,
) -> RecoveryPhase:
    if not ctx.bad_node_ids:
        logger.warning("evict_and_restart called with empty bad_node_ids — skipping to NOTIFY")
        return RecoveryPhase.NOTIFY

    success = await evict_and_notify(
        node_manager=node_manager,
        training_job=training_job,
        bad_node_ids=ctx.bad_node_ids,
        reason=f"recovery eviction: {ctx.trigger}",
        notifier=notifier,
        excluded_node_ids=ctx.bad_node_ids,
        on_new_run=on_new_run,
        fail_fast=True,
    )
    return RecoveryPhase.DONE if success else RecoveryPhase.NOTIFY


# -------------------------------------------------------------------
# NOTIFY
# -------------------------------------------------------------------


async def step_notify(
    ctx: RecoveryContext,
    notifier: NotificationProtocol | None,
) -> RecoveryPhase:
    prev = ctx.phase_before_notify
    message = (
        f"Recovery requires human intervention. "
        f"trigger={ctx.trigger} "
        f"phase_before_notify={prev.value if prev else 'unknown'}"
    )
    logger.warning("recovery_notify reason=%s", message)

    await safe_notify(notifier, title="Recovery Alert", content=message)

    return RecoveryPhase.DONE


# -------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------


async def _reattempt_submit(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
    on_new_run: Callable[[str], None] | None = None,
) -> RecoveryPhase | None:
    success = await stop_and_submit(training_job, on_new_run=on_new_run)
    if not success:
        return RecoveryPhase.NOTIFY

    ctx.reattempt_submitted = True
    ctx.reattempt_submit_time = datetime.now(timezone.utc)
    logger.info("reattempt_submitted trigger=%s", ctx.trigger)
    return None


async def _reattempt_poll(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase | None:
    status = await training_job.get_training_status()

    if status == JobStatus.RUNNING:
        iteration = mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = (
            int(iteration) if iteration is not None and math.isfinite(iteration) else 0
        )
        logger.info("reattempt_running base_iteration=%s", ctx.reattempt_base_iteration)
        return RecoveryPhase.MONITORING

    if status == JobStatus.FAILED:
        logger.warning("reattempt_immediately_failed trigger=%s", ctx.trigger)
        return RecoveryPhase.DIAGNOSING

    if ctx.reattempt_submit_time is not None:
        elapsed = (datetime.now(timezone.utc) - ctx.reattempt_submit_time).total_seconds()
        if elapsed > PENDING_TIMEOUT_SECONDS:
            logger.warning("reattempt_pending_timeout elapsed=%.0f", elapsed)
            return RecoveryPhase.NOTIFY

    return None


def _iteration_progress(ctx: RecoveryContext, mini_wandb: MiniWandb) -> int:
    current_iteration = mini_wandb.latest(metric_name=_WANDB_ITERATION_METRIC)
    if current_iteration is None or not math.isfinite(current_iteration):
        return 0
    base = ctx.reattempt_base_iteration or 0
    raw = int(current_iteration) - base
    if raw < 0:
        logger.warning(
            "iteration_progress_negative current=%d base=%d — possible run reset",
            int(current_iteration), base,
        )
        return 0
    return raw
