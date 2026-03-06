from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery_orchestrator.helpers import (
    safe_notify,
    stop_and_submit,
)
from miles.utils.ft.retry import retry_async
from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator
from miles.utils.ft.models._fault import Decision
from miles.utils.ft.protocols.metrics import MetricQueryProtocol
from miles.utils.ft.protocols.platform import (
    DiagnosticSchedulerProtocol,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)


@dataclass
class PlatformDeps:
    """Bundles platform-level dependencies shared across action handlers."""

    node_manager: NodeManagerProtocol
    training_job: TrainingJobProtocol
    metric_store: MetricQueryProtocol
    mini_wandb: MiniWandb
    notifier: NotificationProtocol | None
    diagnostic_scheduler: DiagnosticSchedulerProtocol
    controller_exporter: ControllerExporter | None
    on_new_run: Callable[[str], None] | None = field(default=None)


async def handle_mark_bad_and_restart(
    decision: Decision,
    deps: PlatformDeps,
) -> None:
    logger.warning(
        "decision_mark_bad_and_restart bad_node_ids=%s reason=%s",
        decision.bad_node_ids, decision.reason,
    )
    failed_nodes: list[str] = []
    for node_id in decision.bad_node_ids:
        result = await retry_async(
            lambda nid=node_id: deps.node_manager.mark_node_bad(
                nid, reason=decision.reason,
            ),
            description=f"mark_node_bad({node_id})",
        )
        if not result.ok:
            failed_nodes.append(node_id)

    if failed_nodes:
        msg = f"mark_node_bad failed for nodes: {failed_nodes}"
        logger.error("mark_bad_partial_failure %s", msg)
        await safe_notify(
            deps.notifier, title="Mark-Bad Failure", content=msg,
        )

    restart_ok = await stop_and_submit(deps.training_job, on_new_run=deps.on_new_run)
    if not restart_ok:
        msg = "stop_and_submit failed after mark_bad_and_restart"
        logger.error(msg)
        await safe_notify(
            deps.notifier, title="Restart Failure", content=msg,
        )


async def handle_enter_recovery(
    decision: Decision,
    deps: PlatformDeps,
) -> RecoveryOrchestrator:
    logger.warning(
        "decision_enter_recovery trigger=%s reason=%s",
        decision.trigger, decision.reason,
    )
    return RecoveryOrchestrator(
        trigger=decision.trigger,
        node_manager=deps.node_manager,
        training_job=deps.training_job,
        metric_store=deps.metric_store,
        mini_wandb=deps.mini_wandb,
        notifier=deps.notifier,
        diagnostic_scheduler=deps.diagnostic_scheduler,
        controller_exporter=deps.controller_exporter,
        on_new_run=deps.on_new_run,
    )


async def handle_notify_human(
    decision: Decision,
    notifier: NotificationProtocol | None,
) -> None:
    logger.warning(
        "decision_notify_human reason=%s",
        decision.reason,
    )
    await safe_notify(
        notifier, title="Fault Alert", content=decision.reason,
    )
