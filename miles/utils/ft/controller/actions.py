from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.utils.ft.controller.recovery.lifecycle_manager import RecoveryLifecycleManager

from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.helpers import (
    evict_and_notify,
    safe_notify,
)
from miles.utils.ft.models.fault import ActionType, Decision
from miles.utils.ft.protocols.metrics import MetricQueryProtocol
from miles.utils.ft.protocols.platform import (
    DiagnosticOrchestratorProtocol,
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
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    controller_exporter: ControllerExporter | None
    on_new_run: Callable[[str], None] | None = field(default=None)
    rank_pids_provider: Callable[[str], dict[int, int]] | None = field(default=None)


async def handle_mark_bad_and_restart(
    decision: Decision,
    deps: PlatformDeps,
) -> None:
    logger.warning(
        "decision_mark_bad_and_restart bad_node_ids=%s reason=%s",
        decision.bad_node_ids, decision.reason,
    )

    success = await evict_and_notify(
        node_manager=deps.node_manager,
        training_job=deps.training_job,
        bad_node_ids=list(decision.bad_node_ids),
        reason=decision.reason,
        notifier=deps.notifier,
        on_new_run=deps.on_new_run,
        fail_fast=False,
    )
    if not success:
        await safe_notify(
            deps.notifier, title="Restart Failure",
            content="stop_and_submit failed after mark_bad_and_restart",
        )


async def handle_enter_recovery(
    decision: Decision,
    deps: PlatformDeps,
    recovery_manager: RecoveryLifecycleManager,
) -> None:
    started = await recovery_manager.start(
        decision=decision, deps=deps,
    )
    if not started:
        await handle_notify_human(
            decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"Recovery cooldown: {decision.trigger.value} triggered too many times",
                trigger=decision.trigger,
            ),
            notifier=deps.notifier,
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
