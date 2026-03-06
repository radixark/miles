from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.helpers import (
    get_already_bad_nodes,
    retry_mark_node_bad,
    safe_notify,
    stop_and_submit,
)
from miles.utils.ft.models.fault import Decision
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

    already_bad = await get_already_bad_nodes(deps.node_manager)
    nodes_to_mark = [nid for nid in decision.bad_node_ids if nid not in already_bad]
    skipped = set(decision.bad_node_ids) - set(nodes_to_mark)
    if len(skipped) > 0:
        logger.info(
            "mark_bad_skipped_already_bad skipped=%s",
            sorted(skipped),
        )

    failed_nodes: list[str] = []
    for node_id in nodes_to_mark:
        result = await retry_mark_node_bad(
            deps.node_manager, node_id=node_id, reason=decision.reason,
        )
        if not result.ok:
            failed_nodes.append(node_id)

    if failed_nodes:
        msg = f"mark_node_bad failed for nodes: {failed_nodes}"
        logger.error("mark_bad_partial_failure %s", msg)

    restart_ok = await stop_and_submit(deps.training_job, on_new_run=deps.on_new_run)
    if not restart_ok:
        msg = "stop_and_submit failed after mark_bad_and_restart"
        logger.error(msg)
        # We cannot restart the training job, thus cannot do anything and have to notify human
        await safe_notify(
            deps.notifier, title="Restart Failure", content=msg,
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
