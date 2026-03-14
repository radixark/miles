from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.types import ActionType, Decision, MetricStore, TriggerType

logger = logging.getLogger(__name__)


def _filter_node_ids_by_active(node_ids: list[str], active_node_ids: frozenset[str]) -> list[str]:
    """Keep only node IDs that are in the active training placement."""
    return [n for n in node_ids if n in active_node_ids]


@dataclass
class DetectorContext:
    metric_store: MetricStore
    active_node_ids: frozenset[str] = field(default_factory=frozenset)
    job_status: JobStatus | None = None
    active_run_id: str | None = None
    seconds_since_run_start: float = 0.0


class BaseFaultDetector(ABC):
    """Base class for fault detectors.

    Detectors must be **stateless**: ``_evaluate_raw()`` must derive its
    answer entirely from the data available in ``DetectorContext`` (metric
    stores, job status, active node IDs).  No mutable instance state should
    be accumulated across calls.  Constructor parameters (thresholds,
    config) are fine because they are immutable after init.

    The public ``evaluate()`` method wraps ``_evaluate_raw()`` and filters
    out bad-node IDs that are not in the current training's
    ``active_node_ids``, so individual detectors do not need to handle this.
    """

    def _should_filter_by_active_nodes(self) -> bool:
        """Whether evaluate() gates on active_node_ids.

        Most detectors need this (hardware faults are meaningless without
        a known training placement). Override to return False for detectors
        that operate independently of node placement (e.g. cell-level checks).
        """
        return True

    def evaluate(self, ctx: DetectorContext) -> Decision:
        logger.debug("detector: %s.evaluate called", type(self).__name__)
        decision = self._evaluate_raw(ctx)
        if self._should_filter_by_active_nodes() and not ctx.active_node_ids:
            if decision.bad_node_ids:
                logger.info("detector_skipped_no_active_nodes detector=%s", type(self).__name__)
            return Decision.no_fault(reason=f"no active nodes ({type(self).__name__})")
        if decision.bad_node_ids and ctx.active_node_ids:
            filtered = _filter_node_ids_by_active(decision.bad_node_ids, ctx.active_node_ids)
            if not filtered:
                logger.info(
                    "detector_bad_nodes_not_active detector=%s bad=%s active=%s",
                    type(self).__name__,
                    decision.bad_node_ids,
                    sorted(ctx.active_node_ids),
                )
                return Decision.no_fault(
                    reason=f"all bad nodes not active ({type(self).__name__})",
                )
            if len(filtered) != len(decision.bad_node_ids):
                logger.debug(
                    "detector: %s filtered bad_node_ids: %d -> %d",
                    type(self).__name__,
                    len(decision.bad_node_ids),
                    len(filtered),
                )
                decision = decision.model_copy(update={"bad_node_ids": filtered})
        if decision.action != ActionType.NONE:
            logger.info(
                "detector: %s result: action=%s, trigger=%s, bad_nodes=%d",
                type(self).__name__,
                decision.action.value,
                decision.trigger,
                len(decision.bad_node_ids),
            )
        return decision

    @abstractmethod
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision: ...


_METRIC_BLIND_STARTUP_GRACE_SECONDS: float = 120.0


def check_metric_blind(
    ctx: DetectorContext,
    metric_name: str,
    *,
    detector_name: str,
    startup_grace_seconds: float = _METRIC_BLIND_STARTUP_GRACE_SECONDS,
) -> Decision | None:
    """Return a TELEMETRY_BLIND decision if *metric_name* has no data for
    any active node, or ``None`` if data is present (caller should proceed
    with normal evaluation).

    During the first *startup_grace_seconds* after run start, metric absence
    is expected (scrapers haven't collected yet), so we suppress the alert.
    """
    if not ctx.active_node_ids:
        return None

    if ctx.seconds_since_run_start < startup_grace_seconds:
        return None

    df = ctx.metric_store.time_series_store.query_latest(metric_name)
    if df is not None and not df.is_empty():
        return None

    logger.warning("detector: %s metric_blind: metric=%s missing for active nodes", detector_name, metric_name)
    return Decision(
        action=ActionType.NOTIFY_HUMAN,
        reason=f"{detector_name}: core metric {metric_name} missing for active nodes",
        trigger=TriggerType.TELEMETRY_BLIND,
        notify_deduplicator_id=f"metric_blind:{detector_name}:{metric_name}",
    )
