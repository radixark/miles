from __future__ import annotations

import logging
from datetime import timedelta

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType

logger = logging.getLogger(__name__)

_CONSECUTIVE_FAILURES_METRIC = "ft_collector_consecutive_failures"
_CRITICAL_COLLECTORS = frozenset({"GpuCollector", "NetworkCollector", "DiskCollector"})
_DEFAULT_FAILURE_THRESHOLD = 10


class CollectorHealthDetector(BaseFaultDetector):
    """Detects nodes where critical collectors have failed repeatedly.

    If any critical collector (GPU, network, disk) has consecutive failures
    above the threshold on an active node, this detector returns a
    NOTIFY_HUMAN decision with TELEMETRY_BLIND trigger so operators know
    the node's metrics are unreliable.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        critical_collectors: frozenset[str] = _CRITICAL_COLLECTORS,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._critical_collectors = critical_collectors

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        df = ctx.metric_store.time_series_store.query_latest(
            _CONSECUTIVE_FAILURES_METRIC,
        )
        if df.is_empty():
            return Decision.no_fault(reason="no collector health data yet")

        blind_nodes: list[str] = []
        blind_details: list[str] = []

        for row in df.iter_rows(named=True):
            labels = row.get("labels", {})
            if not isinstance(labels, dict):
                continue

            collector = labels.get("collector", "")
            node_id = labels.get("node_id", "")
            value = row.get("value", 0.0)

            if collector not in self._critical_collectors:
                continue
            if node_id not in ctx.active_node_ids:
                continue
            if value >= self._failure_threshold:
                if node_id not in blind_nodes:
                    blind_nodes.append(node_id)
                blind_details.append(f"{node_id}/{collector}={int(value)}")

        if not blind_nodes:
            return Decision.no_fault(reason="all critical collectors healthy")

        logger.warning("telemetry_blind_nodes: %s", blind_details)
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason=f"telemetry blind: {'; '.join(blind_details)}",
            trigger=TriggerType.TELEMETRY_BLIND,
        )
