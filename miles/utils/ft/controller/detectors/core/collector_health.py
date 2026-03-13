from __future__ import annotations

import logging

import polars as pl

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
            logger.debug("detector: CollectorHealthDetector no data yet")
            return Decision.no_fault(reason="no collector health data yet")

        bad = df.filter(
            pl.col("collector").is_in(self._critical_collectors)
            & pl.col("node_id").is_in(ctx.active_node_ids)
            & (pl.col("value") >= self._failure_threshold)
        )

        if bad.is_empty():
            return Decision.no_fault(reason="all critical collectors healthy")

        blind_details = [
            f"{row['node_id']}/{row['collector']}={int(row['value'])}"
            for row in bad.sort("node_id", "collector").iter_rows(named=True)
        ]
        blind_nodes = list(dict.fromkeys(row["node_id"] for row in bad.iter_rows(named=True)))

        logger.warning("telemetry_blind_nodes: %s", blind_details)
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            bad_node_ids=blind_nodes,
            reason=f"telemetry blind: {'; '.join(blind_details)}",
            trigger=TriggerType.TELEMETRY_BLIND,
        )
