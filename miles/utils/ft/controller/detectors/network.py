from datetime import timedelta

import polars as pl

from miles.utils.ft.metric_names import NODE_NETWORK_UP
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models import Decision, NodeFault

_DEFAULT_ALERT_WINDOW = timedelta(minutes=5)
_DEFAULT_ALERT_THRESHOLD = 2


class NetworkAlertDetector(BaseFaultDetector):
    def __init__(
        self,
        alert_window: timedelta = _DEFAULT_ALERT_WINDOW,
        alert_threshold: int = _DEFAULT_ALERT_THRESHOLD,
    ) -> None:
        if alert_window.total_seconds() <= 0:
            raise ValueError(f"alert_window must be positive, got {alert_window}")
        if alert_threshold < 1:
            raise ValueError(f"alert_threshold must be >= 1, got {alert_threshold}")

        self._alert_window = alert_window
        self._alert_threshold = alert_threshold

    def evaluate(self, ctx: DetectorContext) -> Decision:
        df = ctx.metric_store.query_range(NODE_NETWORK_UP, window=self._alert_window)
        if df.is_empty():
            return Decision.from_node_faults([], fallback_reason="no NIC data in window")

        down_samples = df.filter(pl.col("value") == 0.0)
        if down_samples.is_empty():
            return Decision.from_node_faults([], fallback_reason="no NIC alerts in window")

        node_down_counts: dict[str, int] = {}
        for row in down_samples.iter_rows(named=True):
            node_id = row["node_id"]
            node_down_counts[node_id] = node_down_counts.get(node_id, 0) + 1

        faults: list[NodeFault] = [
            NodeFault(
                node_id=node_id,
                reason=f"NIC down {count} times on {node_id} in {self._alert_window}",
            )
            for node_id, count in sorted(node_down_counts.items())
            if count >= self._alert_threshold
        ]

        return Decision.from_node_faults(faults, fallback_reason="NIC alerts below threshold")
