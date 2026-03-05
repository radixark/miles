from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

from miles.utils.ft.controller.detectors.hardware_checks import check_all_hardware_faults
from miles.utils.ft.protocols.metrics import MetricStoreProtocol
from miles.utils.ft.metric_names import NODE_NETWORK_UP
from miles.utils.ft.models import NodeFault, unique_node_ids

logger = logging.getLogger(__name__)

_DEFAULT_NETWORK_ALERT_WINDOW = timedelta(minutes=5)
_DEFAULT_NETWORK_ALERT_THRESHOLD = 2


class AlertChecker:
    def __init__(
        self,
        metric_store: MetricStoreProtocol,
        network_alert_window: timedelta = _DEFAULT_NETWORK_ALERT_WINDOW,
        network_alert_threshold: int = _DEFAULT_NETWORK_ALERT_THRESHOLD,
    ) -> None:
        self._metric_store = metric_store
        self._network_alert_window = network_alert_window
        self._network_alert_threshold = network_alert_threshold

    def check_alerts(self) -> tuple[list[str], list[str]]:
        """Return (sorted bad_node_ids, reasons)."""
        faults = check_all_hardware_faults(self._metric_store)
        faults.extend(self._check_network_alerts())
        bad_node_ids = sorted(unique_node_ids(faults))
        reasons = [f.reason for f in faults]
        return bad_node_ids, reasons

    def _check_network_alerts(self) -> list[NodeFault]:
        df = self._metric_store.query_range(NODE_NETWORK_UP, window=self._network_alert_window)
        if df.is_empty():
            return []

        down_samples = df.filter(pl.col("value") == 0.0)
        if down_samples.is_empty():
            return []

        node_down_counts: dict[str, int] = {}
        for row in down_samples.iter_rows(named=True):
            node_id = row["node_id"]
            node_down_counts[node_id] = node_down_counts.get(node_id, 0) + 1

        return [
            NodeFault(
                node_id=node_id,
                reason=f"NIC down {count} times on {node_id} in {self._network_alert_window}",
            )
            for node_id, count in sorted(node_down_counts.items())
            if count >= self._network_alert_threshold
        ]
