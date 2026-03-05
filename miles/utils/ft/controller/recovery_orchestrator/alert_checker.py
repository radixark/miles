from __future__ import annotations

from miles.utils.ft.controller.hardware_checks import check_all_hardware_faults
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol


class AlertChecker:
    def __init__(self, metric_store: MetricStoreProtocol) -> None:
        self._metric_store = metric_store

    def check_alerts(self) -> tuple[list[str], list[str]]:
        """Return (sorted bad_node_ids, reasons)."""
        faults = check_all_hardware_faults(self._metric_store)

        seen: set[str] = set()
        bad_node_ids: list[str] = []
        reasons: list[str] = []
        for fault in faults:
            if fault.node_id not in seen:
                seen.add(fault.node_id)
                bad_node_ids.append(fault.node_id)
            reasons.append(fault.reason)

        return sorted(bad_node_ids), reasons
