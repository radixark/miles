from __future__ import annotations

import logging
from datetime import timedelta

from miles.utils.ft.controller.detectors.checks.hardware import (
    check_all_hardware_faults,
    check_nic_down_in_window,
)
from miles.utils.ft.models.fault import NodeFault
from miles.utils.ft.protocols.metrics import MetricQueryProtocol

logger = logging.getLogger(__name__)

_DEFAULT_NETWORK_ALERT_WINDOW = timedelta(minutes=5)
_DEFAULT_NETWORK_ALERT_THRESHOLD = 2


class AlertChecker:
    def __init__(
        self,
        metric_store: MetricQueryProtocol,
        network_alert_window: timedelta = _DEFAULT_NETWORK_ALERT_WINDOW,
        network_alert_threshold: int = _DEFAULT_NETWORK_ALERT_THRESHOLD,
    ) -> None:
        self._metric_store = metric_store
        self._network_alert_window = network_alert_window
        self._network_alert_threshold = network_alert_threshold

    def check_alerts(self) -> list[NodeFault]:
        """Return all detected faults (hardware + network)."""
        faults = check_all_hardware_faults(self._metric_store)
        faults.extend(check_nic_down_in_window(
            self._metric_store,
            window=self._network_alert_window,
            threshold=self._network_alert_threshold,
        ))
        return faults
