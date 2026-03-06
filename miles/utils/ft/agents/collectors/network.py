from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models.metrics import MetricSample
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)

_DEFAULT_INCLUDE_PATTERNS = ["ib*", "eth*", "en*"]
_DEFAULT_EXCLUDE_PATTERNS = ["lo", "docker*", "veth*"]

_STAT_FILE_TO_METRIC: dict[str, str] = {
    "rx_errors": mn.NODE_NETWORK_RECEIVE_ERRS_TOTAL,
    "tx_errors": mn.NODE_NETWORK_TRANSMIT_ERRS_TOTAL,
    "rx_dropped": mn.NODE_NETWORK_RECEIVE_DROP_TOTAL,
    "tx_dropped": mn.NODE_NETWORK_TRANSMIT_DROP_TOTAL,
}


class NetworkCollector(BaseCollector):
    collect_interval: float = 30.0

    def __init__(
        self,
        sysfs_net_path: Path = Path("/sys/class/net"),
        interface_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._sysfs_net_path = sysfs_net_path
        self._include_patterns = interface_patterns or _DEFAULT_INCLUDE_PATTERNS
        self._exclude_patterns = exclude_patterns or _DEFAULT_EXCLUDE_PATTERNS

    def _collect_sync(self) -> list[MetricSample]:
        if not self._sysfs_net_path.exists():
            logger.warning("sysfs net path %s does not exist", self._sysfs_net_path)
            return []

        samples: list[MetricSample] = []

        for iface_dir in sorted(self._sysfs_net_path.iterdir()):
            iface_name = iface_dir.name
            if not self._should_collect(iface_name):
                continue

            iface_label = {"device": iface_name}
            samples.extend(self._collect_operstate(iface_dir, iface_label))
            samples.extend(self._collect_statistics(iface_dir, iface_label))

        return samples

    def _should_collect(self, iface_name: str) -> bool:
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(iface_name, pattern):
                return False

        for pattern in self._include_patterns:
            if fnmatch.fnmatch(iface_name, pattern):
                return True

        return False

    @staticmethod
    @graceful_degrade(default=[])
    def _collect_operstate(
        iface_dir: Path,
        iface_label: dict[str, str],
    ) -> list[MetricSample]:
        operstate_file = iface_dir / "operstate"
        state = operstate_file.read_text().strip().lower()
        value = 1.0 if state == "up" else 0.0
        return [MetricSample(name=mn.NODE_NETWORK_UP, labels=iface_label, value=value)]

    @staticmethod
    def _collect_statistics(
        iface_dir: Path,
        iface_label: dict[str, str],
    ) -> list[MetricSample]:
        stats_dir = iface_dir / "statistics"
        if not stats_dir.exists():
            return []

        samples: list[MetricSample] = []
        for stat_filename, metric_name in _STAT_FILE_TO_METRIC.items():
            stat_file = stats_dir / stat_filename
            try:
                value = int(stat_file.read_text().strip())
                samples.append(MetricSample(name=metric_name, labels=iface_label, value=float(value)))
            except Exception:
                logger.warning(
                    "Failed to read %s for %s",
                    stat_filename,
                    iface_label["device"],
                    exc_info=True,
                )

        return samples
