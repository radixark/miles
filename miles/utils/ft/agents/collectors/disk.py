from __future__ import annotations

import logging
import os
from pathlib import Path

import miles.utils.ft.metric_names as mn
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import MetricSample

logger = logging.getLogger(__name__)


class DiskCollector(BaseCollector):
    collect_interval: float = 60.0

    def __init__(self, *, disk_mounts: list[Path] | None = None) -> None:
        self._disk_mounts = disk_mounts or []
        if not self._disk_mounts:
            logger.warning(
                "DiskCollector initialized with no disk_mounts — "
                "filesystem metrics will not be collected"
            )

    def _collect_sync(self) -> list[MetricSample]:
        samples: list[MetricSample] = []
        samples.extend(self._collect_disk_avail())
        samples.extend(self._collect_disk_io_time())
        return samples

    def _collect_disk_avail(self) -> list[MetricSample]:
        samples: list[MetricSample] = []
        for mount in self._disk_mounts:
            try:
                stat = os.statvfs(mount)
                available_bytes = stat.f_bavail * stat.f_frsize
                samples.append(MetricSample(
                    name=mn.NODE_FILESYSTEM_AVAIL_BYTES,
                    labels={"mountpoint": str(mount)},
                    value=float(available_bytes),
                ))
            except Exception:
                logger.warning("Failed to statvfs %s", mount, exc_info=True)
        return samples

    def _collect_disk_io_time(self) -> list[MetricSample]:
        sys_block = Path("/sys/block")
        if not sys_block.exists():
            return []

        samples: list[MetricSample] = []
        for device_dir in sorted(sys_block.iterdir()):
            stat_file = device_dir / "stat"
            try:
                text = stat_file.read_text().strip()
                fields = text.split()
                if len(fields) >= 10:
                    io_time_ms = int(fields[9])
                    samples.append(MetricSample(
                        name=mn.NODE_DISK_IO_TIME_SECONDS_TOTAL,
                        labels={"device": device_dir.name},
                        value=io_time_ms / 1000.0,
                    ))
            except Exception:
                logger.warning(
                    "Failed to read disk stat for %s", device_dir.name, exc_info=True,
                )
        return samples
