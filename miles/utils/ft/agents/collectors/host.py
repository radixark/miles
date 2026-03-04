from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import CollectorOutput, MetricSample

logger = logging.getLogger(__name__)

_XID_PATTERN = re.compile(r"NVRM: Xid.*?: (\d+)")
_KERNEL_EVENT_KEYWORDS = ("kernel panic", "MCE", "oom-killer", "Hardware Error")

_DISK_COLLECT_EVERY_N = 30


class HostCollector(BaseCollector):
    collect_interval: float = 2.0

    def __init__(
        self,
        xid_window_seconds: float = 300.0,
        disk_mounts: list[str] | None = None,
        kmsg_path: str = "/dev/kmsg",
    ) -> None:
        self._xid_window_seconds = xid_window_seconds
        self._disk_mounts = disk_mounts or []
        self._kmsg_path = kmsg_path

        self._xid_events: deque[tuple[datetime, int]] = deque()
        self._kmsg_reader: _KmsgReader | None = None
        self._collect_count = 0

    async def collect(self) -> CollectorOutput:
        self._collect_count += 1
        metrics = await asyncio.to_thread(self._collect_sync)
        return CollectorOutput(metrics=metrics)

    def _collect_sync(self) -> list[MetricSample]:
        samples: list[MetricSample] = []

        new_lines = self._read_kmsg_lines()
        now = datetime.now(timezone.utc)

        kernel_event_count = 0
        for line in new_lines:
            xid_match = _XID_PATTERN.search(line)
            if xid_match:
                xid_code = int(xid_match.group(1))
                self._xid_events.append((now, xid_code))

            line_lower = line.lower()
            for keyword in _KERNEL_EVENT_KEYWORDS:
                if keyword.lower() in line_lower:
                    kernel_event_count += 1
                    break

        cutoff = now.timestamp() - self._xid_window_seconds
        while self._xid_events and self._xid_events[0][0].timestamp() < cutoff:
            self._xid_events.popleft()

        distinct_codes: set[int] = set()
        for _ts, code in self._xid_events:
            distinct_codes.add(code)

        for code in sorted(distinct_codes):
            samples.append(MetricSample(
                name="xid_code_recent",
                labels={"xid": str(code)},
                value=1.0,
            ))

        samples.append(MetricSample(
            name="xid_count_recent",
            labels={},
            value=float(len(self._xid_events)),
        ))

        samples.append(MetricSample(
            name="kernel_event_count",
            labels={},
            value=float(kernel_event_count),
        ))

        if self._collect_count % _DISK_COLLECT_EVERY_N == 1 or _DISK_COLLECT_EVERY_N == 1:
            self._collect_disk(samples)

        return samples

    def _read_kmsg_lines(self) -> list[str]:
        if self._kmsg_reader is None:
            self._kmsg_reader = _KmsgReader(kmsg_path=self._kmsg_path)
        return self._kmsg_reader.read_new_lines()

    def _collect_disk(self, samples: list[MetricSample]) -> None:
        for mount in self._disk_mounts:
            try:
                stat = os.statvfs(mount)
                available_bytes = stat.f_bavail * stat.f_frsize
                samples.append(MetricSample(
                    name="disk_available_bytes",
                    labels={"mount": mount},
                    value=float(available_bytes),
                ))
            except Exception:
                logger.warning("Failed to statvfs %s", mount)

        self._collect_disk_io_errors(samples)

    def _collect_disk_io_errors(self, samples: list[MetricSample]) -> None:
        sys_block = Path("/sys/block")
        if not sys_block.exists():
            return

        for device_dir in sorted(sys_block.iterdir()):
            stat_file = device_dir / "stat"
            if not stat_file.exists():
                continue
            try:
                text = stat_file.read_text().strip()
                fields = text.split()
                if len(fields) >= 10:
                    io_errors = int(fields[9])
                    samples.append(MetricSample(
                        name="disk_io_errors",
                        labels={"device": device_dir.name},
                        value=float(io_errors),
                    ))
            except Exception:
                logger.warning("Failed to read disk stat for %s", device_dir.name)


class _KmsgReader:
    def __init__(self, kmsg_path: str = "/dev/kmsg") -> None:
        self._kmsg_path = kmsg_path
        self._file_handle: int | None = None
        self._use_dmesg_fallback = False
        self._last_dmesg_time: datetime | None = None

        self._try_open_kmsg()

    def _try_open_kmsg(self) -> None:
        try:
            fd = os.open(self._kmsg_path, os.O_RDONLY | os.O_NONBLOCK)
            os.lseek(fd, 0, os.SEEK_END)
            self._file_handle = fd
        except (OSError, PermissionError):
            logger.info("Cannot open %s, falling back to dmesg", self._kmsg_path)
            self._use_dmesg_fallback = True
            self._last_dmesg_time = datetime.now(timezone.utc)

    def read_new_lines(self) -> list[str]:
        if self._use_dmesg_fallback:
            return self._read_via_dmesg()
        return self._read_via_kmsg()

    def _read_via_kmsg(self) -> list[str]:
        if self._file_handle is None:
            return []

        lines: list[str] = []
        while True:
            try:
                data = os.read(self._file_handle, 8192)
                if not data:
                    break
                lines.extend(data.decode("utf-8", errors="replace").splitlines())
            except BlockingIOError:
                break
            except OSError:
                logger.warning("Error reading /dev/kmsg, switching to dmesg fallback")
                self._close_kmsg()
                self._use_dmesg_fallback = True
                self._last_dmesg_time = datetime.now(timezone.utc)
                break

        return lines

    def _read_via_dmesg(self) -> list[str]:
        if self._last_dmesg_time is None:
            self._last_dmesg_time = datetime.now(timezone.utc)
            return []

        since_str = self._last_dmesg_time.strftime("%Y-%m-%d %H:%M:%S")
        self._last_dmesg_time = datetime.now(timezone.utc)

        try:
            result = subprocess.run(
                ["dmesg", "--since", since_str],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip().splitlines()
        except Exception:
            logger.warning("dmesg fallback failed")

        return []

    def _close_kmsg(self) -> None:
        if self._file_handle is not None:
            try:
                os.close(self._file_handle)
            except OSError:
                pass
            self._file_handle = None
