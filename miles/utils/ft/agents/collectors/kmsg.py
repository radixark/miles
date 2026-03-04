from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import miles.utils.ft.metric_names as mn
from miles.utils.ft.agents.collectors.kernel_log_reader import (
    DmesgSubprocessReader,
    KernelLogReader,
    KmsgFileReader,
)
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import MetricSample

logger = logging.getLogger(__name__)

_XID_PATTERN = re.compile(r"NVRM: Xid.*?: (\d+)")
_KERNEL_EVENT_KEYWORDS = ("kernel panic", "mce", "oom-killer", "hardware error")


class KmsgCollector(BaseCollector):
    collect_interval: float = 2.0

    def __init__(
        self,
        *,
        xid_window_seconds: float = 300.0,
        kmsg_path: Path = Path("/dev/kmsg"),
    ) -> None:
        self._xid_window_seconds = xid_window_seconds
        self._kmsg_path = kmsg_path

        self._xid_events: deque[tuple[datetime, int]] = deque()
        self._reader: KernelLogReader | None = None

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.close()

    def _collect_sync(self) -> list[MetricSample]:
        lines = self._read_lines()
        samples: list[MetricSample] = []
        samples.extend(self._process_xid_events(lines))
        samples.extend(self._count_kernel_events(lines))
        return samples

    def _read_lines(self) -> list[str]:
        if self._reader is None:
            self._reader = _create_reader(self._kmsg_path)

        try:
            return self._reader.read_new_lines()
        except OSError:
            logger.warning(
                "KmsgFileReader failed, switching to DmesgSubprocessReader",
                exc_info=True,
            )
            self._reader.close()
            self._reader = DmesgSubprocessReader()
            return self._reader.read_new_lines()

    def _process_xid_events(self, lines: list[str]) -> list[MetricSample]:
        now = datetime.now(timezone.utc)

        new_xid_count = 0
        for line in lines:
            match = _XID_PATTERN.search(line)
            if match:
                self._xid_events.append((now, int(match.group(1))))
                new_xid_count += 1

        cutoff = now.timestamp() - self._xid_window_seconds
        while self._xid_events and self._xid_events[0][0].timestamp() < cutoff:
            self._xid_events.popleft()

        samples = [
            MetricSample(
                name=mn.XID_CODE_RECENT,
                labels={"xid": str(code)},
                value=1.0,
            )
            for code in sorted({code for _, code in self._xid_events})
        ]
        samples.append(MetricSample(
            name=mn.XID_COUNT_TOTAL,
            labels={},
            value=float(new_xid_count),
            metric_type="counter",
        ))
        return samples

    def _count_kernel_events(self, lines: list[str]) -> list[MetricSample]:
        count = sum(
            1 for line in lines
            if any(kw in line.lower() for kw in _KERNEL_EVENT_KEYWORDS)
        )
        return [MetricSample(
            name=mn.KERNEL_EVENT_COUNT,
            labels={},
            value=float(count),
        )]


def _create_reader(kmsg_path: Path) -> KernelLogReader:
    try:
        return KmsgFileReader(kmsg_path=kmsg_path)
    except OSError:
        logger.info("Cannot open %s, falling back to dmesg", kmsg_path)
        return DmesgSubprocessReader()
