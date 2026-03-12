from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import miles.utils.ft.utils.metric_names as mn
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.kernel_log_reader import DmesgSubprocessReader, KernelLogReader, KmsgFileReader
from miles.utils.ft.agents.types import CounterSample, GaugeSample
from miles.utils.ft.utils.gpu_constants import NON_AUTO_RECOVERABLE_XIDS

logger = logging.getLogger(__name__)

_XID_PATTERN = re.compile(r"NVRM: Xid.*?: (\d+)")
_KERNEL_EVENT_KEYWORDS = ("kernel panic", "mce", "oom-killer", "hardware error")


def _parse_xid_codes(lines: list[str]) -> list[int]:
    codes: list[int] = []
    for line in lines:
        match = _XID_PATTERN.search(line)
        if match:
            codes.append(int(match.group(1)))
    return codes


def _prune_xid_window(
    events: deque[tuple[datetime, int]],
    window: timedelta,
    now: datetime,
) -> None:
    cutoff = now - window
    while events and events[0][0] < cutoff:
        events.popleft()


def _build_xid_samples(
    current_codes: set[int],
    prev_codes: set[int],
    new_count: int,
    non_auto_recoverable_count: int,
) -> list[GaugeSample | CounterSample]:
    samples: list[GaugeSample | CounterSample] = [
        GaugeSample(
            name=mn.XID_CODE_RECENT,
            labels={"xid": str(code)},
            value=1.0,
        )
        for code in sorted(current_codes)
    ]

    for gone_code in prev_codes - current_codes:
        samples.append(
            GaugeSample(
                name=mn.XID_CODE_RECENT,
                labels={"xid": str(gone_code)},
                value=0.0,
            )
        )

    samples.append(
        CounterSample(
            name=mn.XID_COUNT_TOTAL,
            labels={},
            delta=float(new_count),
        )
    )
    samples.append(
        CounterSample(
            name=mn.XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
            labels={},
            delta=float(non_auto_recoverable_count),
        )
    )
    return samples


def _count_kernel_events(lines: list[str]) -> int:
    count = 0
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in _KERNEL_EVENT_KEYWORDS):
            count += 1
    return count


def _kernel_event_samples(count: int) -> list[CounterSample]:
    return [
        CounterSample(
            name=mn.KERNEL_EVENT_COUNT,
            labels={},
            delta=float(count),
        )
    ]


def _create_reader(kmsg_path: Path) -> KernelLogReader:
    try:
        return KmsgFileReader(kmsg_path=kmsg_path)
    except OSError:
        logger.info("Cannot open %s, falling back to dmesg", kmsg_path, exc_info=True)
        return DmesgSubprocessReader()


class KmsgCollector(BaseCollector):
    collect_interval: float = 2.0

    def __init__(
        self,
        *,
        xid_window_seconds: float = 300.0,
        kmsg_path: Path = Path("/dev/kmsg"),
        since: datetime | None = None,
    ) -> None:
        self._xid_window_seconds = xid_window_seconds
        self._kmsg_path = kmsg_path
        self._since = since

        self._xid_events: deque[tuple[datetime, int]] = deque()
        self._prev_xid_codes: set[int] = set()
        self._reader: KernelLogReader | None = None

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        await super().close()

    def _collect_sync(self) -> list[GaugeSample | CounterSample]:
        lines = self._read_lines()
        now = datetime.now(timezone.utc)

        new_xids = _parse_xid_codes(lines)
        for code in new_xids:
            self._xid_events.append((now, code))
        _prune_xid_window(
            self._xid_events,
            timedelta(seconds=self._xid_window_seconds),
            now,
        )

        non_auto_recoverable_count = sum(1 for code in new_xids if code in NON_AUTO_RECOVERABLE_XIDS)

        current_codes = {code for _, code in self._xid_events}
        samples = _build_xid_samples(
            current_codes,
            self._prev_xid_codes,
            len(new_xids),
            non_auto_recoverable_count,
        )
        self._prev_xid_codes = current_codes

        samples.extend(_kernel_event_samples(_count_kernel_events(lines)))
        return samples

    def _read_lines(self) -> list[str]:
        if self._reader is None:
            if self._since is not None:
                self._reader = DmesgSubprocessReader(since=self._since)
            else:
                self._reader = _create_reader(self._kmsg_path)
        return self._reader.read_new_lines()
