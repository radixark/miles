from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from miles.utils.ft.agents.collectors.kmsg import (
    KmsgCollector,
    _build_xid_samples,
    _count_kernel_events,
    _kernel_event_samples,
    _parse_xid_codes,
    _prune_xid_window,
)
from miles.utils.ft.models.metrics import CollectorOutput, MetricSample
from tests.fast.utils.ft.conftest import FakeKmsgReader


def _make_kmsg_collector(lines: list[str], **kwargs: Any) -> KmsgCollector:
    collector = KmsgCollector(kmsg_path=Path("/dev/null"), **kwargs)
    collector._reader = FakeKmsgReader(lines)
    return collector


def _filter_metrics(result: CollectorOutput, name: str) -> list[MetricSample]:
    return [m for m in result.metrics if m.name == name]


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestParseXidCodes:
    def test_single_xid(self) -> None:
        lines = ["NVRM: Xid (PCI:0000:3b:00): 48, pid=1234"]
        assert _parse_xid_codes(lines) == [48]

    def test_multiple_xids(self) -> None:
        lines = [
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            "normal log line",
            "NVRM: Xid (PCI:0000:5e:00): 31, pid=5678",
        ]
        assert _parse_xid_codes(lines) == [48, 31]

    def test_no_xids(self) -> None:
        assert _parse_xid_codes(["normal log", "another line"]) == []

    def test_empty_lines(self) -> None:
        assert _parse_xid_codes([]) == []


class TestPruneXidWindow:
    def test_removes_old_events(self) -> None:
        now = datetime.now(timezone.utc)
        events: deque[tuple[datetime, int]] = deque([
            (now - timedelta(seconds=120), 48),
            (now - timedelta(seconds=30), 31),
        ])

        _prune_xid_window(events, window=timedelta(seconds=60), now=now)

        assert len(events) == 1
        assert events[0][1] == 31

    def test_keeps_all_when_within_window(self) -> None:
        now = datetime.now(timezone.utc)
        events: deque[tuple[datetime, int]] = deque([
            (now - timedelta(seconds=10), 48),
            (now - timedelta(seconds=5), 31),
        ])

        _prune_xid_window(events, window=timedelta(seconds=60), now=now)

        assert len(events) == 2

    def test_removes_all_when_expired(self) -> None:
        now = datetime.now(timezone.utc)
        events: deque[tuple[datetime, int]] = deque([
            (now - timedelta(seconds=200), 48),
            (now - timedelta(seconds=100), 31),
        ])

        _prune_xid_window(events, window=timedelta(seconds=60), now=now)

        assert len(events) == 0

    def test_empty_deque(self) -> None:
        now = datetime.now(timezone.utc)
        events: deque[tuple[datetime, int]] = deque()

        _prune_xid_window(events, window=timedelta(seconds=60), now=now)

        assert len(events) == 0


class TestBuildXidSamples:
    def test_active_codes_emitted_as_gauge_1(self) -> None:
        samples = _build_xid_samples(
            current_codes={48, 31},
            prev_codes=set(),
            new_count=2,
        )

        gauge_samples = [s for s in samples if s.name == "miles_ft_xid_code_recent"]
        assert len(gauge_samples) == 2
        codes = {s.labels["xid"] for s in gauge_samples}
        assert codes == {"31", "48"}
        assert all(s.value == 1.0 for s in gauge_samples)

    def test_gone_codes_emitted_as_gauge_0(self) -> None:
        samples = _build_xid_samples(
            current_codes=set(),
            prev_codes={48},
            new_count=0,
        )

        gauge_samples = [s for s in samples if s.name == "miles_ft_xid_code_recent"]
        assert len(gauge_samples) == 1
        assert gauge_samples[0].labels == {"xid": "48"}
        assert gauge_samples[0].value == 0.0

    def test_counter_reflects_new_count(self) -> None:
        samples = _build_xid_samples(
            current_codes={48},
            prev_codes=set(),
            new_count=5,
        )

        counter = [s for s in samples if s.name == "miles_ft_xid_count_total"]
        assert len(counter) == 1
        assert counter[0].value == 5.0
        assert counter[0].metric_type == "counter"


class TestCountKernelEvents:
    def test_kernel_panic(self) -> None:
        assert _count_kernel_events([
            "Kernel panic - not syncing: Fatal exception",
        ]) == 1

    def test_mce(self) -> None:
        assert _count_kernel_events([
            "MCE: CPU 0: Machine Check Exception: 4 Bank 5",
        ]) == 1

    def test_oom_killer(self) -> None:
        assert _count_kernel_events(["oom-killer invoked"]) == 1

    def test_hardware_error(self) -> None:
        assert _count_kernel_events(["HARDWARE ERROR detected"]) == 1

    def test_no_events(self) -> None:
        assert _count_kernel_events(["normal log message"]) == 0

    def test_multiple_events(self) -> None:
        assert _count_kernel_events([
            "Kernel panic",
            "normal line",
            "MCE detected",
        ]) == 2


class TestKernelEventSamples:
    def test_builds_counter_sample(self) -> None:
        samples = _kernel_event_samples(3)
        assert len(samples) == 1
        assert samples[0].name == "miles_ft_kernel_event_count"
        assert samples[0].value == 3.0
        assert samples[0].metric_type == "counter"


# ---------------------------------------------------------------------------
# Integration tests (KmsgCollector.collect)
# ---------------------------------------------------------------------------


class TestKmsgCollectorXid:
    @pytest.mark.anyio
    async def test_xid_48_detected(self) -> None:
        collector = _make_kmsg_collector([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
        ])

        result = await collector.collect()
        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        assert len(xid_samples) == 1
        assert xid_samples[0].labels == {"xid": "48"}
        assert xid_samples[0].value == 1.0

    @pytest.mark.anyio
    async def test_multiple_xid_codes(self) -> None:
        collector = _make_kmsg_collector([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            "NVRM: Xid (PCI:0000:5e:00): 31, pid=5678",
        ])

        result = await collector.collect()
        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        assert len(xid_samples) == 2
        xid_codes = {m.labels["xid"] for m in xid_samples}
        assert xid_codes == {"31", "48"}

    @pytest.mark.anyio
    async def test_xid_window_expiry(self) -> None:
        collector = _make_kmsg_collector([], xid_window_seconds=60.0)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        collector._xid_events.append((old_time, 48))

        result = await collector.collect()

        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        assert len(xid_samples) == 0

        count_sample = _filter_metrics(result, "miles_ft_xid_count_total")
        assert count_sample[0].value == 0.0
        assert count_sample[0].metric_type == "counter"

    @pytest.mark.anyio
    async def test_xid_count_total(self) -> None:
        collector = _make_kmsg_collector([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=5678",
            "NVRM: Xid (PCI:0000:5e:00): 31, pid=9012",
        ])

        result = await collector.collect()
        count = _filter_metrics(result, "miles_ft_xid_count_total")
        assert count[0].value == 3.0
        assert count[0].metric_type == "counter"

    @pytest.mark.anyio
    async def test_xid_stale_gauge_cleared(self) -> None:
        collector = _make_kmsg_collector(
            ["NVRM: Xid (PCI:0000:3b:00): 48, pid=1234"],
            xid_window_seconds=60.0,
        )

        result1 = await collector.collect()
        xid_samples_1 = _filter_metrics(result1, "miles_ft_xid_code_recent")
        assert len(xid_samples_1) == 1
        assert xid_samples_1[0].labels == {"xid": "48"}
        assert xid_samples_1[0].value == 1.0

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        collector._xid_events[0] = (old_time, collector._xid_events[0][1])

        result2 = await collector.collect()
        xid_samples_2 = _filter_metrics(result2, "miles_ft_xid_code_recent")
        assert len(xid_samples_2) == 1
        assert xid_samples_2[0].labels == {"xid": "48"}
        assert xid_samples_2[0].value == 0.0


class TestKmsgCollectorKernelEvents:
    @pytest.mark.anyio
    async def test_kernel_panic_detected(self) -> None:
        collector = _make_kmsg_collector([
            "6,1234,5678;Kernel panic - not syncing: Fatal exception",
        ])

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].value == 1.0

    @pytest.mark.anyio
    async def test_mce_detected(self) -> None:
        collector = _make_kmsg_collector([
            "MCE: CPU 0: Machine Check Exception: 4 Bank 5",
        ])

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].value == 1.0

    @pytest.mark.anyio
    async def test_no_kernel_events(self) -> None:
        collector = _make_kmsg_collector([
            "Normal log message here",
        ])

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].value == 0.0


class TestKmsgCollectorDmesgFallback:
    @pytest.mark.anyio
    async def test_fallback_when_kmsg_unavailable(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))
        assert collector._reader is None

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("Result", (), {
                "returncode": 0,
                "stdout": "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n",
            })()

            result = await collector.collect()

        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        assert len(xid_samples) == 1
        assert xid_samples[0].labels["xid"] == "48"


class TestDmesgTimeWindowBug:
    @pytest.mark.anyio
    async def test_dmesg_failure_does_not_advance_time_window(self) -> None:
        """If dmesg subprocess fails, _last_dmesg_time must NOT advance,
        so the same time window is retried on the next collection cycle.
        """
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))
        assert collector._reader is None

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("Result", (), {
                "returncode": 0,
                "stdout": "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n",
            })()
            await collector.collect()

        reader = collector._reader
        assert reader is not None
        time_after_success = reader._last_dmesg_time

        with patch("subprocess.run", side_effect=OSError("dmesg broken")):
            await collector.collect()

        assert reader._last_dmesg_time == time_after_success

    @pytest.mark.anyio
    async def test_dmesg_nonzero_returncode_does_not_advance(self) -> None:
        """A non-zero returncode from dmesg should also preserve the time window."""
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("Result", (), {
                "returncode": 0, "stdout": "",
            })()
            await collector.collect()

        reader = collector._reader
        assert reader is not None
        time_after_first = reader._last_dmesg_time

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("Result", (), {
                "returncode": 1, "stdout": "",
            })()
            await collector.collect()

        assert reader._last_dmesg_time == time_after_first


class TestKmsgCollectorReadLinesNoFallback:
    @pytest.mark.anyio
    async def test_reader_oserror_propagates(self) -> None:
        """Reader errors must propagate so the upper-level collection
        loop can log and retry."""
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        reader = FakeKmsgReader([])
        reader.read_new_lines = lambda: (_ for _ in ()).throw(OSError("fd gone"))  # type: ignore[assignment]
        collector._reader = reader

        with pytest.raises(OSError, match="fd gone"):
            collector._collect_sync()


class TestKmsgCollectorInterval:
    def test_default_collect_interval(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        assert collector.collect_interval == 2.0
