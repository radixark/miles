from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from tests.fast.utils.ft.conftest import FakeKmsgReader

from miles.utils.ft.agents.collectors.base import CollectorOutput
from miles.utils.ft.agents.collectors.kmsg import (
    KmsgCollector,
    _build_xid_samples,
    _count_kernel_events,
    _kernel_event_samples,
    _parse_xid_codes,
    _prune_xid_window,
)
from miles.utils.ft.agents.types import CounterSample, GaugeSample


def _make_kmsg_collector(lines: list[str], **kwargs: Any) -> KmsgCollector:
    collector = KmsgCollector(kmsg_path=Path("/dev/null"), **kwargs)
    collector._reader = FakeKmsgReader(lines)
    return collector


def _filter_metrics(result: CollectorOutput, name: str) -> list[GaugeSample | CounterSample]:
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
        events: deque[tuple[datetime, int]] = deque(
            [
                (now - timedelta(seconds=120), 48),
                (now - timedelta(seconds=30), 31),
            ]
        )

        _prune_xid_window(events, window=timedelta(seconds=60), now=now)

        assert len(events) == 1
        assert events[0][1] == 31

    def test_keeps_all_when_within_window(self) -> None:
        now = datetime.now(timezone.utc)
        events: deque[tuple[datetime, int]] = deque(
            [
                (now - timedelta(seconds=10), 48),
                (now - timedelta(seconds=5), 31),
            ]
        )

        _prune_xid_window(events, window=timedelta(seconds=60), now=now)

        assert len(events) == 2

    def test_removes_all_when_expired(self) -> None:
        now = datetime.now(timezone.utc)
        events: deque[tuple[datetime, int]] = deque(
            [
                (now - timedelta(seconds=200), 48),
                (now - timedelta(seconds=100), 31),
            ]
        )

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
            non_auto_recoverable_count=1,
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
            non_auto_recoverable_count=0,
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
            non_auto_recoverable_count=2,
        )

        counter = [s for s in samples if s.name == "miles_ft_xid_count_total"]
        assert len(counter) == 1
        assert isinstance(counter[0], CounterSample)
        assert counter[0].delta == 5.0

    def test_non_auto_recoverable_counter(self) -> None:
        samples = _build_xid_samples(
            current_codes={48},
            prev_codes=set(),
            new_count=3,
            non_auto_recoverable_count=2,
        )

        counter = [s for s in samples if s.name == "miles_ft_xid_non_auto_recoverable_count_total"]
        assert len(counter) == 1
        assert isinstance(counter[0], CounterSample)
        assert counter[0].delta == 2.0

    def test_non_auto_recoverable_counter_zero_when_no_critical(self) -> None:
        samples = _build_xid_samples(
            current_codes={31},
            prev_codes=set(),
            new_count=1,
            non_auto_recoverable_count=0,
        )

        counter = [s for s in samples if s.name == "miles_ft_xid_non_auto_recoverable_count_total"]
        assert len(counter) == 1
        assert isinstance(counter[0], CounterSample)
        assert counter[0].delta == 0.0


class TestCountKernelEvents:
    def test_kernel_panic(self) -> None:
        assert (
            _count_kernel_events(
                [
                    "Kernel panic - not syncing: Fatal exception",
                ]
            )
            == 1
        )

    def test_mce(self) -> None:
        assert (
            _count_kernel_events(
                [
                    "MCE: CPU 0: Machine Check Exception: 4 Bank 5",
                ]
            )
            == 1
        )

    def test_oom_killer(self) -> None:
        assert _count_kernel_events(["oom-killer invoked"]) == 1

    def test_hardware_error(self) -> None:
        assert _count_kernel_events(["HARDWARE ERROR detected"]) == 1

    def test_no_events(self) -> None:
        assert _count_kernel_events(["normal log message"]) == 0

    def test_multiple_events(self) -> None:
        assert (
            _count_kernel_events(
                [
                    "Kernel panic",
                    "normal line",
                    "MCE detected",
                ]
            )
            == 2
        )


class TestKernelEventSamples:
    def test_builds_counter_sample(self) -> None:
        samples = _kernel_event_samples(3)
        assert len(samples) == 1
        assert samples[0].name == "miles_ft_kernel_event_count"
        assert isinstance(samples[0], CounterSample)
        assert samples[0].delta == 3.0


# ---------------------------------------------------------------------------
# Integration tests (KmsgCollector.collect)
# ---------------------------------------------------------------------------


class TestKmsgCollectorXid:
    @pytest.mark.anyio
    async def test_xid_48_detected(self) -> None:
        collector = _make_kmsg_collector(
            [
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            ]
        )

        result = await collector.collect()
        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        assert len(xid_samples) == 1
        assert xid_samples[0].labels == {"xid": "48"}
        assert xid_samples[0].value == 1.0

    @pytest.mark.anyio
    async def test_multiple_xid_codes(self) -> None:
        collector = _make_kmsg_collector(
            [
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
                "NVRM: Xid (PCI:0000:5e:00): 31, pid=5678",
            ]
        )

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
        assert isinstance(count_sample[0], CounterSample)
        assert count_sample[0].delta == 0.0

    @pytest.mark.anyio
    async def test_xid_count_total(self) -> None:
        collector = _make_kmsg_collector(
            [
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=5678",
                "NVRM: Xid (PCI:0000:5e:00): 31, pid=9012",
            ]
        )

        result = await collector.collect()
        count = _filter_metrics(result, "miles_ft_xid_count_total")
        assert isinstance(count[0], CounterSample)
        assert count[0].delta == 3.0

    @pytest.mark.anyio
    async def test_non_auto_recoverable_counter_with_critical_xid(self) -> None:
        """XID 48 is non-auto-recoverable; XID 31 is not. Counter should reflect only XID 48."""
        collector = _make_kmsg_collector(
            [
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
                "NVRM: Xid (PCI:0000:5e:00): 31, pid=5678",
            ]
        )

        result = await collector.collect()
        counter = _filter_metrics(result, "miles_ft_xid_non_auto_recoverable_count_total")
        assert len(counter) == 1
        assert isinstance(counter[0], CounterSample)
        assert counter[0].delta == 1.0

    @pytest.mark.anyio
    async def test_non_auto_recoverable_counter_zero_for_benign_xids(self) -> None:
        collector = _make_kmsg_collector(
            [
                "NVRM: Xid (PCI:0000:3b:00): 31, pid=1234",
            ]
        )

        result = await collector.collect()
        counter = _filter_metrics(result, "miles_ft_xid_non_auto_recoverable_count_total")
        assert len(counter) == 1
        assert isinstance(counter[0], CounterSample)
        assert counter[0].delta == 0.0

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
        collector = _make_kmsg_collector(
            [
                "6,1234,5678;Kernel panic - not syncing: Fatal exception",
            ]
        )

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].delta == 1.0

    @pytest.mark.anyio
    async def test_mce_detected(self) -> None:
        collector = _make_kmsg_collector(
            [
                "MCE: CPU 0: Machine Check Exception: 4 Bank 5",
            ]
        )

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].delta == 1.0

    @pytest.mark.anyio
    async def test_no_kernel_events(self) -> None:
        collector = _make_kmsg_collector(
            [
                "Normal log message here",
            ]
        )

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].delta == 0.0


class TestKmsgCollectorDmesgFallback:
    @pytest.mark.anyio
    async def test_fallback_when_kmsg_unavailable(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))
        assert collector._reader is None

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n",
                },
            )()

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
            mock_run.return_value = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n",
                },
            )()
            await collector.collect()

        reader = collector._reader
        assert reader is not None
        time_after_success = reader._last_dmesg_time

        with patch("subprocess.run", side_effect=OSError("dmesg broken")):
            await collector.collect()

        assert reader._last_dmesg_time == time_after_success

    @pytest.mark.anyio
    async def test_dmesg_nonzero_returncode_does_not_advance_time(self) -> None:
        """Non-zero returncode must NOT advance _last_dmesg_time so that
        the failed window is retried on the next cycle, preventing
        permanent loss of kernel logs."""
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "",
                },
            )()
            await collector.collect()

        reader = collector._reader
        assert reader is not None
        time_after_first = reader._last_dmesg_time

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type(
                "Result",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                },
            )()
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


# ---------------------------------------------------------------------------
# P2 item 22: KmsgCollector additional edge cases
# ---------------------------------------------------------------------------


class TestCreateReaderFallback:
    @pytest.mark.anyio
    async def test_concurrent_xid_from_multiple_gpus_in_single_cycle(self) -> None:
        """Multiple XID events from different GPU PCIs in a single collection cycle."""
        collector = _make_kmsg_collector(
            [
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
                "NVRM: Xid (PCI:0000:5e:00): 48, pid=5678",
                "NVRM: Xid (PCI:0000:87:00): 31, pid=9012",
            ]
        )

        result = await collector.collect()
        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        xid_codes = {m.labels["xid"] for m in xid_samples}
        assert xid_codes == {"31", "48"}

        count_total = _filter_metrics(result, "miles_ft_xid_count_total")
        assert count_total[0].delta == 3.0

    def test_create_reader_falls_back_to_dmesg_on_oserror(self) -> None:
        """When /dev/kmsg fails to open, _create_reader() returns DmesgSubprocessReader."""
        from miles.utils.ft.agents.collectors.kernel_log_reader import DmesgSubprocessReader
        from miles.utils.ft.agents.collectors.kmsg import _create_reader

        reader = _create_reader(Path("/nonexistent/path/kmsg"))
        assert isinstance(reader, DmesgSubprocessReader)


class TestKmsgCollectorInterval:
    def test_default_collect_interval(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        assert collector.collect_interval == 2.0


# ---------------------------------------------------------------------------
# Real file I/O tests (no FakeKmsgReader)
# ---------------------------------------------------------------------------


class TestKmsgCollectorRealFile:
    """End-to-end: write XID/kernel-event strings to a tmp file,
    verify KmsgCollector parses them via real KmsgFileReader.

    KmsgFileReader does SEEK_END on creation, so we must call collect()
    once first to initialize the reader, then append new content, then
    collect() again to read it.
    """

    @pytest.mark.anyio
    async def test_xid_detected_from_real_file(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_text("")

        collector = KmsgCollector(kmsg_path=kmsg_file)
        await collector.collect()

        with open(kmsg_file, "a") as f:
            f.write("NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n")

        result = await collector.collect()
        xid_samples = _filter_metrics(result, "miles_ft_xid_code_recent")
        assert len(xid_samples) == 1
        assert xid_samples[0].labels == {"xid": "48"}
        assert xid_samples[0].value == 1.0

    @pytest.mark.anyio
    async def test_kernel_panic_detected_from_real_file(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_text("")

        collector = KmsgCollector(kmsg_path=kmsg_file)
        await collector.collect()

        with open(kmsg_file, "a") as f:
            f.write("Kernel panic - not syncing: Fatal exception\n")

        result = await collector.collect()
        kernel = _filter_metrics(result, "miles_ft_kernel_event_count")
        assert kernel[0].delta == 1.0

    @pytest.mark.anyio
    async def test_incremental_reads_across_collects(self, tmp_path: Path) -> None:
        """Two collect() cycles after init, each seeing only newly appended lines."""
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_text("")

        collector = KmsgCollector(kmsg_path=kmsg_file)
        await collector.collect()

        with open(kmsg_file, "a") as f:
            f.write("NVRM: Xid (PCI:0000:3b:00): 48, pid=1\n")
        r1 = await collector.collect()
        assert any(
            s.labels.get("xid") == "48" and s.value == 1.0 for s in r1.metrics if s.name == "miles_ft_xid_code_recent"
        )

        with open(kmsg_file, "a") as f:
            f.write("NVRM: Xid (PCI:0000:5e:00): 31, pid=2\n")
        r2 = await collector.collect()
        xid_codes = {s.labels["xid"] for s in r2.metrics if s.name == "miles_ft_xid_code_recent" and s.value == 1.0}
        assert xid_codes == {"48", "31"}

    @pytest.mark.anyio
    async def test_real_dmesg_fallback_does_not_crash(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))
        result = await collector.collect()
        assert result is not None
