from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from tests.fast.utils.ft.conftest import FakeKmsgReader


class TestKmsgCollectorXid:
    @pytest.mark.asyncio()
    async def test_xid_48_detected(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        collector._reader = FakeKmsgReader([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
        ])

        result = await collector.collect()
        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 1
        assert xid_samples[0].labels == {"xid": "48"}
        assert xid_samples[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_multiple_xid_codes(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        collector._reader = FakeKmsgReader([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            "NVRM: Xid (PCI:0000:5e:00): 31, pid=5678",
        ])

        result = await collector.collect()
        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 2
        xid_codes = {m.labels["xid"] for m in xid_samples}
        assert xid_codes == {"31", "48"}

    @pytest.mark.asyncio()
    async def test_xid_window_expiry(self) -> None:
        collector = KmsgCollector(
            kmsg_path=Path("/dev/null"),
            xid_window_seconds=60.0,
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        collector._xid_events.append((old_time, 48))

        collector._reader = FakeKmsgReader([])
        result = await collector.collect()

        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 0

        count_sample = [m for m in result.metrics if m.name == "miles_ft_xid_count_total"]
        assert count_sample[0].value == 0.0
        assert count_sample[0].metric_type == "counter"

    @pytest.mark.asyncio()
    async def test_xid_count_total(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        collector._reader = FakeKmsgReader([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=5678",
            "NVRM: Xid (PCI:0000:5e:00): 31, pid=9012",
        ])

        result = await collector.collect()
        count = [m for m in result.metrics if m.name == "miles_ft_xid_count_total"]
        assert count[0].value == 3.0
        assert count[0].metric_type == "counter"


class TestKmsgCollectorKernelEvents:
    @pytest.mark.asyncio()
    async def test_kernel_panic_detected(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        collector._reader = FakeKmsgReader([
            "6,1234,5678;Kernel panic - not syncing: Fatal exception",
        ])

        result = await collector.collect()
        kernel = [m for m in result.metrics if m.name == "miles_ft_kernel_event_count"]
        assert kernel[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_mce_detected(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        collector._reader = FakeKmsgReader([
            "MCE: CPU 0: Machine Check Exception: 4 Bank 5",
        ])

        result = await collector.collect()
        kernel = [m for m in result.metrics if m.name == "miles_ft_kernel_event_count"]
        assert kernel[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_no_kernel_events(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        collector._reader = FakeKmsgReader([
            "Normal log message here",
        ])

        result = await collector.collect()
        kernel = [m for m in result.metrics if m.name == "miles_ft_kernel_event_count"]
        assert kernel[0].value == 0.0


class TestKmsgCollectorDmesgFallback:
    @pytest.mark.asyncio()
    async def test_fallback_when_kmsg_unavailable(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/nonexistent/kmsg"))
        assert collector._reader is None

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("Result", (), {
                "returncode": 0,
                "stdout": "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n",
            })()

            result = await collector.collect()

        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 1
        assert xid_samples[0].labels["xid"] == "48"


class TestDmesgTimeWindowBug:
    @pytest.mark.asyncio()
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

    @pytest.mark.asyncio()
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


class TestKmsgCollectorInterval:
    def test_default_collect_interval(self) -> None:
        collector = KmsgCollector(kmsg_path=Path("/dev/null"))
        assert collector.collect_interval == 2.0
