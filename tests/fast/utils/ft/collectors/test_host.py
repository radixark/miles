from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.ft.agents.collectors.host import HostCollector
from tests.fast.utils.ft.conftest import FakeKmsgReader


class TestHostCollectorXid:
    @pytest.mark.asyncio()
    async def test_xid_48_detected(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        collector._kmsg_reader = FakeKmsgReader([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
        ])

        result = await collector.collect()
        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 1
        assert xid_samples[0].labels == {"xid": "48"}
        assert xid_samples[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_multiple_xid_codes(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        collector._kmsg_reader = FakeKmsgReader([
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
        collector = HostCollector(
            kmsg_path=Path("/dev/null"),
            xid_window_seconds=60.0,
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        collector._xid_events.append((old_time, 48))

        collector._kmsg_reader = FakeKmsgReader([])
        result = await collector.collect()

        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 0

        count_sample = [m for m in result.metrics if m.name == "miles_ft_xid_count_total"]
        assert count_sample[0].value == 0.0
        assert count_sample[0].metric_type == "counter"

    @pytest.mark.asyncio()
    async def test_xid_count_total(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        collector._kmsg_reader = FakeKmsgReader([
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            "NVRM: Xid (PCI:0000:3b:00): 48, pid=5678",
            "NVRM: Xid (PCI:0000:5e:00): 31, pid=9012",
        ])

        result = await collector.collect()
        count = [m for m in result.metrics if m.name == "miles_ft_xid_count_total"]
        assert count[0].value == 3.0
        assert count[0].metric_type == "counter"


class TestHostCollectorKernelEvents:
    @pytest.mark.asyncio()
    async def test_kernel_panic_detected(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        collector._kmsg_reader = FakeKmsgReader([
            "6,1234,5678;Kernel panic - not syncing: Fatal exception",
        ])

        result = await collector.collect()
        kernel = [m for m in result.metrics if m.name == "miles_ft_kernel_event_count"]
        assert kernel[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_mce_detected(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        collector._kmsg_reader = FakeKmsgReader([
            "MCE: CPU 0: Machine Check Exception: 4 Bank 5",
        ])

        result = await collector.collect()
        kernel = [m for m in result.metrics if m.name == "miles_ft_kernel_event_count"]
        assert kernel[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_no_kernel_events(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        collector._kmsg_reader = FakeKmsgReader([
            "Normal log message here",
        ])

        result = await collector.collect()
        kernel = [m for m in result.metrics if m.name == "miles_ft_kernel_event_count"]
        assert kernel[0].value == 0.0


class TestHostCollectorDisk:
    @pytest.mark.asyncio()
    async def test_disk_available_bytes(self, tmp_path: Path) -> None:
        collector = HostCollector(
            kmsg_path=Path("/dev/null"),
            disk_mounts=[tmp_path],
        )
        collector._kmsg_reader = FakeKmsgReader([])

        result = await collector.collect()
        disk = [m for m in result.metrics if m.name == "miles_ft_node_filesystem_avail_bytes"]
        assert len(disk) == 1
        assert disk[0].labels == {"mountpoint": str(tmp_path)}
        assert disk[0].value > 0



class TestHostCollectorDmesgFallback:
    @pytest.mark.asyncio()
    async def test_fallback_when_kmsg_unavailable(self) -> None:
        collector = HostCollector(kmsg_path=Path("/nonexistent/kmsg"))
        assert collector._kmsg_reader is None

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("Result", (), {
                "returncode": 0,
                "stdout": "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234\n",
            })()

            result = await collector.collect()

        xid_samples = [m for m in result.metrics if m.name == "miles_ft_xid_code_recent"]
        assert len(xid_samples) == 1
        assert xid_samples[0].labels["xid"] == "48"


class TestHostCollectorInterval:
    def test_default_collect_interval(self) -> None:
        collector = HostCollector(kmsg_path=Path("/dev/null"))
        assert collector.collect_interval == 2.0
