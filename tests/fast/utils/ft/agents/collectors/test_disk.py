from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import miles.utils.ft.metric_names as mn
from miles.utils.ft.agents.collectors.disk import DiskCollector


class TestDiskCollector:
    async def test_disk_available_bytes(self, tmp_path: Path) -> None:
        collector = DiskCollector(disk_mounts=[tmp_path])

        result = await collector.collect()
        disk = [m for m in result.metrics if m.name == "miles_ft_node_filesystem_avail_bytes"]
        assert len(disk) == 1
        assert disk[0].labels == {"mountpoint": str(tmp_path)}
        assert disk[0].value > 0

    def test_default_collect_interval(self) -> None:
        collector = DiskCollector()
        assert collector.collect_interval == 60.0


# -------------------------------------------------------------------
# _collect_disk_io_time
# -------------------------------------------------------------------


@contextmanager
def _patch_sys_block(fake_path: Path) -> Iterator[None]:
    """Redirect ``Path("/sys/block")`` inside DiskCollector to *fake_path*."""
    real_path = Path

    def _factory(p: str) -> Path:  # type: ignore[type-arg]
        if p == "/sys/block":
            return fake_path
        return real_path(p)

    with patch("miles.utils.ft.agents._support.collectors.disk.Path", side_effect=_factory):
        yield


def _make_sysblock(tmp_path: Path, devices: dict[str, str]) -> Path:
    sys_block = tmp_path / "sys" / "block"
    sys_block.mkdir(parents=True)
    for device_name, stat_content in devices.items():
        device_dir = sys_block / device_name
        device_dir.mkdir()
        (device_dir / "stat").write_text(stat_content)
    return sys_block


class TestCollectDiskIoTime:
    def test_parses_io_time_and_converts_ms_to_seconds(self, tmp_path: Path) -> None:
        stat_line = "   1234    567    8910   1112  1314   1516  1718   1920  0  5000  2122"
        sys_block = _make_sysblock(tmp_path, {"sda": stat_line})

        collector = DiskCollector()
        with _patch_sys_block(sys_block):
            samples = collector._collect_disk_io_time()

        assert len(samples) == 1
        assert samples[0].name == mn.NODE_DISK_IO_TIME_SECONDS_TOTAL
        assert samples[0].labels == {"device": "sda"}
        assert samples[0].value == 5.0

    def test_multiple_devices_sorted(self, tmp_path: Path) -> None:
        stat = "0 0 0 0 0 0 0 0 0 2000 0"
        sys_block = _make_sysblock(tmp_path, {"nvme0n1": stat, "sda": stat})

        collector = DiskCollector()
        with _patch_sys_block(sys_block):
            samples = collector._collect_disk_io_time()

        assert len(samples) == 2
        device_names = [s.labels["device"] for s in samples]
        assert device_names == sorted(device_names)

    def test_fewer_than_10_fields_skipped(self, tmp_path: Path) -> None:
        sys_block = _make_sysblock(tmp_path, {"sda": "1 2 3 4 5 6 7 8 9"})

        collector = DiskCollector()
        with _patch_sys_block(sys_block):
            samples = collector._collect_disk_io_time()

        assert samples == []

    def test_sys_block_not_exists(self) -> None:
        collector = DiskCollector()
        with _patch_sys_block(Path("/nonexistent/sys/block")):
            samples = collector._collect_disk_io_time()

        assert samples == []

    def test_stat_read_error_gracefully_skipped(self, tmp_path: Path) -> None:
        sys_block = tmp_path / "sys" / "block"
        sys_block.mkdir(parents=True)
        device_dir = sys_block / "sda"
        device_dir.mkdir()

        collector = DiskCollector()
        with _patch_sys_block(sys_block):
            samples = collector._collect_disk_io_time()

        assert samples == []
