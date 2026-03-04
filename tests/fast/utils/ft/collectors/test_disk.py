from __future__ import annotations

from pathlib import Path

import pytest

from miles.utils.ft.agents.collectors.disk import DiskCollector


class TestDiskCollector:
    @pytest.mark.asyncio()
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
