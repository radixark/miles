from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.host import HostCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.node_agent import FtNodeAgent
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from tests.fast.utils.ft.conftest import (
    FakeKmsgReader,
    create_sysfs_interface,
    make_mock_pynvml,
)


def _create_sysfs(tmp_path: Path) -> Path:
    sysfs = tmp_path / "sysfs_net"
    for name in ["ib0", "ib1"]:
        create_sysfs_interface(sysfs, name, operstate="up")
    return sysfs


class TestNodeAgentAllCollectorsIntegration:
    @pytest.mark.asyncio()
    async def test_all_collectors_expose_metrics(self, tmp_path: Path) -> None:
        mock_nvml = make_mock_pynvml(device_count=2)
        sysfs = _create_sysfs(tmp_path)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            gpu_collector = GpuCollector()
            gpu_collector.collect_interval = 0.05

            host_collector = HostCollector(
                kmsg_path=Path("/dev/null"),
                disk_mounts=[tmp_path],
            )
            host_collector.collect_interval = 0.05
            host_collector._kmsg_reader = FakeKmsgReader([
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            ])

            network_collector = NetworkCollector(sysfs_net_path=sysfs)
            network_collector.collect_interval = 0.05

            agent = FtNodeAgent(
                node_id="integ-hw-node",
                collectors=[gpu_collector, host_collector, network_collector],
            )

            try:
                await agent.start()
                await asyncio.sleep(0.5)

                prom = MiniPrometheus(config=MiniPrometheusConfig())
                address = agent.get_exporter_address()
                prom.add_scrape_target(target_id="integ-hw-node", address=address)
                await prom.scrape_once()

                gpu_df = prom.instant_query("ft_node_gpu_available")
                assert not gpu_df.is_empty()

                gpu_temp = prom.instant_query("ft_node_gpu_temperature_celsius")
                assert not gpu_temp.is_empty()

                xid_df = prom.instant_query("ft_node_xid_code_recent")
                assert not xid_df.is_empty()

                xid_count = prom.instant_query("ft_node_xid_count_recent")
                assert not xid_count.is_empty()

                nic_df = prom.instant_query("ft_node_nic_up")
                assert not nic_df.is_empty()
                assert len(nic_df) == 2  # ib0 + ib1
            finally:
                await agent.stop()

    @pytest.mark.asyncio()
    async def test_failing_collector_does_not_block_others(self, tmp_path: Path) -> None:
        sysfs = _create_sysfs(tmp_path)

        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.side_effect = RuntimeError("NVML gone")
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            gpu_collector = GpuCollector()
            gpu_collector.collect_interval = 0.05

        network_collector = NetworkCollector(sysfs_net_path=sysfs)
        network_collector.collect_interval = 0.05

        agent = FtNodeAgent(
            node_id="integ-fail-node",
            collectors=[gpu_collector, network_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            prom = MiniPrometheus(config=MiniPrometheusConfig())
            address = agent.get_exporter_address()
            prom.add_scrape_target(target_id="integ-fail-node", address=address)
            await prom.scrape_once()

            nic_df = prom.instant_query("ft_node_nic_up")
            assert not nic_df.is_empty()
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_per_collector_interval_with_hw_collectors(self, tmp_path: Path) -> None:
        sysfs = _create_sysfs(tmp_path)

        host_collector = HostCollector(kmsg_path=Path("/dev/null"))
        host_collector.collect_interval = 0.05
        host_collector._kmsg_reader = FakeKmsgReader([])

        network_collector = NetworkCollector(sysfs_net_path=sysfs)
        network_collector.collect_interval = 0.3

        agent = FtNodeAgent(
            node_id="integ-interval-node",
            collectors=[host_collector, network_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.6)

            prom = MiniPrometheus(config=MiniPrometheusConfig())
            address = agent.get_exporter_address()
            prom.add_scrape_target(target_id="integ-interval-node", address=address)
            await prom.scrape_once()

            xid_count = prom.instant_query("ft_node_xid_count_recent")
            assert not xid_count.is_empty()

            nic_df = prom.instant_query("ft_node_nic_up")
            assert not nic_df.is_empty()
        finally:
            await agent.stop()
