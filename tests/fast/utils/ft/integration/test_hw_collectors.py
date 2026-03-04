from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.host import HostCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.node_agent import FtNodeAgent
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig


def _make_mock_pynvml(device_count: int = 2) -> MagicMock:
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0
    mock.NVML_PCIE_UTIL_TX_BYTES = 1
    mock.nvmlInit.return_value = None
    mock.nvmlShutdown.return_value = None
    mock.nvmlDeviceGetCount.return_value = device_count
    mock.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle-{i}"
    mock.nvmlDeviceGetTemperature.return_value = 70
    mock.nvmlDeviceGetRemappedRows.return_value = (0, 0, 0, 0)
    mock.nvmlDeviceGetPcieThroughput.return_value = 1048576
    mock.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=45)
    return mock


def _create_sysfs(tmp_path: Path) -> Path:
    sysfs = tmp_path / "sysfs_net"
    for name in ["ib0", "ib1"]:
        iface = sysfs / name
        iface.mkdir(parents=True)
        (iface / "operstate").write_text("up\n")
        stats = iface / "statistics"
        stats.mkdir()
        for stat in ["rx_errors", "tx_errors", "rx_dropped", "tx_dropped"]:
            (stats / stat).write_text("0\n")
    return sysfs


class _FakeKmsgReader:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._consumed = False

    def read_new_lines(self) -> list[str]:
        if self._consumed:
            return []
        self._consumed = True
        return list(self._lines)


class TestNodeAgentAllCollectorsIntegration:
    @pytest.mark.asyncio()
    async def test_all_collectors_expose_metrics(self, tmp_path: Path) -> None:
        mock_pynvml = _make_mock_pynvml(device_count=2)
        sysfs = _create_sysfs(tmp_path)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            gpu_collector = GpuCollector()
            gpu_collector.collect_interval = 0.05

            host_collector = HostCollector(
                kmsg_path="/dev/null",
                disk_mounts=[str(tmp_path)],
            )
            host_collector.collect_interval = 0.05
            host_collector._kmsg_reader = _FakeKmsgReader([
                "NVRM: Xid (PCI:0000:3b:00): 48, pid=1234",
            ])

            network_collector = NetworkCollector(sysfs_net_path=str(sysfs))
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

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML gone")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            gpu_collector = GpuCollector()
            gpu_collector.collect_interval = 0.05

        network_collector = NetworkCollector(sysfs_net_path=str(sysfs))
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

        host_collector = HostCollector(kmsg_path="/dev/null")
        host_collector.collect_interval = 0.05
        host_collector._kmsg_reader = _FakeKmsgReader([])

        network_collector = NetworkCollector(sysfs_net_path=str(sysfs))
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
