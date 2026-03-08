from __future__ import annotations

from pathlib import Path

import pytest
from tests.fast.utils.ft.utils import create_sysfs_interface

from miles.utils.ft.agents.collectors.network import NetworkCollector


class TestNetworkCollector:
    async def test_nic_up(self, tmp_path: Path) -> None:
        create_sysfs_interface(tmp_path, "ib0", operstate="up")
        collector = NetworkCollector(sysfs_net_path=tmp_path)

        result = await collector.collect()
        nic_up = [m for m in result.metrics if m.name == "miles_ft_node_network_up"]
        assert len(nic_up) == 1
        assert nic_up[0].labels == {"device": "ib0"}
        assert nic_up[0].value == 1.0

    async def test_nic_down(self, tmp_path: Path) -> None:
        create_sysfs_interface(tmp_path, "eth0", operstate="down")
        collector = NetworkCollector(sysfs_net_path=tmp_path)

        result = await collector.collect()
        nic_up = [m for m in result.metrics if m.name == "miles_ft_node_network_up"]
        assert len(nic_up) == 1
        assert nic_up[0].value == 0.0

    async def test_missing_sysfs_dir(self, tmp_path: Path) -> None:
        collector = NetworkCollector(sysfs_net_path=tmp_path / "nonexistent")
        result = await collector.collect()
        assert result.metrics == []

    async def test_missing_operstate_file(self, tmp_path: Path) -> None:
        iface_dir = tmp_path / "ib0"
        iface_dir.mkdir()

        collector = NetworkCollector(sysfs_net_path=tmp_path)
        result = await collector.collect()
        nic_up = [m for m in result.metrics if m.name == "miles_ft_node_network_up"]
        assert len(nic_up) == 0

    async def test_interface_filtering_excludes_lo_docker_veth(self, tmp_path: Path) -> None:
        create_sysfs_interface(tmp_path, "lo", operstate="up")
        create_sysfs_interface(tmp_path, "docker0", operstate="up")
        create_sysfs_interface(tmp_path, "veth1234", operstate="up")
        create_sysfs_interface(tmp_path, "ib0", operstate="up")

        collector = NetworkCollector(sysfs_net_path=tmp_path)
        result = await collector.collect()

        interfaces = {m.labels["device"] for m in result.metrics if "device" in m.labels}
        assert "ib0" in interfaces
        assert "lo" not in interfaces
        assert "docker0" not in interfaces
        assert "veth1234" not in interfaces

    async def test_multiple_interfaces(self, tmp_path: Path) -> None:
        create_sysfs_interface(tmp_path, "ib0", operstate="up", rx_errors=10)
        create_sysfs_interface(tmp_path, "ib1", operstate="up", tx_errors=5)
        create_sysfs_interface(tmp_path, "eth0", operstate="down", rx_dropped=100)

        collector = NetworkCollector(sysfs_net_path=tmp_path)
        result = await collector.collect()

        ib0_rx = [
            m
            for m in result.metrics
            if m.name == "miles_ft_node_network_receive_errs_total" and m.labels.get("device") == "ib0"
        ]
        assert len(ib0_rx) == 1
        assert ib0_rx[0].value == 10.0

        ib1_tx = [
            m
            for m in result.metrics
            if m.name == "miles_ft_node_network_transmit_errs_total" and m.labels.get("device") == "ib1"
        ]
        assert len(ib1_tx) == 1
        assert ib1_tx[0].value == 5.0

        eth0_up = [
            m for m in result.metrics if m.name == "miles_ft_node_network_up" and m.labels.get("device") == "eth0"
        ]
        assert eth0_up[0].value == 0.0

    async def test_statistics_values(self, tmp_path: Path) -> None:
        create_sysfs_interface(
            tmp_path,
            "ib0",
            operstate="up",
            rx_errors=42,
            tx_errors=7,
            rx_dropped=3,
            tx_dropped=1,
        )
        collector = NetworkCollector(sysfs_net_path=tmp_path)
        result = await collector.collect()

        metrics_by_name = {m.name: m.value for m in result.metrics}
        assert metrics_by_name["miles_ft_node_network_receive_errs_total"] == 42.0
        assert metrics_by_name["miles_ft_node_network_transmit_errs_total"] == 7.0
        assert metrics_by_name["miles_ft_node_network_receive_drop_total"] == 3.0
        assert metrics_by_name["miles_ft_node_network_transmit_drop_total"] == 1.0

    async def test_custom_include_patterns(self, tmp_path: Path) -> None:
        create_sysfs_interface(tmp_path, "bond0", operstate="up")
        create_sysfs_interface(tmp_path, "ib0", operstate="up")

        collector = NetworkCollector(
            sysfs_net_path=tmp_path,
            interface_patterns=["bond*"],
        )
        result = await collector.collect()

        interfaces = {m.labels["device"] for m in result.metrics if "device" in m.labels}
        assert "bond0" in interfaces
        assert "ib0" not in interfaces

    def test_default_collect_interval(self) -> None:
        collector = NetworkCollector(sysfs_net_path=Path("/nonexistent"))
        assert collector.collect_interval == 30.0


class TestNetworkCollectorRealHardware:
    """Zero-mock tests against real /sys/class/net."""

    @pytest.mark.anyio
    async def test_collect_returns_network_metrics(self) -> None:
        collector = NetworkCollector()
        result = await collector.collect()

        names = {s.name for s in result.metrics}
        assert "miles_ft_node_network_up" in names

    @pytest.mark.anyio
    async def test_infiniband_interfaces_detected(self) -> None:
        """H200 nodes should have InfiniBand interfaces visible in sysfs."""
        collector = NetworkCollector(interface_patterns=["ib*"])
        result = await collector.collect()
        ib_devices = {s.labels["device"] for s in result.metrics if s.labels.get("device", "").startswith("ib")}
        assert len(ib_devices) > 0, "H200 node should have InfiniBand interfaces"
        up_samples = [s for s in result.metrics if s.name == "miles_ft_node_network_up"]
        assert len(up_samples) > 0
