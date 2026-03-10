"""Tests for node discovery utilities (mocked ray.nodes)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from miles.utils.ft.adapters.impl.ray.node_discovery import (
    assert_cpu_only_nodes_exist,
    build_node_address_map,
    get_alive_gpu_nodes,
)

_SAMPLE_NODES = [
    {
        "NodeID": "aaa",
        "NodeName": "k8s-node-0",
        "NodeManagerAddress": "10.0.0.1",
        "Alive": True,
        "Resources": {"GPU": 8, "CPU": 64},
    },
    {
        "NodeID": "bbb",
        "NodeName": "k8s-node-1",
        "NodeManagerAddress": "10.0.0.2",
        "Alive": True,
        "Resources": {"GPU": 4, "CPU": 32},
    },
    {
        "NodeID": "ccc",
        "NodeName": "head-node",
        "NodeManagerAddress": "10.0.0.3",
        "Alive": True,
        "Resources": {"CPU": 16},
    },
    {
        "NodeID": "ddd",
        "NodeName": "dead-gpu-node",
        "NodeManagerAddress": "10.0.0.4",
        "Alive": False,
        "Resources": {"GPU": 8, "CPU": 64},
    },
]


class TestGetAliveGpuNodes:
    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_returns_only_alive_gpu_nodes(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = get_alive_gpu_nodes()
        node_ids = {n["NodeID"] for n in result}
        assert node_ids == {"aaa", "bbb"}

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_filters_by_node_ids(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = get_alive_gpu_nodes(node_ids=["bbb"])
        assert len(result) == 1
        assert result[0]["NodeID"] == "bbb"

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_empty_when_no_match(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = get_alive_gpu_nodes(node_ids=["zzz"])
        assert result == []


class TestBuildNodeAddressMap:
    def test_builds_mapping(self) -> None:
        nodes = [
            {"NodeID": "aaa", "NodeManagerAddress": "10.0.0.1"},
            {"NodeID": "bbb", "NodeManagerAddress": "10.0.0.2"},
        ]
        result = build_node_address_map(nodes)
        assert result == {"aaa": "10.0.0.1", "bbb": "10.0.0.2"}

    def test_skips_nodes_without_address(self) -> None:
        nodes = [
            {"NodeID": "aaa", "NodeManagerAddress": "10.0.0.1"},
            {"NodeID": "bbb"},
        ]
        result = build_node_address_map(nodes)
        assert result == {"aaa": "10.0.0.1"}

    def test_empty_input(self) -> None:
        assert build_node_address_map([]) == {}


_NODES_WITH_CPU_ONLY = [
    {
        "NodeID": "aaa",
        "Alive": True,
        "Resources": {"GPU": 8, "CPU": 64},
    },
    {
        "NodeID": "head",
        "Alive": True,
        "Resources": {"CPU": 16, "cpu_only": 1},
    },
]

_NODES_WITHOUT_CPU_ONLY = [
    {
        "NodeID": "aaa",
        "Alive": True,
        "Resources": {"GPU": 8, "CPU": 64},
    },
    {
        "NodeID": "ccc",
        "Alive": True,
        "Resources": {"CPU": 16},
    },
]

_NODES_DEAD_CPU_ONLY = [
    {
        "NodeID": "head",
        "Alive": False,
        "Resources": {"CPU": 16, "cpu_only": 1},
    },
]


class TestAssertCpuOnlyNodesExist:
    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_passes_when_cpu_only_node_exists(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _NODES_WITH_CPU_ONLY
        assert_cpu_only_nodes_exist()

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_raises_when_no_cpu_only_node(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _NODES_WITHOUT_CPU_ONLY
        with pytest.raises(RuntimeError, match="cpu_only"):
            assert_cpu_only_nodes_exist()

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_raises_when_cpu_only_node_is_dead(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _NODES_DEAD_CPU_ONLY
        with pytest.raises(RuntimeError, match="cpu_only"):
            assert_cpu_only_nodes_exist()

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_raises_on_empty_cluster(self, mock_ray) -> None:
        mock_ray.nodes.return_value = []
        with pytest.raises(RuntimeError, match="cpu_only"):
            assert_cpu_only_nodes_exist()
