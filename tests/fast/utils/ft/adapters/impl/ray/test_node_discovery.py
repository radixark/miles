"""Tests for node discovery utilities (mocked ray.nodes)."""

from __future__ import annotations

from unittest.mock import patch

from miles.utils.ft.adapters.impl.ray.node_discovery import (
    build_node_address_map,
    get_alive_gpu_nodes,
    resolve_to_ray_node_ids,
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


class TestResolveToRayNodeIds:
    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_resolves_by_node_name(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = resolve_to_ray_node_ids(["k8s-node-0"])
        assert result == ["aaa"]

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_resolves_by_ip_address(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = resolve_to_ray_node_ids(["10.0.0.2"])
        assert result == ["bbb"]

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_resolves_by_ray_id_passthrough(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = resolve_to_ray_node_ids(["aaa"])
        assert result == ["aaa"]

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_deduplicates_resolved_ids(self, mock_ray) -> None:
        """Same node referenced by name and IP -> only one entry."""
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = resolve_to_ray_node_ids(["k8s-node-0", "10.0.0.1", "aaa"])
        assert result == ["aaa"]

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_skips_dead_nodes(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = resolve_to_ray_node_ids(["dead-gpu-node"])
        assert result == []

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_skips_unknown_identifiers(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _SAMPLE_NODES
        result = resolve_to_ray_node_ids(["unknown-host"])
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
