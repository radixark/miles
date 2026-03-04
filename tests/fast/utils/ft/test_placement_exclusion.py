from __future__ import annotations

import socket as socket_mod
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from miles.ray.placement_group import (
    _check_placement_has_excluded_nodes,
    _get_excluded_node_ids,
)


class TestCheckPlacementHasExcludedNodes:
    """Unit tests for the pure validation function (no Ray runtime needed)."""

    def test_no_exclusions_returns_empty(self) -> None:
        gpu_ids = [("10.0.0.1", "0"), ("10.0.0.1", "1"), ("10.0.0.2", "0")]
        result = _check_placement_has_excluded_nodes(gpu_ids=gpu_ids, excluded=set())
        assert result == set()

    def test_no_match_returns_empty(self) -> None:
        gpu_ids = [("10.0.0.1", "0"), ("10.0.0.2", "0")]
        excluded = {"10.0.0.99", "bad-host"}
        result = _check_placement_has_excluded_nodes(gpu_ids=gpu_ids, excluded=excluded)
        assert result == set()

    def test_ip_match_detected(self) -> None:
        gpu_ids = [("10.0.0.1", "0"), ("10.0.0.2", "0"), ("10.0.0.3", "0")]
        excluded = {"10.0.0.2"}
        result = _check_placement_has_excluded_nodes(gpu_ids=gpu_ids, excluded=excluded)
        assert result == {"10.0.0.2"}

    def test_multiple_matches(self) -> None:
        gpu_ids = [("10.0.0.1", "0"), ("10.0.0.2", "0"), ("10.0.0.3", "0")]
        excluded = {"10.0.0.1", "10.0.0.3"}
        result = _check_placement_has_excluded_nodes(gpu_ids=gpu_ids, excluded=excluded)
        assert result == {"10.0.0.1", "10.0.0.3"}

    def test_hostname_match_detected(self) -> None:
        gpu_ids = [("worker-01", "0"), ("worker-02", "0")]
        excluded = {"worker-01"}
        result = _check_placement_has_excluded_nodes(gpu_ids=gpu_ids, excluded=excluded)
        assert result == {"worker-01"}

    def test_empty_gpu_ids(self) -> None:
        result = _check_placement_has_excluded_nodes(gpu_ids=[], excluded={"10.0.0.1"})
        assert result == set()


def _make_mock_k8s_module(bad_nodes: list[str]) -> ModuleType:
    """Create a mock miles.utils.ft.platform.k8s_node_manager module."""
    mock_module = ModuleType("miles.utils.ft.platform.k8s_node_manager")
    mock_manager = AsyncMock()
    mock_manager.get_bad_nodes.return_value = bad_nodes
    mock_module.K8sNodeManager = MagicMock(return_value=mock_manager)  # type: ignore[attr-defined]
    return mock_module


class TestGetExcludedNodeIds:
    """Test K8s bad node query with mocked K8sNodeManager."""

    def test_returns_hostnames_and_resolved_ips(self) -> None:
        mock_module = _make_mock_k8s_module(["bad-node-1"])

        with (
            patch.dict(sys.modules, {"miles.utils.ft.platform.k8s_node_manager": mock_module}),
            patch(
                "miles.ray.placement_group.socket.gethostbyname",
                return_value="10.0.0.99",
            ),
        ):
            result = _get_excluded_node_ids()

        assert "bad-node-1" in result
        assert "10.0.0.99" in result

    def test_k8s_failure_returns_empty(self) -> None:
        mock_module = ModuleType("miles.utils.ft.platform.k8s_node_manager")
        mock_module.K8sNodeManager = MagicMock(side_effect=RuntimeError("no k8s"))  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"miles.utils.ft.platform.k8s_node_manager": mock_module}):
            result = _get_excluded_node_ids()

        assert result == set()

    def test_dns_resolution_failure_still_includes_hostname(self) -> None:
        mock_module = _make_mock_k8s_module(["unresolvable-host"])

        with (
            patch.dict(sys.modules, {"miles.utils.ft.platform.k8s_node_manager": mock_module}),
            patch(
                "miles.ray.placement_group.socket.gethostbyname",
                side_effect=socket_mod.gaierror("resolve failed"),
            ),
        ):
            result = _get_excluded_node_ids()

        assert "unresolvable-host" in result
        assert len(result) == 1

    def test_no_bad_nodes_returns_empty(self) -> None:
        mock_module = _make_mock_k8s_module([])

        with patch.dict(sys.modules, {"miles.utils.ft.platform.k8s_node_manager": mock_module}):
            result = _get_excluded_node_ids()

        assert result == set()

    def test_os_error_still_includes_hostname(self) -> None:
        mock_module = _make_mock_k8s_module(["host-with-os-error"])

        with (
            patch.dict(sys.modules, {"miles.utils.ft.platform.k8s_node_manager": mock_module}),
            patch(
                "miles.ray.placement_group.socket.gethostbyname",
                side_effect=OSError("network unreachable"),
            ),
        ):
            result = _get_excluded_node_ids()

        assert "host-with-os-error" in result
        assert len(result) == 1

    def test_multiple_bad_nodes(self) -> None:
        mock_module = _make_mock_k8s_module(["node-a", "node-b"])

        with (
            patch.dict(sys.modules, {"miles.utils.ft.platform.k8s_node_manager": mock_module}),
            patch(
                "miles.ray.placement_group.socket.gethostbyname",
                side_effect=lambda h: {"node-a": "10.0.0.1", "node-b": "10.0.0.2"}[h],
            ),
        ):
            result = _get_excluded_node_ids()

        assert result == {"node-a", "node-b", "10.0.0.1", "10.0.0.2"}
