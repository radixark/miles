"""Tests for cpu-only scheduling helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from miles.utils.ft.adapters.impl.ray.node_discovery import CPU_ONLY_RESOURCE
from miles.utils.ft.factories.scheduling import get_cpu_only_scheduling_options

_NODES_WITH_CPU_ONLY = [
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
]


class TestGetCpuOnlySchedulingOptions:
    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_returns_correct_resources(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _NODES_WITH_CPU_ONLY
        result = get_cpu_only_scheduling_options()
        assert "resources" in result
        assert result["resources"] == {CPU_ONLY_RESOURCE: 0.001}

    @patch("miles.utils.ft.adapters.impl.ray.node_discovery.ray")
    def test_raises_when_no_cpu_only_node(self, mock_ray) -> None:
        mock_ray.nodes.return_value = _NODES_WITHOUT_CPU_ONLY
        with pytest.raises(RuntimeError, match="cpu_only"):
            get_cpu_only_scheduling_options()
