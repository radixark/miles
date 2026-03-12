"""Unit tests for cli/diagnostics/cluster.py — _run_cluster_checks (P0 item 7)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.cli.diagnostics.cluster import _run_cluster_checks

pytestmark = pytest.mark.anyio


def _make_executor(*, bad_nodes: list[str] | None = None, raises: Exception | None = None) -> MagicMock:
    executor = MagicMock()
    if raises:
        executor.execute = AsyncMock(side_effect=raises)
    else:
        executor.execute = AsyncMock(return_value=bad_nodes or [])
    return executor


class TestRunClusterChecksExceptionHandling:
    async def test_executor_exception_produces_fail_result(self) -> None:
        registry = {"nccl_check": _make_executor(raises=RuntimeError("timeout"))}
        results = await _run_cluster_checks(
            registry=registry,
            node_agents={},
            checks=["nccl_check"],
            timeout=60,
        )

        assert len(results) == 1
        assert results[0].passed is False
        assert "exception during check" in results[0].details


class TestRunClusterChecksPassResult:
    async def test_empty_bad_nodes_produces_pass_result(self) -> None:
        registry = {"gpu_check": _make_executor(bad_nodes=[])}
        results = await _run_cluster_checks(
            registry=registry,
            node_agents={"node-0": MagicMock()},
            checks=["gpu_check"],
            timeout=60,
        )

        assert len(results) == 1
        assert results[0].passed is True
        assert "all nodes healthy" in results[0].details


class TestRunClusterChecksFailResult:
    async def test_bad_nodes_produces_fail_with_node_list(self) -> None:
        registry = {"gpu_check": _make_executor(bad_nodes=["node-1", "node-3"])}
        results = await _run_cluster_checks(
            registry=registry,
            node_agents={"node-0": MagicMock(), "node-1": MagicMock(), "node-3": MagicMock()},
            checks=["gpu_check"],
            timeout=60,
        )

        assert len(results) == 1
        assert results[0].passed is False
        assert "node-1" in results[0].details
        assert "node-3" in results[0].details


class TestRunClusterChecksMixedResults:
    async def test_multiple_checks_with_mixed_outcomes(self) -> None:
        registry = {
            "healthy_check": _make_executor(bad_nodes=[]),
            "failing_check": _make_executor(bad_nodes=["node-2"]),
            "crashing_check": _make_executor(raises=ValueError("oops")),
        }
        results = await _run_cluster_checks(
            registry=registry,
            node_agents={},
            checks=["healthy_check", "failing_check", "crashing_check"],
            timeout=60,
        )

        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is False
        assert "node-2" in results[1].details
        assert results[2].passed is False
        assert "exception" in results[2].details
