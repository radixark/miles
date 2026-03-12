"""Tests for GPU cluster executor and hash outlier detection."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import FakeNodeAgent

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.diagnostics.executors.gpu import GpuClusterExecutor, _find_gpu_hash_outlier_nodes


def _make_result(
    node_id: str,
    *,
    passed: bool = True,
    compute_hashes: dict[str, str] | None = None,
) -> DiagnosticResult:
    metadata = {"compute_hashes": compute_hashes} if compute_hashes else None
    return DiagnosticResult(
        diagnostic_type="gpu",
        node_id=node_id,
        passed=passed,
        details="ok" if passed else "fail",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# find_gpu_hash_outlier_nodes
# ---------------------------------------------------------------------------


class TestFindGpuHashOutlierNodes:
    def test_all_same_hash_returns_empty(self) -> None:
        results = {
            "n0": _make_result("n0", compute_hashes={"0": "aaa", "1": "bbb"}),
            "n1": _make_result("n1", compute_hashes={"0": "aaa", "1": "bbb"}),
            "n2": _make_result("n2", compute_hashes={"0": "aaa", "1": "bbb"}),
        }
        assert _find_gpu_hash_outlier_nodes(results) == []

    def test_single_outlier_detected(self) -> None:
        results = {
            "n0": _make_result("n0", compute_hashes={"0": "aaa"}),
            "n1": _make_result("n1", compute_hashes={"0": "aaa"}),
            "n2": _make_result("n2", compute_hashes={"0": "xxx"}),
        }
        assert _find_gpu_hash_outlier_nodes(results) == ["n2"]

    def test_no_majority_skipped(self) -> None:
        """When hashes split 50/50, no node should be flagged (possible non-determinism)."""
        results = {
            "n0": _make_result("n0", compute_hashes={"0": "aaa"}),
            "n1": _make_result("n1", compute_hashes={"0": "bbb"}),
        }
        assert _find_gpu_hash_outlier_nodes(results) == []

    def test_fewer_than_two_nodes_returns_empty(self) -> None:
        results = {
            "n0": _make_result("n0", compute_hashes={"0": "aaa"}),
        }
        assert _find_gpu_hash_outlier_nodes(results) == []

    def test_no_metadata_returns_empty(self) -> None:
        results = {
            "n0": _make_result("n0"),
            "n1": _make_result("n1"),
        }
        assert _find_gpu_hash_outlier_nodes(results) == []

    def test_multiple_gpu_indices_outliers_combined(self) -> None:
        results = {
            "n0": _make_result("n0", compute_hashes={"0": "aaa", "1": "bbb"}),
            "n1": _make_result("n1", compute_hashes={"0": "aaa", "1": "bbb"}),
            "n2": _make_result("n2", compute_hashes={"0": "xxx", "1": "bbb"}),
            "n3": _make_result("n3", compute_hashes={"0": "aaa", "1": "yyy"}),
        }
        outliers = _find_gpu_hash_outlier_nodes(results)
        assert sorted(outliers) == ["n2", "n3"]

    def test_empty_hash_ignored(self) -> None:
        results = {
            "n0": _make_result("n0", compute_hashes={"0": "aaa"}),
            "n1": _make_result("n1", compute_hashes={"0": "aaa"}),
            "n2": _make_result("n2", compute_hashes={"0": ""}),
        }
        assert _find_gpu_hash_outlier_nodes(results) == []


# ---------------------------------------------------------------------------
# GpuClusterExecutor.execute
# ---------------------------------------------------------------------------


class TestGpuClusterExecutor:
    @pytest.mark.asyncio
    async def test_all_pass_no_outliers(self) -> None:
        node_agents = {
            "n0": FakeNodeAgent(
                diagnostic_results={
                    "gpu": _make_result("n0", compute_hashes={"0": "aaa"}),
                },
                node_id="n0",
            ),
            "n1": FakeNodeAgent(
                diagnostic_results={
                    "gpu": _make_result("n1", compute_hashes={"0": "aaa"}),
                },
                node_id="n1",
            ),
        }
        executor = GpuClusterExecutor()
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=60)
        assert bad == []

    @pytest.mark.asyncio
    async def test_failed_node_plus_hash_outlier(self) -> None:
        node_agents = {
            "n0": FakeNodeAgent(
                diagnostic_results={
                    "gpu": _make_result("n0", passed=False),
                },
                node_id="n0",
            ),
            "n1": FakeNodeAgent(
                diagnostic_results={
                    "gpu": _make_result("n1", compute_hashes={"0": "aaa"}),
                },
                node_id="n1",
            ),
            "n2": FakeNodeAgent(
                diagnostic_results={
                    "gpu": _make_result("n2", compute_hashes={"0": "aaa"}),
                },
                node_id="n2",
            ),
            "n3": FakeNodeAgent(
                diagnostic_results={
                    "gpu": _make_result("n3", compute_hashes={"0": "xxx"}),
                },
                node_id="n3",
            ),
        }
        executor = GpuClusterExecutor()
        bad = await executor.execute(node_agents=node_agents, timeout_seconds=60)
        assert "n0" in bad
        assert "n3" in bad
