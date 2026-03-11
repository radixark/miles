"""Tests for miles.utils.ft.controller.diagnostics.executors.per_node."""

from __future__ import annotations

import asyncio

from tests.fast.utils.ft.conftest import make_fake_agents

from miles.utils.ft.controller.diagnostics.executors.per_node import PerNodeClusterExecutor


class TestPerNodeClusterExecutor:
    def test_all_pass_returns_empty(self) -> None:
        executor = PerNodeClusterExecutor(diagnostic_type="test_diag")
        agents = make_fake_agents(
            {
                "node-0": {"test_diag": True},
                "node-1": {"test_diag": True},
            }
        )

        bad_nodes = asyncio.run(executor.execute(agents=agents, timeout_seconds=30))

        assert bad_nodes == []

    def test_partial_failure_returns_failed_nodes(self) -> None:
        executor = PerNodeClusterExecutor(diagnostic_type="test_diag")
        agents = make_fake_agents(
            {
                "node-0": {"test_diag": True},
                "node-1": {"test_diag": False},
            }
        )

        bad_nodes = asyncio.run(executor.execute(agents=agents, timeout_seconds=30))

        assert bad_nodes == ["node-1"]

    def test_empty_agents_returns_empty(self) -> None:
        executor = PerNodeClusterExecutor(diagnostic_type="test_diag")

        bad_nodes = asyncio.run(executor.execute(agents={}, timeout_seconds=30))

        assert bad_nodes == []
