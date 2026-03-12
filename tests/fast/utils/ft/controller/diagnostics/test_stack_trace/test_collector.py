"""Tests for miles.utils.ft.controller.diagnostics.stack_trace.collector."""

from __future__ import annotations

import asyncio
import json

from tests.fast.utils.ft.utils import make_trace_result
from tests.fast.utils.ft.utils.diagnostic_fakes import FakeNodeAgent

from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyThread
from miles.utils.ft.controller.diagnostics.stack_trace.collector import collect_stack_trace_suspects


def _normal_threads() -> list[dict[str, object]]:
    return [
        PySpyThread(
            id=1,
            name="MainThread",
            active=True,
            owns_gil=False,
            frames=[{"name": "train", "filename": "train.py", "line": 10}],
        ).model_dump()
    ]


def _make_agent_with_trace(node_id: str, *, passed: bool = True, details: str = "[]") -> FakeNodeAgent:
    return FakeNodeAgent(
        node_id=node_id,
        diagnostic_results={"stack_trace": make_trace_result(node_id, passed=passed, details=details)},
    )


class TestCollectStackTraceSuspects:
    def test_empty_agents_returns_empty(self) -> None:
        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents={},
                rank_pids_provider=lambda nid: {},
                default_timeout_seconds=30,
            )
        )

        assert result == []

    def test_rank_pids_provider_exception_marks_node_as_suspect(self) -> None:
        node_agents = {"node-0": _make_agent_with_trace("node-0")}

        def _failing_provider(node_id: str) -> dict[int, int]:
            raise RuntimeError("cannot get pids")

        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents=node_agents,
                rank_pids_provider=_failing_provider,
                default_timeout_seconds=30,
            )
        )

        assert "node-0" in result

    def test_diagnostic_execution_failure_marks_node_as_suspect(self) -> None:
        node_agents = {"node-0": _make_agent_with_trace("node-0", passed=False, details="py-spy failed")}

        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents=node_agents,
                rank_pids_provider=lambda nid: {0: 1234},
                default_timeout_seconds=30,
            )
        )

        assert "node-0" in result

    def test_successful_collection_returns_aggregation_suspects(self) -> None:
        threads_json = json.dumps(_normal_threads())
        node_agents = {
            "node-0": _make_agent_with_trace("node-0", details=threads_json),
            "node-1": _make_agent_with_trace("node-1", details=threads_json),
            "node-2": _make_agent_with_trace("node-2", details=threads_json),
        }

        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents=node_agents,
                rank_pids_provider=lambda nid: {0: 1234},
                default_timeout_seconds=30,
            )
        )

        assert isinstance(result, list)
