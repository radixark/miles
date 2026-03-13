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

    def test_invalid_json_marks_node_as_suspect(self) -> None:
        """M-6: json.loads on corrupt details used to raise unhandled
        JSONDecodeError, crashing the entire gather. Now caught and the
        node is marked as suspect."""
        node_agents = {"node-0": _make_agent_with_trace("node-0", details="not valid json{{")}

        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents=node_agents,
                rank_pids_provider=lambda nid: {0: 1234},
                default_timeout_seconds=30,
            )
        )

        assert "node-0" in result

    def test_invalid_model_marks_node_as_suspect(self) -> None:
        """M-6: if PySpyThread.model_validate fails (e.g. missing required
        fields), the node should be treated as suspect, not crash."""
        bad_thread_data = json.dumps([{"bad_field": "value"}])
        node_agents = {"node-0": _make_agent_with_trace("node-0", details=bad_thread_data)}

        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents=node_agents,
                rank_pids_provider=lambda nid: {0: 1234},
                default_timeout_seconds=30,
            )
        )

        assert "node-0" in result

    def test_empty_rank_pids_marks_node_as_suspect(self) -> None:
        """Previously an empty rank_pids dict caused a silent return with
        no suspect recorded. During a hang, if the training process has
        just exited and rank_pids_provider reads no PIDs, the node should
        still be treated as a diagnostic failure."""
        node_agents = {"node-0": _make_agent_with_trace("node-0")}

        result = asyncio.run(
            collect_stack_trace_suspects(
                node_agents=node_agents,
                rank_pids_provider=lambda nid: {},
                default_timeout_seconds=30,
            )
        )

        assert "node-0" in result

    def test_empty_threads_marks_node_as_suspect(self) -> None:
        """Previously when py-spy returned an empty thread list, the node
        was recorded as traces[node_id]=[] with an empty fingerprint "".
        This polluted aggregation by treating it as a valid (but vacuous)
        trace. Now empty threads are treated as diagnostic failure."""
        node_agents = {"node-0": _make_agent_with_trace("node-0", details="[]")}

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
