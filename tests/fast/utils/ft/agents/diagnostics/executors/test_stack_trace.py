"""Tests for StackTraceNodeExecutor."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from tests.fast.utils.ft.utils import make_mock_subprocess

from miles.utils.ft.agents.diagnostics.executors.stack_trace import StackTraceNodeExecutor

SAMPLE_PYSPY_JSON = json.dumps(
    [
        {
            "thread_name": "MainThread",
            "thread_id": "0x7F1234",
            "active": True,
            "owns_gil": False,
            "frames": [
                {"name": "func_a", "filename": "file.py", "module": "mod", "line": 10, "locals": []},
            ],
        },
    ]
).encode()


class TestStackTraceNodeExecutorEmptyPids:
    async def test_empty_pids_returns_failed(self) -> None:
        diag = StackTraceNodeExecutor()
        result = await diag.run(node_id="node-0", pids=[])

        assert result.passed is False
        assert "no PIDs provided" in result.details

    async def test_none_pids_returns_failed(self) -> None:
        diag = StackTraceNodeExecutor()
        result = await diag.run(node_id="node-0", pids=None)

        assert result.passed is False
        assert "no PIDs provided" in result.details


class TestStackTraceNodeExecutorSinglePid:
    async def test_single_pid_success_returns_json_details(self) -> None:
        mock_proc = make_mock_subprocess(stdout=SAMPLE_PYSPY_JSON)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceNodeExecutor()
            result = await diag.run(node_id="node-0", pids=[1234])

        assert result.passed is True
        threads = json.loads(result.details)
        assert len(threads) == 1
        assert threads[0]["thread_name"] == "MainThread"
        assert threads[0]["frames"][0]["name"] == "func_a"

    async def test_single_pid_pyspy_failure(self) -> None:
        mock_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"process not found",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceNodeExecutor()
            result = await diag.run(node_id="node-0", pids=[1234])

        assert result.passed is False
        assert json.loads(result.details) == []


class TestStackTraceNodeExecutorMultiplePids:
    async def test_partial_failure_marks_not_passed(self) -> None:
        """Partial PID failure was treated as pass (only all_failed → not passed).
        Now any failure marks passed=False with failure metadata."""
        good_proc = make_mock_subprocess(stdout=SAMPLE_PYSPY_JSON)
        bad_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"error",
            returncode=1,
        )

        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return good_proc if call_count == 1 else bad_proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess):
            diag = StackTraceNodeExecutor()
            result = await diag.run(node_id="node-0", pids=[100, 200])

        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["failed_pids"] == 1
        assert result.metadata["total_pids"] == 2
        threads = json.loads(result.details)
        assert len(threads) == 1

    async def test_all_pids_fail_returns_not_passed(self) -> None:
        bad_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"error",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=bad_proc):
            diag = StackTraceNodeExecutor()
            result = await diag.run(node_id="node-0", pids=[100, 200, 300])

        assert result.passed is False
        assert json.loads(result.details) == []

    async def test_timeout_treated_as_failure(self) -> None:
        mock_proc = make_mock_subprocess()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceNodeExecutor()
            result = await diag.run(node_id="node-0", timeout_seconds=10, pids=[1234])

        assert result.passed is False
        assert json.loads(result.details) == []
