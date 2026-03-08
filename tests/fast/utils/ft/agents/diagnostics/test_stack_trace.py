"""Tests for StackTraceDiagnostic."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from tests.fast.utils.ft.utils import make_mock_subprocess

from miles.utils.ft.agents.diagnostics.stack_trace import StackTraceDiagnostic

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


class TestStackTraceDiagnosticEmptyPids:
    async def test_empty_pids_returns_failed(self) -> None:
        diag = StackTraceDiagnostic(pids=[])
        result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "no PIDs provided" in result.details

    async def test_none_pids_returns_failed(self) -> None:
        diag = StackTraceDiagnostic(pids=None)
        result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "no PIDs provided" in result.details


class TestStackTraceDiagnosticSinglePid:
    async def test_single_pid_success_returns_json_details(self) -> None:
        mock_proc = make_mock_subprocess(stdout=SAMPLE_PYSPY_JSON)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceDiagnostic(pids=[1234])
            result = await diag.run(node_id="node-0")

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
            diag = StackTraceDiagnostic(pids=[1234])
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert json.loads(result.details) == []


class TestStackTraceDiagnosticMultiplePids:
    async def test_partial_failure_still_passes(self) -> None:
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
            diag = StackTraceDiagnostic(pids=[100, 200])
            result = await diag.run(node_id="node-0")

        assert result.passed is True
        threads = json.loads(result.details)
        assert len(threads) == 1

    async def test_all_pids_fail_returns_not_passed(self) -> None:
        bad_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"error",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=bad_proc):
            diag = StackTraceDiagnostic(pids=[100, 200, 300])
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert json.loads(result.details) == []

    async def test_timeout_treated_as_failure(self) -> None:
        mock_proc = make_mock_subprocess()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceDiagnostic(pids=[1234])
            result = await diag.run(node_id="node-0", timeout_seconds=10)

        assert result.passed is False
        assert json.loads(result.details) == []
