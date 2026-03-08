from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

from miles.utils.ft.adapters.types import NodeAgentProtocol
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl import NcclNodeExecutor
from miles.utils.ft.agents.types import DiagnosticPipelineResult, DiagnosticResult
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol

# ---------------------------------------------------------------------------
# Diagnostic test helpers
# ---------------------------------------------------------------------------


class StubDiagnostic(BaseNodeExecutor):
    """Test diagnostic that returns a configurable pass/fail result."""

    diagnostic_type = "stub"

    def __init__(
        self,
        passed: bool = True,
        details: str = "stub passed",
        diagnostic_type: str = "stub",
    ) -> None:
        self._passed = passed
        self._details = details
        self.diagnostic_type = diagnostic_type

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=self._passed,
            details=self._details,
        )


class SlowDiagnostic(BaseNodeExecutor):
    """Test diagnostic that sleeps longer than its timeout."""

    diagnostic_type = "slow"

    def __init__(self, sleep_seconds: float = 300.0) -> None:
        self._sleep_seconds = sleep_seconds

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        await asyncio.sleep(self._sleep_seconds)
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=True,
            details="should not reach here",
        )


# ---------------------------------------------------------------------------
# Diagnostic orchestrator fakes
# ---------------------------------------------------------------------------


class FakeDiagnosticOrchestrator(DiagnosticOrchestratorProtocol):
    """Programmable stub for DiagnosticOrchestrator in recovery tests."""

    def __init__(self, result: DiagnosticPipelineResult | None = None) -> None:
        self._result = result or DiagnosticPipelineResult(
            bad_node_ids=[],
            reason="fake diagnostic — all passed",
        )
        self.call_count: int = 0

    async def run_diagnostic_pipeline(
        self,
        pre_executors: object = None,
    ) -> DiagnosticPipelineResult:
        self.call_count += 1
        return self._result


class HangingDiagnosticOrchestrator(DiagnosticOrchestratorProtocol):
    """Orchestrator whose run_diagnostic_pipeline never returns (simulates hang)."""

    async def run_diagnostic_pipeline(
        self,
        pre_executors: object = None,
    ) -> DiagnosticPipelineResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Agent test helpers (node agents for diagnostic scheduling)
# ---------------------------------------------------------------------------


class FakeNodeAgent(NodeAgentProtocol):
    def __init__(
        self,
        diagnostic_results: dict[str, DiagnosticResult] | None = None,
        node_id: str = "fake",
    ) -> None:
        self._diagnostic_results = diagnostic_results or {}
        self._node_id = node_id

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        result = self._diagnostic_results.get(diagnostic_type)
        if result is None:
            return DiagnosticResult(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                passed=False,
                details=f"unknown diagnostic type: {diagnostic_type}",
            )
        return result


class HangingNodeAgent(NodeAgentProtocol):
    """Agent whose run_diagnostic never returns (simulates unreachable node / RPC hang)."""

    def __init__(self, node_id: str = "hanging") -> None:
        self._node_id = node_id

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def make_fake_agents(
    node_results: dict[str, dict[str, bool]],
) -> dict[str, FakeNodeAgent]:
    """Build FakeNodeAgents from {node_id: {diag_type: passed}} mapping."""
    agents: dict[str, FakeNodeAgent] = {}
    for node_id, results in node_results.items():
        diagnostic_results = {
            diag_type: DiagnosticResult(
                diagnostic_type=diag_type,
                node_id=node_id,
                passed=passed,
                details="pass" if passed else "fail",
            )
            for diag_type, passed in results.items()
        }
        agents[node_id] = FakeNodeAgent(
            diagnostic_results=diagnostic_results,
            node_id=node_id,
        )
    return agents


# ---------------------------------------------------------------------------
# Inter-machine diagnostic mock helper
# ---------------------------------------------------------------------------


def mock_nccl_pairwise_run(
    node_pass_map: dict[str, bool],
) -> contextlib.AbstractContextManager[None]:
    """Patch NcclNodeExecutor.run to return results per node_id.

    ``node_pass_map`` maps node_id → True (pass) or False (fail).
    """

    async def _fake_run(
        self: NcclNodeExecutor,
        node_id: str,
        timeout_seconds: int = 180,
    ) -> DiagnosticResult:
        passed = node_pass_map.get(node_id, True)
        return DiagnosticResult(
            diagnostic_type="nccl_pairwise",
            node_id=node_id,
            passed=passed,
            details="pass" if passed else "fail",
        )

    return patch.object(NcclNodeExecutor, "run", _fake_run)


# ---------------------------------------------------------------------------
# Subprocess mock helper
# ---------------------------------------------------------------------------


def make_mock_subprocess(
    stdout: str | bytes = b"",
    stderr: str | bytes = b"",
    returncode: int = 0,
) -> AsyncMock:
    """Unified async subprocess mock for diagnostic tests."""
    process = AsyncMock()
    stdout_bytes = stdout.encode() if isinstance(stdout, str) else stdout
    stderr_bytes = stderr.encode() if isinstance(stderr, str) else stderr
    process.communicate.return_value = (stdout_bytes, stderr_bytes)
    process.returncode = returncode
    process.kill = AsyncMock()
    process.wait = AsyncMock()
    return process
