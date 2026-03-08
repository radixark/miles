"""Tests for controller diagnostic utility functions (call, gather, partition)."""

from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.controller.diagnostics.utils import (
    call_agent_diagnostic,
    gather_diagnostic_results,
    partition_results,
)
from miles.utils.ft.models.diagnostic import DiagnosticResult, UnknownDiagnosticError
from tests.fast.utils.ft.conftest import FakeNodeAgent, HangingNodeAgent


_DIAG_TYPE = "test_diag"


# ---------------------------------------------------------------------------
# call_agent_diagnostic
# ---------------------------------------------------------------------------


class TestCallAgentDiagnostic:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        expected = DiagnosticResult(
            diagnostic_type=_DIAG_TYPE, node_id="n0", passed=True, details="ok"
        )
        agent = FakeNodeAgent(diagnostic_results={_DIAG_TYPE: expected}, node_id="n0")
        result = await call_agent_diagnostic(
            agent=agent, node_id="n0", diagnostic_type=_DIAG_TYPE, timeout_seconds=10
        )
        assert result.passed
        assert result.node_id == "n0"

    @pytest.mark.asyncio
    async def test_timeout_returns_fail(self) -> None:
        agent = HangingNodeAgent(node_id="n0")
        result = await call_agent_diagnostic(
            agent=agent, node_id="n0", diagnostic_type=_DIAG_TYPE, timeout_seconds=0
        )
        assert not result.passed
        assert "timed out" in result.details

    @pytest.mark.asyncio
    async def test_unknown_diagnostic_error_returns_fail(self) -> None:
        class RaisingAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120, **kwargs: object
            ) -> DiagnosticResult:
                raise UnknownDiagnosticError(diagnostic_type)

        result = await call_agent_diagnostic(
            agent=RaisingAgent(),  # type: ignore[arg-type]
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            timeout_seconds=10,
        )
        assert not result.passed
        assert "config error" in result.details

    @pytest.mark.asyncio
    async def test_generic_exception_returns_fail(self) -> None:
        class CrashingAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120, **kwargs: object
            ) -> DiagnosticResult:
                raise RuntimeError("boom")

        result = await call_agent_diagnostic(
            agent=CrashingAgent(),  # type: ignore[arg-type]
            node_id="n0",
            diagnostic_type=_DIAG_TYPE,
            timeout_seconds=10,
        )
        assert not result.passed
        assert "exception" in result.details


# ---------------------------------------------------------------------------
# gather_diagnostic_results
# ---------------------------------------------------------------------------


class TestGatherDiagnosticResults:
    @pytest.mark.asyncio
    async def test_gathers_from_multiple_agents(self) -> None:
        agents = {
            "n0": FakeNodeAgent(
                diagnostic_results={
                    _DIAG_TYPE: DiagnosticResult(
                        diagnostic_type=_DIAG_TYPE, node_id="n0", passed=True, details="ok"
                    )
                },
                node_id="n0",
            ),
            "n1": FakeNodeAgent(
                diagnostic_results={
                    _DIAG_TYPE: DiagnosticResult(
                        diagnostic_type=_DIAG_TYPE, node_id="n1", passed=False, details="bad"
                    )
                },
                node_id="n1",
            ),
        }
        results = await gather_diagnostic_results(
            diagnostic_type=_DIAG_TYPE, agents=agents, timeout_seconds=10
        )
        assert set(results.keys()) == {"n0", "n1"}
        assert results["n0"].passed
        assert not results["n1"].passed

    @pytest.mark.asyncio
    async def test_empty_agents_returns_empty(self) -> None:
        results = await gather_diagnostic_results(
            diagnostic_type=_DIAG_TYPE, agents={}, timeout_seconds=10
        )
        assert results == {}


# ---------------------------------------------------------------------------
# partition_results
# ---------------------------------------------------------------------------


class TestPartitionResults:
    def test_separates_failed_nodes(self) -> None:
        results = {
            "n0": DiagnosticResult(
                diagnostic_type=_DIAG_TYPE, node_id="n0", passed=True, details="ok"
            ),
            "n1": DiagnosticResult(
                diagnostic_type=_DIAG_TYPE, node_id="n1", passed=False, details="fail"
            ),
            "n2": DiagnosticResult(
                diagnostic_type=_DIAG_TYPE, node_id="n2", passed=False, details="fail"
            ),
        }
        bad = partition_results(results=results, diagnostic_type=_DIAG_TYPE)
        assert sorted(bad) == ["n1", "n2"]

    def test_all_pass_returns_empty(self) -> None:
        results = {
            "n0": DiagnosticResult(
                diagnostic_type=_DIAG_TYPE, node_id="n0", passed=True, details="ok"
            ),
        }
        assert partition_results(results=results, diagnostic_type=_DIAG_TYPE) == []

    def test_empty_results_returns_empty(self) -> None:
        assert partition_results(results={}, diagnostic_type=_DIAG_TYPE) == []
