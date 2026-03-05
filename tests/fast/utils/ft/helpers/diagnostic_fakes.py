from __future__ import annotations

import asyncio
import contextlib
from typing import Generator
from unittest.mock import AsyncMock, patch

from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.diagnostics.inter_machine_comm import (
    InterMachineCommDiagnostic,
)
from miles.utils.ft.models import (
    ActionType,
    Decision,
    DiagnosticResult,
)


# ---------------------------------------------------------------------------
# Diagnostic test helpers
# ---------------------------------------------------------------------------


class StubDiagnostic(BaseDiagnostic):
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
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=self._passed,
            details=self._details,
        )


class SlowDiagnostic(BaseDiagnostic):
    """Test diagnostic that sleeps longer than its timeout."""

    diagnostic_type = "slow"

    def __init__(self, sleep_seconds: float = 300.0) -> None:
        self._sleep_seconds = sleep_seconds

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        await asyncio.sleep(self._sleep_seconds)
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=True,
            details="should not reach here",
        )


# ---------------------------------------------------------------------------
# Diagnostic scheduler fakes
# ---------------------------------------------------------------------------


class FakeDiagnosticScheduler:
    """Programmable stub for DiagnosticScheduler in recovery tests."""

    def __init__(self, decision: Decision | None = None) -> None:
        self._decision = decision or Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="fake diagnostic — all passed",
        )
        self.call_count: int = 0
        self.last_trigger_reason: str | None = None
        self.last_suspect_node_ids: list[str] | None = None

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: str,
        suspect_node_ids: list[str] | None = None,
    ) -> Decision:
        self.call_count += 1
        self.last_trigger_reason = trigger_reason
        self.last_suspect_node_ids = suspect_node_ids
        return self._decision


# ---------------------------------------------------------------------------
# Agent test helpers (node agents for diagnostic scheduling)
# ---------------------------------------------------------------------------


class FakeNodeAgent:
    def __init__(
        self,
        diagnostic_results: dict[str, DiagnosticResult] | None = None,
        node_id: str = "fake",
    ) -> None:
        self._diagnostic_results = diagnostic_results or {}
        self._node_id = node_id

    async def run_diagnostic(
        self, diagnostic_type: str, timeout_seconds: int = 120,
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


def mock_inter_machine_run(
    node_pass_map: dict[str, bool],
) -> contextlib.AbstractContextManager[None]:
    """Patch InterMachineCommDiagnostic.run to return results per node_id.

    ``node_pass_map`` maps node_id → True (pass) or False (fail).
    """
    async def _fake_run(
        self: InterMachineCommDiagnostic,
        node_id: str,
        timeout_seconds: int = 180,
    ) -> DiagnosticResult:
        passed = node_pass_map.get(node_id, True)
        return DiagnosticResult(
            diagnostic_type="inter_machine",
            node_id=node_id,
            passed=passed,
            details="pass" if passed else "fail",
        )

    return patch.object(InterMachineCommDiagnostic, "run", _fake_run)


# ---------------------------------------------------------------------------
# Stack trace diagnostic mock helper
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def mock_stack_trace_diagnostic(
    side_effects: list[DiagnosticResult | Exception],
) -> Generator[AsyncMock, None, None]:
    """Patch StackTraceDiagnostic and wire an AsyncMock with the given side_effects."""
    with patch(
        "miles.utils.ft.controller.diagnostics.scheduler.StackTraceDiagnostic"
    ) as mock_diag_cls:
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(side_effect=side_effects)
        mock_diag_cls.return_value = mock_instance
        yield mock_diag_cls
