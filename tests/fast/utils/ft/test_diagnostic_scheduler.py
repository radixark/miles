"""Tests for DiagnosticScheduler and BaseDiagnostic."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models import DiagnosticResult


class _ConcreteDiagnostic(BaseDiagnostic):
    diagnostic_type = "test_concrete"

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=True,
            details="ok",
        )


class TestBaseDiagnostic:
    @pytest.mark.asyncio
    async def test_concrete_subclass_can_run(self) -> None:
        diag = _ConcreteDiagnostic()
        result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert result.node_id == "node-0"
        assert result.diagnostic_type == "test_concrete"

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDiagnostic()  # type: ignore[abstract]
