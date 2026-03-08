"""Tests for miles.utils.ft.agents.diagnostics.base."""

from __future__ import annotations

import pytest

from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.types import DiagnosticResult


class _ValidExecutor(BaseNodeExecutor):
    diagnostic_type = "test_diag"

    async def run(self, node_id: str, timeout_seconds: int = 120) -> DiagnosticResult:
        return self._pass(node_id=node_id, details="ok")


class TestInitSubclass:
    def test_concrete_subclass_without_diagnostic_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must define a 'diagnostic_type'"):

            class _MissingType(BaseNodeExecutor):
                async def run(self, node_id: str, timeout_seconds: int = 120) -> DiagnosticResult:
                    return self._pass(node_id=node_id, details="ok")

    def test_concrete_subclass_with_diagnostic_type_succeeds(self) -> None:
        executor = _ValidExecutor()
        assert executor.diagnostic_type == "test_diag"

    def test_abstract_subclass_without_diagnostic_type_allowed(self) -> None:
        """Intermediate abstract classes don't need diagnostic_type."""
        from abc import abstractmethod

        class _IntermediateBase(BaseNodeExecutor):
            @abstractmethod
            async def run(self, node_id: str, timeout_seconds: int = 120) -> DiagnosticResult: ...


class TestHelpers:
    def test_fail_helper_returns_failed_result(self) -> None:
        executor = _ValidExecutor()

        result = executor._fail(node_id="node-0", details="something broke")

        assert result.passed is False
        assert result.diagnostic_type == "test_diag"
        assert result.node_id == "node-0"
        assert result.details == "something broke"

    def test_pass_helper_returns_passed_result(self) -> None:
        executor = _ValidExecutor()

        result = executor._pass(node_id="node-0", details="all good")

        assert result.passed is True
        assert result.diagnostic_type == "test_diag"
        assert result.node_id == "node-0"

    def test_fail_with_metadata(self) -> None:
        executor = _ValidExecutor()

        result = executor._fail(node_id="node-0", details="err", metadata={"key": "val"})

        assert result.metadata == {"key": "val"}
