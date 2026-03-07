from __future__ import annotations

from abc import ABC, abstractmethod

from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import DiagnosticProtocol


class BaseDiagnostic(DiagnosticProtocol, ABC):
    """Base class for all on-demand diagnostics.

    Subclasses implement run() to execute a diagnostic on a single node.
    NodeAgent dispatches via the diagnostic_type attribute.
    Heavy work should use asyncio.create_subprocess_exec to avoid
    blocking the NodeAgent event loop (see 3-discussions.md #48).
    """

    diagnostic_type: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not hasattr(cls, "diagnostic_type"):
            raise TypeError(
                f"{cls.__name__} must define a 'diagnostic_type' class attribute"
            )

    def _fail(self, node_id: str, details: str) -> DiagnosticResult:
        return DiagnosticResult.fail_result(
            diagnostic_type=self.diagnostic_type, node_id=node_id, details=details,
        )

    def _pass(self, node_id: str, details: str) -> DiagnosticResult:
        return DiagnosticResult.pass_result(
            diagnostic_type=self.diagnostic_type, node_id=node_id, details=details,
        )

    @abstractmethod
    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult: ...
