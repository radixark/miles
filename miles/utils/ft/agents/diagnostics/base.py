from __future__ import annotations

from abc import abstractmethod
from typing import Any

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS, NodeExecutorProtocol
from miles.utils.ft.agents.types import DiagnosticResult


class BaseNodeExecutor(NodeExecutorProtocol):
    """Base class for all on-demand diagnostics.

    Subclasses implement run() to execute a diagnostic on a single node.
    NodeAgent dispatches via the diagnostic_type attribute.
    Heavy work should use asyncio.create_subprocess_exec to avoid
    blocking the NodeAgent event loop (see 3-discussions.md #48).
    """

    diagnostic_type: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            if "diagnostic_type" not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define a 'diagnostic_type' class attribute")

    def _fail(
        self,
        node_id: str,
        details: str,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticResult:
        return DiagnosticResult.fail_result(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            details=details,
            metadata=metadata,
        )

    def _pass(
        self,
        node_id: str,
        details: str,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticResult:
        return DiagnosticResult.pass_result(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            details=details,
            metadata=metadata,
        )

    @abstractmethod
    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult: ...
