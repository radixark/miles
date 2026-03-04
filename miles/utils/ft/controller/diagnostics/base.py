from __future__ import annotations

from abc import ABC, abstractmethod

from miles.utils.ft.models import DiagnosticResult


class BaseDiagnostic(ABC):
    """Base class for all on-demand diagnostics.

    Subclasses implement run() to execute a diagnostic on a single node.
    NodeAgent dispatches via the diagnostic_type attribute.
    Heavy work should use asyncio.create_subprocess_exec to avoid
    blocking the NodeAgent event loop (see 3-discussions.md #48).
    """

    diagnostic_type: str

    @abstractmethod
    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult: ...
