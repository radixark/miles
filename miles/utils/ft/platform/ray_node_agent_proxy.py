from __future__ import annotations

import logging
from typing import Any

from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import NodeAgentProtocol

logger = logging.getLogger(__name__)


class RayNodeAgentProxy(NodeAgentProtocol):
    """Adapts a Ray actor handle to NodeAgentProtocol.

    DiagnosticScheduler calls ``agent.run_diagnostic(...)`` as a normal
    async method.  This proxy translates the call into a Ray remote
    invocation (``handle.run_diagnostic.remote(...)``), which returns an
    ObjectRef that can be awaited in async code.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        return await self._handle.run_diagnostic.remote(
            diagnostic_type=diagnostic_type,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
