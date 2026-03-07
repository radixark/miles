from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.models.diagnostics import DiagnosticResult, UnknownDiagnosticError
from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS, DiagnosticProtocol, NodeAgentProtocol

logger = logging.getLogger(__name__)


class DiagnosticRunner(NodeAgentProtocol):
    """Registry-based diagnostic dispatcher with timeout protection.

    Holds a set of DiagnosticProtocol implementations keyed by
    diagnostic_type and dispatches run_diagnostic calls to the
    matching one.  Provides asyncio.wait_for timeout and exception
    catch-all so callers always get a DiagnosticResult back.
    """

    def __init__(
        self,
        node_id: str,
        diagnostics: list[DiagnosticProtocol] | None = None,
    ) -> None:
        self._node_id = node_id
        self._diagnostics: dict[str, DiagnosticProtocol] = {
            d.diagnostic_type: d for d in (diagnostics or [])
        }

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult:
        diagnostic = self._diagnostics.get(diagnostic_type)
        if diagnostic is None:
            raise UnknownDiagnosticError(
                f"node {self._node_id}: unknown diagnostic type '{diagnostic_type}', "
                f"registered types: {sorted(self._diagnostics.keys())}"
            )

        try:
            return await asyncio.wait_for(
                diagnostic.run(
                    node_id=self._node_id,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                ),
                timeout=timeout_seconds + 5,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_timeout type=%s node_id=%s timeout=%d",
                diagnostic_type,
                self._node_id,
                timeout_seconds,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details=f"diagnostic timed out after {timeout_seconds}s",
            )
        except Exception:
            logger.warning(
                "diagnostic_error type=%s node_id=%s",
                diagnostic_type,
                self._node_id,
                exc_info=True,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details="diagnostic raised exception",
            )
