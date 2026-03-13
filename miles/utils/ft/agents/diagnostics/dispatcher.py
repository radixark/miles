from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS, NodeAgentProtocol, NodeExecutorProtocol
from miles.utils.ft.agents.types import DiagnosticResult, UnknownDiagnosticError

logger = logging.getLogger(__name__)


class NodeDiagnosticDispatcher(NodeAgentProtocol):
    """Registry-based diagnostic dispatcher with timeout protection.

    Holds a set of NodeExecutorProtocol implementations keyed by
    diagnostic_type and dispatches run_diagnostic calls to the
    matching one.  Provides asyncio.wait_for timeout and exception
    catch-all so callers always get a DiagnosticResult back.
    """

    def __init__(
        self,
        node_id: str,
        diagnostics: list[NodeExecutorProtocol] | None = None,
    ) -> None:
        self._node_id = node_id
        self._diagnostics: dict[str, NodeExecutorProtocol] = {d.diagnostic_type: d for d in (diagnostics or [])}
        logger.info(
            "diagnostics: dispatcher initialized: node_id=%s, registered_types=%s",
            node_id, sorted(self._diagnostics.keys()),
        )

    @property
    def available_types(self) -> list[str]:
        return sorted(self._diagnostics.keys())

    async def run_selected(
        self,
        diagnostic_types: list[str],
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> list[DiagnosticResult]:
        return [await self.run_diagnostic(dt, timeout_seconds=timeout_seconds) for dt in diagnostic_types]

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult:
        diagnostic = self._diagnostics.get(diagnostic_type)
        if diagnostic is None:
            logger.error(
                "diagnostics: unknown diagnostic type: node_id=%s, type=%s, registered=%s",
                self._node_id, diagnostic_type, sorted(self._diagnostics.keys()),
            )
            raise UnknownDiagnosticError(
                f"node {self._node_id}: unknown diagnostic type '{diagnostic_type}', "
                f"registered types: {sorted(self._diagnostics.keys())}"
            )

        logger.info(
            "diagnostics: dispatching diagnostic: node_id=%s, type=%s, timeout=%d",
            self._node_id, diagnostic_type, timeout_seconds,
        )
        try:
            result = await asyncio.wait_for(
                diagnostic.run(
                    node_id=self._node_id,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                ),
                timeout=timeout_seconds + 5,
            )
            logger.info(
                "diagnostics: diagnostic complete: node_id=%s, type=%s, passed=%s",
                self._node_id, diagnostic_type, result.passed,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostics: diagnostic timed out: type=%s, node_id=%s, timeout=%d",
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
            logger.error(
                "diagnostics: diagnostic raised exception: type=%s, node_id=%s",
                diagnostic_type,
                self._node_id,
                exc_info=True,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details="diagnostic raised exception",
            )
