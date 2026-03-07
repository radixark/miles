from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.agents.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.utils.subprocess import run_subprocess_with_timeout

logger = logging.getLogger(__name__)


class StackTraceDiagnostic(BaseDiagnostic):
    """Collect stack traces from training processes via py-spy dump."""

    diagnostic_type = "stack_trace"

    def __init__(self, pids: list[int] | None = None) -> None:
        self._pids = pids or []

    async def run(
        self, node_id: str, timeout_seconds: int = 30,
    ) -> DiagnosticResult:
        if not self._pids:
            return self._fail(node_id, "no PIDs provided")

        async def _collect_one(pid: int) -> tuple[str, bool]:
            try:
                trace = await self._dump_pid(pid, timeout_seconds=timeout_seconds)
                return f"=== PID {pid} ===\n{trace}", True
            except Exception:
                logger.warning(
                    "stack_trace_dump_failed pid=%d node=%s",
                    pid, node_id,
                    exc_info=True,
                )
                return f"=== PID {pid} ===\nFAILED: could not collect stack trace", False

        results = await asyncio.gather(*(_collect_one(pid) for pid in self._pids))

        traces: list[str] = []
        failures: int = 0
        for trace_text, success in results:
            traces.append(trace_text)
            if not success:
                failures += 1

        all_failed = failures == len(self._pids)
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=not all_failed,
            details="\n".join(traces),
        )

    async def _dump_pid(self, pid: int, timeout_seconds: int) -> str:
        stdout, stderr, returncode = await run_subprocess_with_timeout(
            cmd=["py-spy", "dump", "--pid", str(pid)],
            timeout_seconds=timeout_seconds,
        )
        if returncode != 0:
            raise RuntimeError(f"py-spy failed: {stderr.decode()}")
        return stdout.decode()
