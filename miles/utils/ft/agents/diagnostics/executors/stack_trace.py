from __future__ import annotations

import asyncio
import json
import logging

from pydantic import ConfigDict

from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.utils.base_model import FtBaseModel
from miles.utils.ft.utils.subprocess import run_subprocess_with_timeout

logger = logging.getLogger(__name__)


class PySpyFrame(FtBaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    filename: str
    line: int


class PySpyThread(FtBaseModel):
    model_config = ConfigDict(extra="ignore")

    thread_name: str | None = None
    active: bool
    owns_gil: bool
    frames: list[PySpyFrame]


class StackTraceNodeExecutor(BaseNodeExecutor):
    """Collect stack traces from training processes via py-spy dump --json."""

    diagnostic_type = "stack_trace"

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = 30,
        *,
        pids: list[int] | None = None,
    ) -> DiagnosticResult:
        if not pids:
            logger.warning("diagnostics: stack trace called with no PIDs: node_id=%s", node_id)
            return self._fail(node_id, "no PIDs provided")

        logger.info(
            "diagnostics: collecting stack traces: node_id=%s, num_pids=%d, timeout=%d",
            node_id,
            len(pids),
            timeout_seconds,
        )

        async def _collect_one(pid: int) -> tuple[list[PySpyThread], bool]:
            try:
                threads = await self._dump_pid(pid, timeout_seconds=timeout_seconds)
                return threads, True
            except Exception:
                logger.warning(
                    "diagnostics: stack trace dump failed: pid=%d, node=%s",
                    pid,
                    node_id,
                    exc_info=True,
                )
                return [], False

        results = await asyncio.gather(*(_collect_one(pid) for pid in pids))

        all_threads: list[PySpyThread] = []
        failures: int = 0
        for threads, success in results:
            all_threads.extend(threads)
            if not success:
                failures += 1

        logger.info(
            "diagnostics: stack trace collection complete: node_id=%s, total_threads=%d, failures=%d, total_pids=%d",
            node_id,
            len(all_threads),
            failures,
            len(pids),
        )
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=(failures == 0),
            details=json.dumps([t.model_dump() for t in all_threads]),
            metadata={"failed_pids": failures, "total_pids": len(pids)} if failures > 0 else None,
        )

    async def _dump_pid(self, pid: int, timeout_seconds: int) -> list[PySpyThread]:
        logger.debug("diagnostics: running py-spy dump: pid=%d, timeout=%d", pid, timeout_seconds)
        stdout, stderr, returncode = await run_subprocess_with_timeout(
            cmd=["py-spy", "dump", "--json", "--pid", str(pid)],
            timeout_seconds=timeout_seconds,
        )
        if returncode != 0:
            logger.warning("diagnostics: py-spy dump failed: pid=%d, returncode=%d", pid, returncode)
            raise RuntimeError(f"py-spy failed: {stderr.decode()}")

        raw = json.loads(stdout.decode())
        logger.debug("diagnostics: py-spy dump complete: pid=%d, num_threads=%d", pid, len(raw))
        return [PySpyThread.model_validate(t) for t in raw]
