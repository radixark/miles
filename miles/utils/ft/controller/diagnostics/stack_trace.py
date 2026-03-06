from __future__ import annotations

import asyncio
import logging
import re

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"^(\S+)\s+\((.+?)(?::\d+)?\)$")


class StackTraceDiagnostic(BaseDiagnostic):
    """Collect stack traces from training processes via py-spy dump."""

    diagnostic_type = "stack_trace"

    def __init__(self, pids: list[int] | None = None) -> None:
        self._pids = pids or []

    async def run(
        self, node_id: str, timeout_seconds: int = 30,
    ) -> DiagnosticResult:
        if not self._pids:
            return DiagnosticResult.fail_result(
                diagnostic_type=self.diagnostic_type, node_id=node_id,
                details="no PIDs provided",
            )

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
        proc = await asyncio.create_subprocess_exec(
            "py-spy", "dump", "--pid", str(pid),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"py-spy failed: {stderr.decode()}")
        return stdout.decode()


class StackTraceAggregator:
    """Aggregate stack traces from multiple nodes and identify suspect nodes.

    Uses majority/minority fingerprint grouping: nodes whose stack trace
    fingerprint differs from the majority group are considered suspects.
    """

    def aggregate(self, traces: dict[str, str]) -> list[str]:
        if len(traces) <= 1:
            return []

        fingerprints: dict[str, str] = {}
        for node_id, trace_text in traces.items():
            fingerprints[node_id] = self._extract_fingerprint(trace_text)

        groups: dict[str, list[str]] = {}
        for node_id, fingerprint in fingerprints.items():
            groups.setdefault(fingerprint, []).append(node_id)

        if len(groups) <= 1:
            return []

        majority_size = max(len(nodes) for nodes in groups.values())
        suspects: list[str] = []
        for fingerprint, nodes in groups.items():
            if len(nodes) < majority_size:
                suspects.extend(nodes)

        return sorted(suspects)

    def _extract_fingerprint(self, trace_text: str) -> str:
        """Extract a normalized fingerprint from py-spy dump output.

        For each thread block, extract the top (innermost) frame's
        function name and file path (without line numbers or addresses).
        Sort the resulting frame signatures to produce a stable fingerprint.
        """
        top_frames: list[str] = []
        lines = trace_text.strip().splitlines()

        current_top_frame: str | None = None
        for line in lines:
            stripped = line.strip()

            if self._is_thread_header(stripped):
                if current_top_frame is not None:
                    top_frames.append(current_top_frame)
                current_top_frame = None
                continue

            if current_top_frame is None:
                frame_sig = self._parse_frame_line(stripped)
                if frame_sig is not None:
                    current_top_frame = frame_sig

        if current_top_frame is not None:
            top_frames.append(current_top_frame)

        top_frames.sort()
        return "|".join(top_frames)

    @staticmethod
    def _is_thread_header(line: str) -> bool:
        return line.startswith("Thread") or line.startswith("=== PID")

    @staticmethod
    def _parse_frame_line(line: str) -> str | None:
        """Parse a py-spy frame line and return 'function_name (file_path)'.

        py-spy dump frames look like:
            function_name (file_path:line_number)
        or with a leading dash for active frames:
            - function_name (file_path:line_number)
        """
        cleaned = line.lstrip("- ").strip()
        if not cleaned:
            return None

        match = _FRAME_RE.match(cleaned)
        if match:
            func_name = match.group(1)
            file_path = match.group(2)
            return f"{func_name} ({file_path})"

        return None
