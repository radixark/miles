from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable

from miles.utils.ft.agents.diagnostics.stack_trace import StackTraceDiagnostic
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import NodeAgentProtocol

logger = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"^(\S+)\s+\((.+?)(?::\d+)?\)$")


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


async def collect_stack_trace_suspects(
    agents: dict[str, NodeAgentProtocol],
    rank_pids_provider: Callable[[str], dict[int, int]],
    default_timeout_seconds: int,
) -> list[str]:
    """Collect stack traces from all nodes and identify suspects via aggregation.

    Returns sorted list of suspect node IDs.
    """
    traces: dict[str, str] = {}
    suspect_from_failures: list[str] = []

    async def _collect_node(node_id: str) -> None:
        try:
            rank_pids = rank_pids_provider(node_id)
        except Exception:
            suspect_from_failures.append(node_id)
            logger.warning(
                "rank_pids_provider_failed node=%s",
                node_id,
                exc_info=True,
            )
            return

        if not rank_pids:
            return

        pids = list(rank_pids.values())
        diag = StackTraceDiagnostic(pids=pids)

        try:
            result = await diag.run(
                node_id=node_id,
                timeout_seconds=default_timeout_seconds,
            )
            if result.passed:
                traces[node_id] = result.details
            else:
                suspect_from_failures.append(node_id)
                logger.info(
                    "stack_trace_collection_failed node=%s details=%s",
                    node_id, result.details,
                )
        except Exception:
            suspect_from_failures.append(node_id)
            logger.warning(
                "stack_trace_collect_exception node=%s",
                node_id,
                exc_info=True,
            )

    await asyncio.gather(*(_collect_node(nid) for nid in agents))

    suspect_from_aggregation = StackTraceAggregator().aggregate(traces=traces)
    all_suspects = sorted(set(suspect_from_failures) | set(suspect_from_aggregation))

    logger.info(
        "collect_stack_trace_suspects_done traces_collected=%d suspect_from_failures=%s suspect_from_aggregation=%s",
        len(traces), suspect_from_failures, suspect_from_aggregation,
    )
    return all_suspects
