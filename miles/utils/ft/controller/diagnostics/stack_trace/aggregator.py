from __future__ import annotations

import logging

from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyThread
from miles.utils.ft.utils.base_model import FtBaseModel

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FRAMES = 8


class StackTraceTieError(Exception):
    """Fingerprint groups are tied — cannot determine majority, require human intervention."""


class AggregationResult(FtBaseModel):
    suspect_node_ids: list[str]
    fingerprint_groups: dict[str, list[str]]
    raw_traces: dict[str, list[PySpyThread]]


class StackTraceAggregator:
    """Aggregate stack traces from multiple nodes and identify suspect nodes.

    Uses majority/minority fingerprint grouping: nodes whose stack trace
    fingerprint differs from the majority group are considered suspects.

    The fingerprint is built from the top-N innermost frames (function names
    only, no filenames or line numbers) of each thread, which captures the
    meaningful call path while truncating noisy outermost entry-point frames.
    """

    def __init__(self, max_frames: int = _DEFAULT_MAX_FRAMES) -> None:
        self._max_frames = max_frames

    def aggregate(self, traces: dict[str, list[PySpyThread]]) -> AggregationResult:
        if len(traces) <= 1:
            result = AggregationResult(
                suspect_node_ids=[],
                fingerprint_groups={},
                raw_traces=traces,
            )
            logger.info("stack_trace_aggregation result=%s", result.model_dump_json())
            return result

        fp_to_nodes: dict[str, list[str]] = {}
        empty_fingerprint_nodes: list[str] = []
        for node_id, threads in traces.items():
            fp = self._extract_fingerprint(threads)
            if not fp:
                empty_fingerprint_nodes.append(node_id)
                logger.warning("stack_trace_empty_fingerprint node=%s", node_id)
                continue
            fp_to_nodes.setdefault(fp, []).append(node_id)

        if len(fp_to_nodes) <= 1:
            suspects: list[str] = []
        else:
            majority_size = max(len(nodes) for nodes in fp_to_nodes.values())
            minority_exists = any(len(nodes) < majority_size for nodes in fp_to_nodes.values())
            if not minority_exists:
                raise StackTraceTieError(
                    f"stack trace tie: {len(fp_to_nodes)} fingerprint groups of equal size {majority_size}, "
                    "unable to determine majority — require human intervention"
                )
            suspects = sorted(nid for nodes in fp_to_nodes.values() if len(nodes) < majority_size for nid in nodes)

        result = AggregationResult(
            suspect_node_ids=sorted(set(suspects) | set(empty_fingerprint_nodes)),
            fingerprint_groups=fp_to_nodes,
            raw_traces=traces,
        )
        logger.info("stack_trace_aggregation result=%s", result.model_dump_json())
        return result

    def _extract_fingerprint(self, threads: list[PySpyThread]) -> str:
        """Build a stable fingerprint from the top-N innermost frames of each thread."""
        thread_fps: list[str] = []
        for thread in threads:
            if thread.frames:
                top_n = thread.frames[: self._max_frames]
                chain = " > ".join(f.name for f in top_n)
                thread_fps.append(chain)

        thread_fps.sort()
        return "|".join(thread_fps)
