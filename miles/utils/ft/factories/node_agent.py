"""Composition root for building FtNodeAgent instances with default components.

Extracted from node_agent_actor.py so that the actor wrapper stays thin
and the agent assembly logic is independently testable.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from miles.utils.ft.adapters.types import AgentMetadataProvider, NodeExecutorProtocol
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.disk import DiskCollector
from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl import NcclNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.stack_trace import StackTraceNodeExecutor
from miles.utils.ft.agents.types import CounterSample, GaugeSample, SampleEvaluator
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.core.disk_space import DiskSpaceLowDetector
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.nic_majority_down import NicMajorityDownDetector
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import ActionType, MetricStore

logger = logging.getLogger(__name__)

FALLBACK_NUM_GPUS: int = 8
DEFAULT_COLLECT_INTERVAL_SECONDS: float = 10.0


def detect_gpu_count() -> int:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        if count > 0:
            logger.info("detected_gpu_count=%d via pynvml", count)
            return count
    except Exception:
        logger.debug("pynvml GPU detection failed, using fallback", exc_info=True)
    logger.info("using fallback_gpu_count=%d", FALLBACK_NUM_GPUS)
    return FALLBACK_NUM_GPUS


def _detector_to_evaluator(detector: BaseFaultDetector) -> SampleEvaluator:
    """Adapt a controller-layer BaseFaultDetector into an agent-layer SampleEvaluator."""

    def _evaluate(
        node_id: str,
        samples: Sequence[GaugeSample | CounterSample],
    ) -> tuple[bool, str]:
        store = InMemoryMetricStore()
        store.ingest_samples(target_id=node_id, samples=list(samples))

        ctx = DetectorContext(
            metric_store=MetricStore(
                time_series_store=store,
                mini_wandb=MiniWandb(),
            )
        )
        decision = detector.evaluate(ctx)
        passed = decision.action == ActionType.NONE
        return passed, decision.reason

    return _evaluate


def build_default_collectors() -> list[BaseCollector]:
    return [GpuCollector(), KmsgCollector(), NetworkCollector(), DiskCollector()]


def build_all_diagnostics(
    num_gpus: int | None = None,
    disk_mounts: list[Path] | None = None,
    xid_since: datetime | None = None,
) -> list[NodeExecutorProtocol]:
    """Build all available node-level diagnostic executors.

    Registers every diagnostic type the system supports.  Callers choose
    which subset to actually run (e.g. local CLI excludes nccl_pairwise).
    """
    resolved_num_gpus = num_gpus if num_gpus is not None else detect_gpu_count()
    return [
        StackTraceNodeExecutor(),
        GpuNodeExecutor(),
        NcclNodeExecutor(diagnostic_type="nccl_simple", expected_bandwidth_gbps=350.0, num_gpus=resolved_num_gpus),
        NcclNodeExecutor(
            diagnostic_type="nccl_pairwise",
            expected_bandwidth_gbps=40.0,
            num_gpus=resolved_num_gpus,
            nccl_test_binary="all_gather_perf",
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="disk",
            collector=DiskCollector(disk_mounts=disk_mounts),
            evaluator=_detector_to_evaluator(DiskSpaceLowDetector()),
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="network",
            collector=NetworkCollector(),
            evaluator=_detector_to_evaluator(NicMajorityDownDetector()),
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="xid",
            collector=KmsgCollector(since=xid_since),
            evaluator=_detector_to_evaluator(GpuFaultDetector()),
        ),
    ]


def build_node_agent(
    node_id: str = "",
    num_gpus: int | None = None,
    collect_interval_seconds: float = DEFAULT_COLLECT_INTERVAL_SECONDS,
    collectors_override: list[BaseCollector] | None = None,
    diagnostics_override: list[NodeExecutorProtocol] | None = None,
    metadata_provider: AgentMetadataProvider | None = None,
    **kwargs: Any,
) -> FtNodeAgent:
    resolved_node_id = node_id or socket.gethostname()
    collectors = collectors_override if collectors_override is not None else build_default_collectors()
    diagnostics = (
        diagnostics_override if diagnostics_override is not None else build_all_diagnostics(num_gpus=num_gpus)
    )
    return FtNodeAgent(
        node_id=resolved_node_id,
        collectors=collectors,
        collect_interval_seconds=collect_interval_seconds,
        diagnostics=diagnostics,
        metadata_provider=metadata_provider,
    )
