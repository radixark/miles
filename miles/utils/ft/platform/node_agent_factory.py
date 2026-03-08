"""Composition root for building FtNodeAgent instances with default components.

Extracted from node_agent_actor.py so that the actor wrapper stays thin
and the agent assembly logic is independently testable.
"""

from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.disk import DiskCollector
from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl_pairwise import NcclPairwiseNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl_simple import NcclSimpleNodeExecutor
from miles.utils.ft.controller.detectors.checks.gpu.checks import check_gpu_faults
from miles.utils.ft.controller.detectors.checks.hardware import _check_disk_fault, _check_majority_nic_down
from miles.utils.ft.protocols.agents import NodeExecutorProtocol

DEFAULT_NUM_GPUS: int = 8
DEFAULT_COLLECT_INTERVAL_SECONDS: float = 10.0


def build_default_collectors() -> list[GpuCollector | KmsgCollector | NetworkCollector | DiskCollector]:
    return [GpuCollector(), KmsgCollector(), NetworkCollector(), DiskCollector()]


def build_all_diagnostics(
    num_gpus: int = DEFAULT_NUM_GPUS,
    disk_mounts: list[Path] | None = None,
    xid_since: datetime | None = None,
) -> list[NodeExecutorProtocol]:
    """Build all available node-level diagnostic executors.

    Registers every diagnostic type the system supports.  Callers choose
    which subset to actually run (e.g. local CLI excludes nccl_pairwise).
    """
    return [
        GpuNodeExecutor(),
        NcclSimpleNodeExecutor(num_gpus=num_gpus),
        NcclPairwiseNodeExecutor(num_gpus=num_gpus),
        CollectorBasedNodeExecutor(
            diagnostic_type="disk",
            collector=DiskCollector(disk_mounts=disk_mounts),
            check_fn=_check_disk_fault,
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="network",
            collector=NetworkCollector(),
            check_fn=_check_majority_nic_down,
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="xid",
            collector=KmsgCollector(since=xid_since),
            check_fn=check_gpu_faults,
        ),
    ]


def build_node_agent(
    node_id: str = "",
    num_gpus: int = DEFAULT_NUM_GPUS,
    collect_interval_seconds: float = DEFAULT_COLLECT_INTERVAL_SECONDS,
    collectors_override: list[BaseCollector] | None = None,
    diagnostics_override: list[NodeExecutorProtocol] | None = None,
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
    )
