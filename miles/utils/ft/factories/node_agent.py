"""Composition root for building FtNodeAgent instances with default components.

Extracted from node_agent_actor.py so that the actor wrapper stays thin
and the agent assembly logic is independently testable.
"""

from __future__ import annotations

import socket
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
from miles.utils.ft.controller.detectors.core.disk_space import DiskSpaceLowDetector
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.nic_majority_down import NicMajorityDownDetector

DEFAULT_NUM_GPUS: int = 8
DEFAULT_COLLECT_INTERVAL_SECONDS: float = 10.0


def build_default_collectors() -> list[BaseCollector]:
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
        StackTraceNodeExecutor(),
        GpuNodeExecutor(),
        NcclNodeExecutor(diagnostic_type="nccl_simple", expected_bandwidth_gbps=350.0, num_gpus=num_gpus),
        NcclNodeExecutor(
            diagnostic_type="nccl_pairwise",
            expected_bandwidth_gbps=40.0,
            num_gpus=num_gpus,
            nccl_test_binary="all_gather_perf",
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="disk",
            collector=DiskCollector(disk_mounts=disk_mounts),
            detector=DiskSpaceLowDetector(),
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="network",
            collector=NetworkCollector(),
            detector=NicMajorityDownDetector(),
        ),
        CollectorBasedNodeExecutor(
            diagnostic_type="xid",
            collector=KmsgCollector(since=xid_since),
            detector=GpuFaultDetector(),
        ),
    ]


def build_node_agent(
    node_id: str = "",
    num_gpus: int = DEFAULT_NUM_GPUS,
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


def launch_node_agent_actor(
    node_id: str,
    ft_id: str = "",
    actor_name: str = "",
    metadata_provider: AgentMetadataProvider | None = None,
    **kwargs: Any,
) -> Any:
    """Create and return a named FtNodeAgentActor with builder injection."""
    from miles.utils.ft.adapters.impl.k8s_metadata_provider import K8sMetadataProvider
    from miles.utils.ft.adapters.impl.ray.node_agent_actor import FtNodeAgentActor

    resolved_provider = metadata_provider or K8sMetadataProvider()

    return FtNodeAgentActor.options(name=actor_name).remote(
        builder=build_node_agent,
        node_id=node_id,
        ft_id=ft_id,
        metadata_provider=resolved_provider,
        **kwargs,
    )
