"""Composition root for building FtNodeAgent instances with default components.

Extracted from node_agent_actor.py so that the actor wrapper stays thin
and the agent assembly logic is independently testable.
"""
from __future__ import annotations

import socket

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.disk import DiskCollector
from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.gpu_diagnostic import GpuDiagnostic
from miles.utils.ft.agents.diagnostics.nccl.inter_machine import InterMachineCommDiagnostic
from miles.utils.ft.agents.diagnostics.nccl.intra_machine import IntraMachineCommDiagnostic
from miles.utils.ft.protocols.agents import DiagnosticProtocol


def build_default_collectors() -> list[GpuCollector | KmsgCollector | NetworkCollector | DiskCollector]:
    return [GpuCollector(), KmsgCollector(), NetworkCollector(), DiskCollector()]


def build_default_diagnostics(
    num_gpus: int,
) -> list[GpuDiagnostic | IntraMachineCommDiagnostic | InterMachineCommDiagnostic]:
    return [
        GpuDiagnostic(),
        IntraMachineCommDiagnostic(num_gpus=num_gpus),
        InterMachineCommDiagnostic(num_gpus=num_gpus),
    ]


def build_node_agent(
    node_id: str = "",
    num_gpus: int = 8,
    collect_interval_seconds: float = 10.0,
    collectors_override: list[BaseCollector] | None = None,
    diagnostics_override: list[DiagnosticProtocol] | None = None,
) -> FtNodeAgent:
    resolved_node_id = node_id or socket.gethostname()
    collectors = collectors_override if collectors_override is not None else build_default_collectors()
    diagnostics = diagnostics_override if diagnostics_override is not None else build_default_diagnostics(num_gpus=num_gpus)
    return FtNodeAgent(
        node_id=resolved_node_id,
        collectors=collectors,
        collect_interval_seconds=collect_interval_seconds,
        diagnostics=diagnostics,
    )
