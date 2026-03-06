from __future__ import annotations

import logging
import socket
import ray

from miles.utils.ft.agents.collectors.disk import DiskCollector
from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.utils.controller_handle import get_controller_handle
from miles.utils.ft.controller.diagnostics.gpu_diagnostic import GpuDiagnostic
from miles.utils.ft.controller.diagnostics.nccl.inter_machine import (
    InterMachineCommDiagnostic,
)
from miles.utils.ft.controller.diagnostics.nccl.intra_machine import (
    IntraMachineCommDiagnostic,
)
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import DiagnosticProtocol
from miles.utils.ft.utils.retry import retry_sync

logger = logging.getLogger(__name__)

_REGISTER_MAX_ATTEMPTS = 3
_REGISTER_RETRY_DELAY = 2.0


def _build_default_collectors() -> list[GpuCollector | KmsgCollector | NetworkCollector | DiskCollector]:
    return [GpuCollector(), KmsgCollector(), NetworkCollector(), DiskCollector()]


def _build_default_diagnostics(
    num_gpus: int,
) -> list[GpuDiagnostic | IntraMachineCommDiagnostic | InterMachineCommDiagnostic]:
    return [
        GpuDiagnostic(),
        IntraMachineCommDiagnostic(num_gpus=num_gpus),
        InterMachineCommDiagnostic(num_gpus=num_gpus),
    ]


class _FtNodeAgentActorCls:
    """Thin Ray actor wrapper around FtNodeAgent.

    Created per node with a scheduling strategy that pins it to the
    target node.  On ``start()`` it launches the metric collection loop
    and registers itself with the FtController so the diagnostic
    pipeline can reach it.
    """

    def __init__(
        self,
        node_id: str = "",
        ft_id: str = "",
        num_gpus: int = 8,
        collect_interval_seconds: float = 10.0,
        collectors_override: list[BaseCollector] | None = None,
        diagnostics_override: list[DiagnosticProtocol] | None = None,
    ) -> None:
        self._node_id = node_id or socket.gethostname()
        self._ft_id = ft_id

        collectors = collectors_override if collectors_override is not None else _build_default_collectors()
        diagnostics = diagnostics_override if diagnostics_override is not None else _build_default_diagnostics(num_gpus=num_gpus)
        self._agent = FtNodeAgent(
            node_id=self._node_id,
            collectors=collectors,
            collect_interval_seconds=collect_interval_seconds,
            diagnostics=diagnostics,
        )

    async def start(self) -> None:
        await self._agent.start()
        self._register_with_controller()

    async def stop(self) -> None:
        await self._agent.stop()

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        return await self._agent.run_diagnostic(
            diagnostic_type=diagnostic_type,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

    def get_exporter_address(self) -> str:
        return self._agent.get_exporter_address()

    def _register_with_controller(self) -> None:
        controller = get_controller_handle(self._ft_id)
        if controller is None:
            logger.warning(
                "Cannot register node agent: controller not available node_id=%s",
                self._node_id,
            )
            return

        self_handle = ray.get_runtime_context().current_actor
        exporter_address = self._agent.get_exporter_address()

        def _do_register() -> None:
            ray.get(
                controller.register_node_agent.remote(
                    node_id=self._node_id,
                    agent=self_handle,
                    exporter_address=exporter_address,
                ),
                timeout=10,
            )

        result = retry_sync(
            func=_do_register,
            description=f"register_node_agent({self._node_id})",
            max_retries=_REGISTER_MAX_ATTEMPTS,
            backoff_base=_REGISTER_RETRY_DELAY,
            max_backoff=_REGISTER_RETRY_DELAY,
        )
        if result.ok:
            logger.info(
                "Node agent registered node_id=%s exporter=%s",
                self._node_id, exporter_address,
            )

FtNodeAgentActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtNodeAgentActorCls)
