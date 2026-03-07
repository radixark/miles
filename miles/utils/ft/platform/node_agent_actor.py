from __future__ import annotations

import logging

import ray

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.platform.node_agent_factory import build_node_agent
from miles.utils.ft.protocols.agents import DiagnosticProtocol
from miles.utils.ft.protocols.platform import ft_controller_actor_name
from miles.utils.ft.utils.graceful_degrade import graceful_degrade
from miles.utils.ft.utils.retry import retry_sync

logger = logging.getLogger(__name__)

_REGISTER_MAX_ATTEMPTS = 3
_REGISTER_RETRY_DELAY = 2.0


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
        self._ft_id = ft_id
        self._agent = build_node_agent(
            node_id=node_id,
            num_gpus=num_gpus,
            collect_interval_seconds=collect_interval_seconds,
            collectors_override=collectors_override,
            diagnostics_override=diagnostics_override,
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

    @graceful_degrade(msg="Failed to register node agent with controller")
    def _register_with_controller(self) -> None:
        controller = ray.get_actor(ft_controller_actor_name(self._ft_id))

        self_handle = ray.get_runtime_context().current_actor
        node_id = self._agent._node_id
        exporter_address = self._agent.get_exporter_address()

        def _do_register() -> None:
            ray.get(
                controller.register_node_agent.remote(
                    node_id=node_id,
                    agent=self_handle,
                    exporter_address=exporter_address,
                ),
                timeout=10,
            )

        result = retry_sync(
            func=_do_register,
            description=f"register_node_agent({node_id})",
            max_retries=_REGISTER_MAX_ATTEMPTS,
            backoff_base=_REGISTER_RETRY_DELAY,
            max_backoff=_REGISTER_RETRY_DELAY,
        )
        if result.ok:
            logger.info(
                "Node agent registered node_id=%s exporter=%s",
                node_id, exporter_address,
            )

FtNodeAgentActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtNodeAgentActorCls)
