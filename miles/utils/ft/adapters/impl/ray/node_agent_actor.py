from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import ray

from miles.utils.ft.adapters.types import (
    DIAGNOSTIC_TIMEOUT_SECONDS,
    REGISTER_TIMEOUT_SECONDS,
    ft_controller_actor_name,
)
from miles.utils.ft.utils.diagnostic_types import DiagnosticResult
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
        *,
        builder: Callable[..., Any],
        node_id: str,
        ft_id: str,
        **kwargs: object,
    ) -> None:
        self._ft_id = ft_id
        self._agent = builder(node_id=node_id, **kwargs)

    async def start(self) -> None:
        await self._agent.start()
        self._agent.wait_for_exporter_ready()
        self._register_with_controller()

    async def stop(self) -> None:
        await self._agent.stop()

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
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
        node_id = self._agent._node_id
        exporter_address = self._agent.get_exporter_address()
        node_metadata = self._agent.metadata

        controller = ray.get_actor(ft_controller_actor_name(self._ft_id))
        self_handle = ray.get_runtime_context().current_actor

        def _do_register() -> None:
            ray.get(
                controller.register_node_agent.remote(
                    node_id=node_id,
                    agent=self_handle,
                    exporter_address=exporter_address,
                    node_metadata=node_metadata,
                ),
                timeout=REGISTER_TIMEOUT_SECONDS,
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
                node_id,
                exporter_address,
            )
        else:
            raise RuntimeError(
                f"Failed to register node agent {node_id} after {_REGISTER_MAX_ATTEMPTS} attempts"
            ) from result.exception


FtNodeAgentActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtNodeAgentActorCls)
