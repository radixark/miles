from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ray

from miles.utils.ft.adapters.impl.ray.node_agent_proxy import RayNodeAgentProxy


class _FtControllerActorCls:
    """Thin wrapper around FtController for use as a Ray Actor.

    Created as a Named Async Actor so that FtTrainingRankAgent
    can find it via ``ray.get_actor(ft_controller_actor_name(ft_id))``.
    FtController remains a plain Python class for testability.

    Agent-facing methods (register_training_rank, log_step, register_node_agent)
    route to the appropriate component on FtController.
    """

    def __init__(
        self,
        *,
        builder: Callable[..., Any],
        config: Any = None,
        **kwargs: object,
    ) -> None:
        self._ctrl = builder(config=config, **kwargs)

    async def run(self) -> None:
        await self._ctrl.run()

    async def submit_and_run(self) -> None:
        await self._ctrl.submit_initial_job()
        await self._ctrl.run()

    async def shutdown(self) -> None:
        await self._ctrl.shutdown()

    async def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        self._ctrl.mini_wandb.log_step(
            run_id=run_id,
            step=step,
            metrics=metrics,
        )

    async def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int,
    ) -> None:
        self._ctrl.rank_roster.register_training_rank(
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            node_id=node_id,
            exporter_address=exporter_address,
            pid=pid,
        )

    def register_node_agent(
        self,
        node_id: str,
        agent: object,
        exporter_address: str = "",
        node_metadata: dict[str, str] | None = None,
    ) -> None:
        proxy = RayNodeAgentProxy(agent)
        self._ctrl.register_node_agent(
            node_id=node_id,
            agent=proxy,
            exporter_address=exporter_address,
            node_metadata=node_metadata,
        )

    def get_status(self) -> object:
        return self._ctrl.get_status()


FtControllerActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtControllerActorCls)
