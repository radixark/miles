from __future__ import annotations

import ray

from miles.utils.ft.platform.controller_factory import (
    FtControllerConfig,
    build_ft_controller,
)


class _FtControllerActorCls:
    """Thin wrapper around FtController for use as a Ray Actor.

    Created as a Named Async Actor so that FtTrainingRankAgent
    can find it via ``ray.get_actor(ft_controller_actor_name(ft_id))``.
    FtController remains a plain Python class for testability.

    Agent-facing methods (register_training_rank, log_step, register_node_agent)
    route to the appropriate component on FtController.
    """

    def __init__(self, config: FtControllerConfig | None = None, **kwargs: object) -> None:
        self._ctrl = build_ft_controller(config=config, **kwargs)

    async def run(self) -> None:
        await self._ctrl.run()

    async def submit_and_run(self) -> None:
        await self._ctrl.submit_initial_training()
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
        pid: int | None = None,
    ) -> None:
        self._ctrl.rank_registry.register_training_rank(
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            node_id=node_id,
            exporter_address=exporter_address,
            pid=pid,
        )

    def register_node_agent(self, node_id: str, agent: object) -> None:
        self._ctrl.register_node_agent(node_id=node_id, agent=agent)

    def get_status(self) -> object:
        return self._ctrl.get_status()


FtControllerActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtControllerActorCls)
