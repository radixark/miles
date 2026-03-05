from __future__ import annotations

import ray

from miles.utils.ft.platform.controller_factory import (
    FtControllerConfig,
    build_ft_controller,
)


class _FtControllerActorCls:
    """Thin wrapper around FtController for use as a Ray Actor.

    Created as a Detached Named Async Actor so that FtMegatronAgent
    can find it via ``ray.get_actor(FT_CONTROLLER_ACTOR_NAME)``.
    FtController remains a plain Python class for testability.
    """

    def __init__(self, config: FtControllerConfig | None = None, **kwargs: object) -> None:
        self._ctrl = build_ft_controller(config=config, **kwargs)

    async def run(self) -> None:
        await self._ctrl.run()

    async def shutdown(self) -> None:
        await self._ctrl.shutdown()

    async def log_step(
        self,
        run_id: str,
        rank: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        await self._ctrl.log_step(
            run_id=run_id,
            rank=rank,
            step=step,
            metrics=metrics,
        )

    async def register_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int | None = None,
    ) -> None:
        await self._ctrl.register_rank(
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            node_id=node_id,
            exporter_address=exporter_address,
            pid=pid,
        )

    def register_agent(self, node_id: str, agent: object) -> None:
        self._ctrl.register_agent(node_id=node_id, agent=agent)

    def get_status(self) -> dict[str, object]:
        return self._ctrl.get_status()


FtControllerActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtControllerActorCls)
