from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import ray

from miles.utils.ft.adapters.impl.ray.node_agent_proxy import RayNodeAgentProxy


class _FtControllerActorCls:
    """Thin wrapper around FtController for use as a Ray Actor.

    Created as a Named Async Actor so that FtTrainingRankAgent
    can find it via ``ray.get_actor(ft_controller_actor_name(ft_id))``.
    FtController remains a plain Python class for testability.

    Agent-facing methods (register_training_rank, log_step, register_node_agent)
    route to the appropriate component on FtController or SubsystemHub.
    """

    def __init__(
        self,
        *,
        builder: Callable[..., Any],
        config: Any = None,
        **kwargs: object,
    ) -> None:
        bundle = builder(config=config, **kwargs)
        self._ctrl = bundle.controller
        self._subsystem_hub = bundle.subsystem_hub

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
        self._ctrl.metric_store.mini_wandb.log_step(
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
        self._subsystem_hub.training_rank_roster.register_training_rank(
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

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self._ctrl.add_scrape_target(target_id=target_id, address=address)

    def is_ready(self) -> bool:
        return self._ctrl.is_ready()

    def get_status(self) -> object:
        return self._ctrl.get_status()

    async def register_rollout(
        self,
        rollout_manager_handle: object,
        metrics_address: str = "",
        cell_node_ids: dict[str, list[str]] | None = None,
    ) -> None:
        self._subsystem_hub.set_rollout_handle(rollout_manager_handle)
        if metrics_address:
            self._ctrl.add_scrape_target("rollout-ft-agent", metrics_address)
        if cell_node_ids:
            for cell_id, node_ids in cell_node_ids.items():
                self._subsystem_hub.set_rollout_node_ids(cell_id, node_ids)

    def set_rollout_node_ids(self, cell_id: str, node_ids: Iterable[str]) -> None:
        self._subsystem_hub.set_rollout_node_ids(cell_id, node_ids)


FtControllerActor = ray.remote(
    num_gpus=0,
    # Accepted product decision: controller restart is not part of the FT
    # contract. Runtime state is intentionally in-memory only, and we do not
    # attempt to reconstruct rank / node-agent / rollout registrations after a
    # controller crash. Keep both Ray auto-restart and actor-task auto-retry
    # disabled so this failure mode is explicit instead of silently entering
    # an unsupported partially-restored state or hanging behind infinite task
    # retries. Future audits should treat controller restart handling as a
    # non-goal unless product requirements change.
)(_FtControllerActorCls)
