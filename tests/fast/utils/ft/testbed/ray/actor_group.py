from __future__ import annotations

import logging
from collections import defaultdict

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tests.fast.utils.ft.testbed.config import TestbedNodeConfig
from tests.fast.utils.ft.testbed.ray.train_actor import TestbedTrainRayActor

logger = logging.getLogger(__name__)


class TestbedRayTrainGroup:
    """Manages a group of TestbedTrainRayActor instances across Ray nodes.

    Uses NodeAffinitySchedulingStrategy to pin actors to specific Ray
    nodes, simulating multi-node training topology.
    """

    def __init__(
        self,
        training_nodes: list[TestbedNodeConfig],
        node_mapping: dict[str, str],
        ft_id: str,
        step_interval: float,
    ) -> None:
        self._training_nodes = training_nodes
        self._node_mapping = node_mapping
        self._ft_id = ft_id
        self._step_interval = step_interval
        self._workers: list[ray.actor.ActorHandle] = []
        self._node_to_workers: dict[str, list[ray.actor.ActorHandle]] = defaultdict(list)

    async def spawn_actors(self, run_id: str) -> list[ray.actor.ActorHandle]:
        workers: list[ray.actor.ActorHandle] = []
        global_rank = 0
        world_size = sum(n.num_ranks for n in self._training_nodes)

        for node_config in self._training_nodes:
            ray_node_id = self._node_mapping[node_config.node_id]
            for _local_rank in range(node_config.num_ranks):
                worker = TestbedTrainRayActor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray_node_id,
                        soft=False,
                    ),
                    max_restarts=0,
                ).remote(
                    ft_id=self._ft_id,
                    rank=global_rank,
                    world_size=world_size,
                    node_id=node_config.node_id,
                    run_id=run_id,
                    step_interval=self._step_interval,
                )
                workers.append(worker)
                self._node_to_workers[node_config.node_id].append(worker)
                global_rank += 1

        self._workers = workers

        for worker in workers:
            await worker.set_peers.remote([h for h in workers if h is not worker])

        for worker in workers:
            await worker.start.remote()

        for worker in workers:
            worker.begin_loop.remote()

        return workers

    def kill_all(self) -> None:
        for worker in self._workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("kill_all: failed to kill worker", exc_info=True)
        self._workers.clear()
        self._node_to_workers.clear()

    def kill_on_node(self, node_id: str) -> None:
        workers = self._node_to_workers.get(node_id, [])
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("kill_on_node: failed to kill worker on %s", node_id, exc_info=True)

    @property
    def all_workers(self) -> list[ray.actor.ActorHandle]:
        return list(self._workers)

    async def set_hung(self, hung: bool) -> None:
        for worker in self._workers:
            await worker.set_hung.remote(hung)

    async def set_custom_log_metrics(self, metrics: dict[str, float]) -> None:
        for worker in self._workers:
            await worker.set_custom_log_metrics.remote(metrics)

    async def all_alive(self) -> bool:
        if not self._workers:
            return False
        for worker in self._workers:
            try:
                ray.get(worker.ping.remote(), timeout=2.0)
            except Exception:
                return False
        return True
