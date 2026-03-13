from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tests.fast.utils.ft.testbed.config import TestbedNodeConfig
from tests.fast.utils.ft.testbed.ray.train_actor import TestbedTrainRayActor

logger = logging.getLogger(__name__)


@ray.remote
class _WorkerHandleStore:
    """Shared Ray actor that stores training worker handles.

    Both the test process and the controller actor process reference the
    same ``_WorkerHandleStore`` instance, so worker handles created inside
    the controller (by ``spawn_actors``) are visible to the test side for
    fault injection (``kill_all``, ``set_hung``, etc.).
    """

    def __init__(self) -> None:
        self._workers: list[ray.actor.ActorHandle] = []
        self._node_to_workers: dict[str, list[ray.actor.ActorHandle]] = {}

    def set_workers(
        self,
        workers: list[ray.actor.ActorHandle],
        node_to_workers: dict[str, list[ray.actor.ActorHandle]],
    ) -> None:
        self._workers = workers
        self._node_to_workers = dict(node_to_workers)

    def get_all(self) -> list[ray.actor.ActorHandle]:
        return list(self._workers)

    def get_by_node(self, node_id: str) -> list[ray.actor.ActorHandle]:
        return list(self._node_to_workers.get(node_id, []))

    def clear(self) -> None:
        self._workers = []
        self._node_to_workers = {}


class TestbedRayTrainGroup:
    """Manages a group of TestbedTrainRayActor instances across Ray nodes.

    Uses NodeAffinitySchedulingStrategy to pin actors to specific Ray
    nodes, simulating multi-node training topology.

    Worker handles are stored in a shared ``_WorkerHandleStore`` Ray actor
    so that both the controller process (which creates workers via
    ``spawn_actors``) and the test process (which injects faults via
    ``kill_all``, ``set_hung``, etc.) operate on the same set of handles.
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
        self._store: ray.actor.ActorHandle = _WorkerHandleStore.remote()

    async def _get_workers(self) -> list[ray.actor.ActorHandle]:
        return await self._store.get_all.remote()

    async def _get_workers_on_node(self, node_id: str) -> list[ray.actor.ActorHandle]:
        return await self._store.get_by_node.remote(node_id)

    async def spawn_actors(self, run_id: str) -> list[ray.actor.ActorHandle]:
        workers: list[ray.actor.ActorHandle] = []
        node_to_workers: dict[str, list[ray.actor.ActorHandle]] = defaultdict(list)
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
                node_to_workers[node_config.node_id].append(worker)
                global_rank += 1

        await self._store.set_workers.remote(workers, dict(node_to_workers))

        for worker in workers:
            await worker.set_peers.remote([h for h in workers if h is not worker])

        for worker in workers:
            await worker.start.remote()

        for worker in workers:
            worker.begin_loop.remote()

        return workers

    async def kill_all(self) -> None:
        workers = await self._get_workers()
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("kill_all: failed to kill worker", exc_info=True)
        await self._store.clear.remote()

    async def kill_on_node(self, node_id: str) -> None:
        workers = await self._get_workers_on_node(node_id)
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                logger.debug("kill_on_node: failed to kill worker on %s", node_id, exc_info=True)

    @property
    def all_workers(self) -> list[ray.actor.ActorHandle]:
        return ray.get(self._store.get_all.remote())

    async def set_hung(self, hung: bool) -> None:
        workers = await self._get_workers()
        for worker in workers:
            await worker.set_hung.remote(hung)

    async def set_custom_log_metrics(self, metrics: dict[str, float]) -> None:
        workers = await self._get_workers()
        for worker in workers:
            await worker.set_custom_log_metrics.remote(metrics)

    async def all_alive(self) -> bool:
        workers = await self._get_workers()
        if not workers:
            return False
        for worker in workers:
            try:
                await asyncio.wait_for(worker.ping.remote(), timeout=2.0)
            except Exception:
                return False
        return True
