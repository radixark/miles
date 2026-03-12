from __future__ import annotations

import asyncio
import logging
import os

import ray

from miles.utils.ft.adapters.impl.ray.controller_client import RayControllerClient
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.utils.env import get_training_run_id

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0, max_restarts=0)
class TestbedTrainRayActor:
    """Simulates a training rank process with real FtTrainingRankAgent.

    Calls real build_training_rank_agent to create FtTrainingRankAgent
    (rank registration, metrics exposition). Loops advancing iteration
    count and checking peer liveness. When a peer dies, calls
    exit_actor() to simulate NCCL timeout cascade.
    """

    def __init__(
        self,
        ft_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        run_id: str,
        step_interval: float,
    ) -> None:
        self._ft_id = ft_id
        self._rank = rank
        self._world_size = world_size
        self._node_id = node_id
        self._run_id = run_id
        self._step_interval = step_interval
        self._peers: list[ray.actor.ActorHandle] = []
        self._agent: FtTrainingRankAgent | None = None
        self._controller_client: RayControllerClient | None = None
        self._iteration: int = 0
        self._loop_task: asyncio.Task[None] | None = None
        self._hung: bool = False
        self._custom_log_metrics: dict[str, float] = {}

    def set_peers(self, peers: list[ray.actor.ActorHandle]) -> None:
        self._peers = peers

    def set_hung(self, hung: bool) -> None:
        self._hung = hung

    def set_custom_log_metrics(self, metrics: dict[str, float]) -> None:
        self._custom_log_metrics = metrics

    async def start(self) -> None:
        os.environ["MILES_FT_ID"] = self._ft_id
        os.environ["MILES_FT_TRAINING_RUN_ID"] = self._run_id

        self._controller_client = RayControllerClient(ft_id=self._ft_id)
        self._agent = FtTrainingRankAgent(
            rank=self._rank,
            world_size=self._world_size,
            controller_client=self._controller_client,
            node_id=self._node_id,
        )
        self._iteration = 0
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while True:
            if not await self._all_peers_alive():
                logger.warning(
                    "Rank %d: peer dead, simulating NCCL timeout",
                    self._rank,
                )
                self._shutdown_agent()
                ray.actor.exit_actor()

            if not self._hung:
                self._iteration += 1
                if self._agent is not None:
                    self._agent.step()

                if self._controller_client is not None:
                    try:
                        merged_metrics: dict[str, float] = {"iteration": float(self._iteration)}
                        merged_metrics.update(self._custom_log_metrics)
                        self._controller_client.log_step(
                            run_id=self._run_id,
                            step=self._iteration,
                            metrics=merged_metrics,
                        )
                    except Exception:
                        logger.debug("log_step failed", exc_info=True)

            await asyncio.sleep(self._step_interval)

    async def _all_peers_alive(self) -> bool:
        for peer in self._peers:
            try:
                ray.get(peer.ping.remote(), timeout=2.0)
            except Exception:
                return False
        return True

    def _shutdown_agent(self) -> None:
        if self._agent is not None:
            self._agent.shutdown()
            self._agent = None

    def ping(self) -> bool:
        return True

    def get_iteration(self) -> int:
        return self._iteration
