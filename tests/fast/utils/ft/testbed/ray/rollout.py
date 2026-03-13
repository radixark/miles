from __future__ import annotations

import logging
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.http_utils import MILES_HOST_IP_ENV
from tests.fast.utils.ft.testbed.backends.sglang_utils.sglang_engine import TestbedSGLangEngine

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0)
class TestbedRolloutManager:
    """Simulates a RolloutManager with real FtRolloutAgent embedded.

    Manages per-cell TestbedSGLangEngine actors and exposes
    start_cell/stop_cell/get_cell_status for RayRolloutActuator.
    The real FtRolloutAgent handles health checking and metrics.
    """

    def __init__(
        self,
        ft_id: str,
        cell_ids: list[str],
        rollout_node_mapping: dict[str, str],
    ) -> None:
        self._ft_id = ft_id
        self._cell_ids = cell_ids
        self._rollout_node_mapping = rollout_node_mapping
        self._engines: dict[str, ray.actor.ActorHandle] = {}
        self.all_rollout_engines: list[ray.actor.ActorHandle] = []

    async def init_ft_agent(self) -> None:
        """Create and register real FtRolloutAgent with the controller.

        Must be called after the controller is running, since registration
        requires the controller actor to be reachable.  Async because
        build_rollout_agent → RolloutHealthChecker.__init__ calls
        asyncio.create_task(), which requires a running event loop.
        """
        from miles.utils.ft.factories.rollout_agent import build_rollout_agent

        os.environ[MILES_HOST_IP_ENV] = "127.0.0.1"
        build_rollout_agent(
            cell_ids=self._cell_ids,
            get_engines=lambda cid: [self._engines[cid]] if cid in self._engines else [],
            ft_id=self._ft_id,
            check_interval=1.0,
        )

    def start_cell(self, cell_id: str) -> str:
        ray_node_id = self._rollout_node_mapping.get(cell_id)
        options_kwargs: dict[str, object] = {"max_restarts": 0}
        if ray_node_id:
            options_kwargs["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=ray_node_id,
                soft=False,
            )

        engine = TestbedSGLangEngine.options(**options_kwargs).remote()
        self._engines[cell_id] = engine
        self.all_rollout_engines.append(engine)
        return f"rollout-{cell_id}"

    def stop_cell(self, cell_id: str) -> None:
        engine = self._engines.pop(cell_id, None)
        if engine is not None:
            try:
                ray.kill(engine, no_restart=True)
            except Exception:
                logger.debug("stop_cell: failed to kill engine for %s", cell_id, exc_info=True)
            if engine in self.all_rollout_engines:
                self.all_rollout_engines.remove(engine)

    def get_cell_status(self, cell_id: str) -> JobStatus:
        engine = self._engines.get(cell_id)
        if engine is None:
            return JobStatus.STOPPED
        try:
            ray.get(engine.health_generate.remote(), timeout=2.0)
            return JobStatus.RUNNING
        except Exception:
            return JobStatus.FAILED

    def kill_engine(self, cell_id: str) -> None:
        """Test helper: kill engine without removing from tracking (simulates crash)."""
        engine = self._engines.get(cell_id)
        if engine is not None:
            ray.kill(engine, no_restart=True)
