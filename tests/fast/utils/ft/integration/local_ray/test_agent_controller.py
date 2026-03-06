"""Local Ray: Agent ↔ Controller — real RayActorResolver, rank registration, tracking."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import ray

from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.agents.utils.controller_handle import RayActorResolver
from miles.utils.ft.models import ControllerMode

from tests.fast.utils.ft.integration.local_ray.conftest import get_status

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.timeout(60),
]


class TestRealRayActorResolver:
    def test_resolver_finds_controller_actor(
        self, controller_actor: ray.actor.ActorHandle,
    ) -> None:
        resolver = RayActorResolver()
        handle = resolver.get_actor("ft_controller")
        status = ray.get(handle.get_status.remote(), timeout=5)
        assert status.mode == ControllerMode.MONITORING

    def test_resolver_raises_for_missing_actor(self, local_ray: None) -> None:
        resolver = RayActorResolver()
        with pytest.raises(ValueError):
            resolver.get_actor("ft_controller_nonexistent_xyz")


class TestTrainingRankAgentRegistration:
    """FtTrainingRankAgent.__init__ registers with live controller via Ray."""

    def test_rank_registers_via_ray(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_TRAINING_RUN_ID", run_id)
        monkeypatch.setenv("MILES_FT_ID", "")

        with patch("socket.gethostname", return_value="fake-node-0"):
            agent = FtTrainingRankAgent(rank=0, world_size=2)

        try:
            status = get_status(handle)
            assert status.active_run_id == run_id
        finally:
            agent.shutdown()

    def test_multiple_ranks_register_from_different_nodes(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_TRAINING_RUN_ID", run_id)
        monkeypatch.setenv("MILES_FT_ID", "")

        agents: list[FtTrainingRankAgent] = []
        try:
            for i in range(4):
                with patch("socket.gethostname", return_value=f"fake-node-{i}"):
                    agent = FtTrainingRankAgent(rank=i, world_size=4)
                    agents.append(agent)

            status = get_status(handle)
            assert status.active_run_id == run_id
        finally:
            for a in agents:
                a.shutdown()


class TestTrackingAgentLogStep:
    """FtTrackingAgent.log() fire-and-forgets log_step to the controller."""

    def test_tracking_agent_logs_metrics_to_mini_wandb(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_TRAINING_RUN_ID", run_id)
        monkeypatch.setenv("MILES_FT_ID", "")

        tracking = FtTrackingAgent(run_id=run_id)
        tracking.log(metrics={"loss": 0.5, "iteration": 10}, step=10)

        time.sleep(0.5)

        status = get_status(handle)
        assert status.latest_iteration == 10

    def test_tracking_agent_without_run_id_is_noop(
        self,
        controller_actor: ray.actor.ActorHandle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MILES_FT_TRAINING_RUN_ID", "")
        monkeypatch.setenv("MILES_FT_ID", "")

        tracking = FtTrackingAgent(run_id="")
        tracking.log(metrics={"loss": 0.5}, step=1)

        status = get_status(controller_actor)
        assert status.latest_iteration is None
