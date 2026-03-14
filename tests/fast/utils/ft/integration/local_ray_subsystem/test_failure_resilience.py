"""Local Ray: Failure resilience — controller death, stale handles, agent graceful degradation."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import get_status

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
from miles.utils.ft.adapters.impl.ray.controller_client import RayControllerClient
from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.factories.controller.from_config import build_ft_controller

pytestmark = [
    pytest.mark.local_ray,
]


class TestAgentSurvivesControllerDeath:
    """Agents should not crash when the controller actor is killed."""

    def test_tracking_agent_log_survives_controller_kill(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_RUN_ID", run_id)

        client = RayControllerClient(ft_id="")
        tracking = FtTrackingAgent(rank=0, run_id=run_id, controller_client=client)
        tracking.log(metrics={"loss": 0.5}, step=1)

        ray.kill(handle, no_restart=True)
        time.sleep(0.5)

        tracking.log(metrics={"loss": 0.3}, step=2)


class TestStaleHandleAfterKill:
    """After controller restarts (max_restarts=-1), the restarted actor
    should be discoverable again."""

    def test_restarted_actor_has_fresh_state(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        ray.kill(controller_actor, no_restart=False)

        name = ft_controller_actor_name("")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                handle = ray.get_actor(name)
                status = ray.get(handle.get_status.remote(), timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise TimeoutError("Actor did not restart within 30s")

        assert status.mode == ControllerMode.MONITORING
        assert status.tick_count == 0


class TestControllerTimeoutBehavior:
    def test_ray_get_timeout_on_dead_actor(self, local_ray: None) -> None:
        name = ft_controller_actor_name("timeout-test")
        handle = FtControllerActor.options(name=name).remote(
            builder=build_ft_controller,
            config=FtControllerConfig(platform="stub", ft_id="timeout-test", rollout_num_cells=0),
        )
        ray.get(handle.get_status.remote(), timeout=5)

        ray.kill(handle, no_restart=True)
        time.sleep(0.5)

        with pytest.raises(ray.exceptions.RayError):
            ray.get(handle.get_status.remote(), timeout=2)


class TestAgentCreatedWithoutController:
    """FtTrainingRankAgent.__init__ should not crash when no controller exists (F8)."""

    def test_training_rank_agent_survives_missing_controller(
        self,
        local_ray: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MILES_FT_RUN_ID", "no-ctrl-test")

        client = RayControllerClient(ft_id="nonexistent_ft_999")
        with patch("socket.gethostname", return_value="fake-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1, controller_client=client)

        agent.shutdown()

    def test_tracking_agent_survives_missing_controller(
        self,
        local_ray: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MILES_FT_RUN_ID", "no-ctrl-track")

        client = RayControllerClient(ft_id="nonexistent_ft_999")
        tracking = FtTrackingAgent(rank=0, run_id="no-ctrl-track", controller_client=client)
        tracking.log(metrics={"loss": 0.5}, step=1)


class TestBlockingCallRetryAfterControllerDeath:
    """FtTrainingRankAgent register uses ray.get(timeout=10) + retry_sync (F2)."""

    def test_register_rank_retries_gracefully_on_dead_controller(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller

        ray.kill(handle, no_restart=True)
        time.sleep(0.5)

        monkeypatch.setenv("MILES_FT_RUN_ID", run_id)

        client = RayControllerClient(ft_id="")
        with patch("socket.gethostname", return_value="fake-retry-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1, controller_client=client)

        agent.shutdown()


class TestApplicationExceptionNotRetried:
    """max_task_retries=-1 does not retry application-level exceptions (F7).

    Ray retries system failures (actor crash, OOM) but not exceptions raised
    by the application code within a method.
    """

    def test_value_error_raises_once_not_retried(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        error_count = 0
        for _ in range(3):
            try:
                ray.get(
                    handle.register_training_rank.remote(
                        run_id=run_id,
                        rank=0,
                        world_size=1,
                        node_id="",
                        exporter_address="",
                        pid=1000,
                    ),
                    timeout=5,
                )
            except ray.exceptions.RayTaskError:
                error_count += 1

        assert error_count == 3

        status = get_status(handle)
        assert isinstance(status.mode, ControllerMode)
