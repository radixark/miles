"""Local Ray: Failure resilience — controller death, stale handles, cooldown."""
from __future__ import annotations

import time

import pytest
import ray

from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.utils.controller_handle import (
    ControllerHandleMixin,
)
from miles.utils.ft.models import ControllerMode
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.timeout(60),
]


def _kill_actor(name: str) -> None:
    try:
        ray.kill(ray.get_actor(name), no_restart=True)
    except ValueError:
        pass


class TestAgentSurvivesControllerDeath:
    """Agents should not crash when the controller actor is killed."""

    def test_tracking_agent_log_survives_controller_kill(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_TRAINING_RUN_ID", run_id)
        monkeypatch.setenv("MILES_FT_ID", "")

        tracking = FtTrackingAgent(run_id=run_id)
        tracking.log(metrics={"loss": 0.5}, step=1)

        ray.kill(handle, no_restart=True)
        time.sleep(0.5)

        tracking.log(metrics={"loss": 0.3}, step=2)


class TestStaleHandleAfterKill:
    """After controller restarts (max_restarts=-1), the restarted actor
    should be discoverable again."""

    def test_restarted_actor_has_fresh_state(
        self, controller_actor: ray.actor.ActorHandle,
    ) -> None:
        ray.kill(controller_actor, no_restart=False)
        time.sleep(2.0)

        name = ft_controller_actor_name("")
        handle = ray.get_actor(name)
        status = ray.get(handle.get_status.remote(), timeout=5)
        assert status.mode == ControllerMode.MONITORING
        assert status.tick_count == 0


class TestCooldownAfterLookupFailure:
    """ControllerHandleMixin suppresses repeated lookup attempts after failure."""

    def test_cooldown_suppresses_immediate_retry(
        self, local_ray: None,
    ) -> None:
        mixin = ControllerHandleMixin(ft_id="nonexistent_test_1234")

        handle = mixin._get_controller_handle()
        assert handle is None
        assert mixin._last_lookup_failure_time is not None

        handle = mixin._get_controller_handle()
        assert handle is None


class TestControllerTimeoutBehavior:
    def test_ray_get_timeout_on_dead_actor(self, local_ray: None) -> None:
        name = ft_controller_actor_name("timeout-test")
        handle = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", ft_id="timeout-test"),
        )
        ray.get(handle.get_status.remote(), timeout=5)

        ray.kill(handle, no_restart=True)
        time.sleep(0.5)

        with pytest.raises(ray.exceptions.RayError):
            ray.get(handle.get_status.remote(), timeout=2)
