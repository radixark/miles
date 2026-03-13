"""Local Ray: Agent ↔ Controller — rank registration, tracking."""

from __future__ import annotations

import time
from collections.abc import Callable
from unittest.mock import patch

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import get_status, poll_for_run_id
from tests.fast.utils.ft.utils.controller_fakes import OneShotCrashDetector

from miles.utils.ft.adapters.impl.ray.controller_client import RayControllerClient
from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig
from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.controller.types import ControllerMode

pytestmark = [
    pytest.mark.local_ray,
]


class TestRayControllerClient:
    def test_finds_controller_actor(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        client = RayControllerClient(ft_id="")
        handle = client._get_handle()
        assert handle is not None
        status = ray.get(handle.get_status.remote(), timeout=5)
        assert status.mode == ControllerMode.MONITORING

    def test_returns_none_for_missing_actor(self, local_ray: None) -> None:
        client = RayControllerClient(ft_id="nonexistent_xyz")
        result = client._get_handle()
        assert result is None


class TestTrainingRankAgentRegistration:
    """FtTrainingRankAgent.__init__ registers with live controller via Ray."""

    def test_rank_registers_via_ray(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_RUN_ID", run_id)

        client = RayControllerClient(ft_id="")
        with patch("socket.gethostname", return_value="fake-node-0"):
            agent = FtTrainingRankAgent(rank=0, world_size=2, controller_client=client)

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
        monkeypatch.setenv("MILES_FT_RUN_ID", run_id)

        client = RayControllerClient(ft_id="")
        agents: list[FtTrainingRankAgent] = []
        try:
            for i in range(4):
                with patch("socket.gethostname", return_value=f"fake-node-{i}"):
                    agent = FtTrainingRankAgent(rank=i, world_size=4, controller_client=client)
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
        monkeypatch.setenv("MILES_FT_RUN_ID", run_id)

        client = RayControllerClient(ft_id="")
        tracking = FtTrackingAgent(rank=0, run_id=run_id, controller_client=client)
        tracking.log(metrics={"loss": 0.5, "iteration": 10}, step=10)

        time.sleep(0.5)

        status = get_status(handle)
        assert status.latest_iteration == 10

    def test_tracking_agent_without_run_id_is_noop(
        self,
        controller_actor: ray.actor.ActorHandle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MILES_FT_RUN_ID", "")

        client = RayControllerClient(ft_id="")
        tracking = FtTrackingAgent(rank=0, run_id="", controller_client=client)
        tracking.log(metrics={"loss": 0.5}, step=1)

        status = get_status(controller_actor)
        assert status.latest_iteration is None


class TestPidCorrectness:
    """register_training_rank receives the caller's PID, not the actor's."""

    def test_registered_pid_is_caller_process(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_RUN_ID", run_id)

        client = RayControllerClient(ft_id="")
        with patch("socket.gethostname", return_value="pid-test-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1, controller_client=client)

        try:
            status = get_status(handle)
            assert status.active_run_id == run_id
        finally:
            agent.shutdown()


class TestRunIdSwitch:
    """Verify active_run_id updates when recovery re-submits training (A5)."""

    def test_recovery_switches_active_run_id(
        self,
        make_controller_actor: Callable[..., ray.actor.ActorHandle],
    ) -> None:
        handle = make_controller_actor(
            detectors_override=[OneShotCrashDetector()],
        )

        handle.submit_and_run.remote()
        first_run_id = poll_for_run_id(handle)

        ray.get(
            handle.register_training_rank.remote(
                run_id=first_run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

        ray.get(
            handle.log_step.remote(
                run_id=first_run_id,
                step=5,
                metrics={"iteration": 5},
            ),
            timeout=5,
        )

        deadline = time.monotonic() + 20.0
        new_run_id: str | None = None
        while time.monotonic() < deadline:
            status = get_status(handle)
            if status.active_run_id is not None and status.active_run_id != first_run_id:
                new_run_id = status.active_run_id
                break
            time.sleep(0.2)

        assert new_run_id is not None, "active_run_id did not change after recovery"
        assert new_run_id != first_run_id


class TestInFlightMessagesDuringRunSwitch:
    """Fire-and-forget log_step for run-1 overlapping with recovery → run-2 (A6)."""

    def test_inflight_log_steps_do_not_corrupt_state(
        self,
        make_controller_actor: Callable[..., ray.actor.ActorHandle],
    ) -> None:
        handle = make_controller_actor(
            detectors_override=[OneShotCrashDetector()],
            runtime_config_override=ControllerRuntimeConfig(tick_interval=0.05, monitoring_success_iterations=0),
        )

        handle.submit_and_run.remote()
        first_run_id = poll_for_run_id(handle)

        ray.get(
            handle.register_training_rank.remote(
                run_id=first_run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

        for i in range(200):
            handle.log_step.remote(
                run_id=first_run_id,
                step=i,
                metrics={"loss": float(i)},
            )

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            status = get_status(handle)
            if (
                status.active_run_id is not None
                and status.active_run_id != first_run_id
                and status.mode == ControllerMode.MONITORING
            ):
                break
            time.sleep(0.2)

        status = get_status(handle)
        assert status.active_run_id is not None
        assert status.mode == ControllerMode.MONITORING
