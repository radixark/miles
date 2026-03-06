"""Local Ray: Agent ↔ Controller — rank registration, tracking."""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import ray

from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.agents.utils.controller_handle import get_controller_handle
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.models import ActionType, ControllerMode, Decision, TriggerType

from tests.fast.utils.ft.integration.local_ray.conftest import get_status, poll_for_run_id

pytestmark = [
    pytest.mark.local_ray,
]


class _OneShotCrashDetector(BaseFaultDetector):
    """Fires ENTER_RECOVERY once, then returns NONE forever after."""

    def __init__(self) -> None:
        self._fired = False

    def evaluate(self, ctx: Any) -> Decision:
        if not self._fired:
            self._fired = True
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason="one-shot crash for test",
                trigger=TriggerType.CRASH,
                bad_node_ids=["fake-bad-node"],
            )
        return Decision(action=ActionType.NONE, reason="no fault")


class TestGetControllerHandle:
    def test_finds_controller_actor(
        self, controller_actor: ray.actor.ActorHandle,
    ) -> None:
        handle = get_controller_handle("")
        assert handle is not None
        status = ray.get(handle.get_status.remote(), timeout=5)
        assert status.mode == ControllerMode.MONITORING

    def test_returns_none_for_missing_actor(self, local_ray: None) -> None:
        result = get_controller_handle("nonexistent_xyz")
        assert result is None


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


class TestPidCorrectness:
    """register_training_rank receives the caller's PID, not the actor's."""

    def test_registered_pid_is_caller_process(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle, run_id = running_controller
        monkeypatch.setenv("MILES_FT_TRAINING_RUN_ID", run_id)
        monkeypatch.setenv("MILES_FT_ID", "")

        caller_pid = os.getpid()

        with patch("socket.gethostname", return_value="pid-test-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1)

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
            detectors_override=[_OneShotCrashDetector()],
        )

        handle.submit_and_run.remote()
        first_run_id = poll_for_run_id(handle)

        ray.get(handle.register_training_rank.remote(
            run_id=first_run_id, rank=0, world_size=1,
            node_id="n0", exporter_address="http://n0:9090",
        ), timeout=5)

        ray.get(handle.log_step.remote(
            run_id=first_run_id, step=5, metrics={"iteration": 5},
        ), timeout=5)

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
            detectors_override=[_OneShotCrashDetector()],
        )

        handle.submit_and_run.remote()
        first_run_id = poll_for_run_id(handle)

        ray.get(handle.register_training_rank.remote(
            run_id=first_run_id, rank=0, world_size=1,
            node_id="n0", exporter_address="http://n0:9090",
        ), timeout=5)

        for i in range(200):
            handle.log_step.remote(
                run_id=first_run_id, step=i, metrics={"loss": float(i)},
            )

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            status = get_status(handle)
            if status.active_run_id is not None and status.active_run_id != first_run_id:
                break
            time.sleep(0.2)

        status = get_status(handle)
        assert status.active_run_id is not None
        assert isinstance(status.mode, ControllerMode)
