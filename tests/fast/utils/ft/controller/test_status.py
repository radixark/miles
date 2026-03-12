"""Tests for miles.utils.ft.controller.status."""

from __future__ import annotations

from datetime import datetime

from tests.fast.utils.ft.utils.metric_injectors import make_fake_mini_wandb

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.subsystem_hub import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalSt
from miles.utils.ft.controller.state_machines.subsystem.models import DetectingAnomalySt, SubsystemState, RecoveringSt
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryDoneSt,
    StopTimeDiagnosticsSt,
)
from miles.utils.ft.controller.state_machines.restart.models import EvictingSt, StoppingAndRestartingSt
from miles.utils.ft.controller.status import build_controller_status, recovery_phase_name
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


def _now() -> datetime:
    return datetime(2025, 1, 1)


# ===================================================================
# recovery_phase_name
# ===================================================================


class TestRecoveryPhaseName:
    def test_realtime_checks(self) -> None:
        assert recovery_phase_name(RealtimeChecksSt()) == "RealtimeChecksSt"

    def test_evicting_and_restarting_returns_evicting(self) -> None:
        state = EvictingAndRestartingSt(
            restart=EvictingSt(bad_node_ids=["node-0"]),
            failed_next_state=StopTimeDiagnosticsSt(),
        )
        assert recovery_phase_name(state) == "EvictingSt"

    def test_evicting_and_restarting_returns_stopping_and_restarting(self) -> None:
        state = EvictingAndRestartingSt(
            restart=StoppingAndRestartingSt(bad_node_ids=[]),
            failed_next_state=StopTimeDiagnosticsSt(),
        )
        assert recovery_phase_name(state) == "StoppingAndRestartingSt"

    def test_stop_time_diagnostics(self) -> None:
        assert recovery_phase_name(StopTimeDiagnosticsSt()) == "StopTimeDiagnosticsSt"

    def test_notify_humans(self) -> None:
        assert recovery_phase_name(NotifyHumansSt(state_before="test")) == "NotifyHumansSt"

    def test_recovery_done(self) -> None:
        assert recovery_phase_name(RecoveryDoneSt()) == "RecoveryDoneSt"


# ===================================================================
# build_controller_status
# ===================================================================


def _make_controller_sm(
    state: SubsystemState,
) -> StateMachine[MainState, MainContext]:
    """Build a MainState SM wrapping a training SubsystemState directly."""
    controller_state = NormalSt(subsystems={"training": state})
    return StateMachine(
        initial_state=controller_state,
        stepper=StateMachineStepper(handler_map={}),
    )


class TestBuildControllerStatus:
    def test_monitoring_mode(self) -> None:
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomalySt()),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=5,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery is None
        assert status.recovery_in_progress is False

    def test_recovery_mode_with_bad_nodes(self) -> None:
        recovery = EvictingAndRestartingSt.evict_and_restart_next_stop_time_diag(bad_node_ids=["node-1"])
        state = RecoveringSt(
            recovery=recovery,
            trigger="crash",
            recovery_start_time=_now(),
        )
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(state),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=10,
        )

        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery is not None
        assert status.recovery.phase == "EvictingSt"
        assert status.recovery.bad_nodes == ["node-1"]
        assert status.recovery_in_progress is True

    def test_latest_iteration_none_when_no_data(self) -> None:
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomalySt()),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.latest_iteration is None

    def test_latest_iteration_from_mini_wandb(self) -> None:
        wandb = make_fake_mini_wandb(steps={10: {"iteration": 42.0}})
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomalySt()),
            mini_wandb=wandb,
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.latest_iteration == 42

    def test_bad_nodes_not_confirmed_in_notify_humans(self) -> None:
        """Previously NotifyHumansSt set bad_nodes_confirmed=True, but
        NotifyHumansSt can be reached without confirming bad nodes (e.g.
        diagnostic pipeline found no fault). Only RecoveryDoneSt truly
        confirms the nodes were bad.
        """
        state = RecoveringSt(
            recovery=NotifyHumansSt(state_before="test"),
            trigger="crash",
            recovery_start_time=_now(),
        )
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(state),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.recovery is not None
        assert status.recovery.bad_nodes_confirmed is False

    def test_bad_nodes_confirmed_in_recovery_done(self) -> None:
        state = RecoveringSt(
            recovery=RecoveryDoneSt(),
            trigger="crash",
            recovery_start_time=_now(),
        )
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(state),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.recovery is not None
        assert status.recovery.bad_nodes_confirmed is True

    def test_subsystem_states_detecting_anomaly(self) -> None:
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomalySt()),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.subsystem_states["training"] == "DetectingAnomalySt"

    def test_subsystem_states_recovering(self) -> None:
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger="crash",
            recovery_start_time=_now(),
        )
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(state),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.subsystem_states["training"] == "RecoveringSt"

    def test_active_run_id_from_training_rank_roster(self) -> None:
        roster = TrainingRankRoster(run_id="run-42")
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomalySt()),
            mini_wandb=MiniWandb(),
            training_rank_roster=roster,
            tick_count=0,
        )

        assert status.active_run_id == "run-42"
