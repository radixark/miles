"""Tests for miles.utils.ft.controller.status."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from tests.fast.utils.ft.utils.metric_injectors import make_fake_mini_wandb

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.context import MainContext
from miles.utils.ft.controller.state_machines.main.models import MainState, NormalState
from miles.utils.ft.controller.state_machines.subsystem.models import DetectingAnomaly, SubsystemState, Recovering
from miles.utils.ft.controller.state_machines.recovery.models import (
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    StopTimeDiagnostics,
)
from miles.utils.ft.controller.state_machines.restart.models import Evicting, StoppingAndRestarting
from miles.utils.ft.controller.status import build_controller_status, build_phase_history, recovery_phase_name
from miles.utils.ft.controller.subsystem import SubsystemEntry
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


def _now() -> datetime:
    return datetime(2025, 1, 1)


# ===================================================================
# recovery_phase_name
# ===================================================================


class TestRecoveryPhaseName:
    def test_realtime_checks(self) -> None:
        assert recovery_phase_name(RealtimeChecks()) == "RealtimeChecks"

    def test_evicting_and_restarting_returns_evicting(self) -> None:
        state = EvictingAndRestarting(
            restart=Evicting(bad_node_ids=["node-0"]),
            failed_next_state=StopTimeDiagnostics(),
        )
        assert recovery_phase_name(state) == "Evicting"

    def test_evicting_and_restarting_returns_stopping_and_restarting(self) -> None:
        state = EvictingAndRestarting(
            restart=StoppingAndRestarting(bad_node_ids=[]),
            failed_next_state=StopTimeDiagnostics(),
        )
        assert recovery_phase_name(state) == "StoppingAndRestarting"

    def test_stop_time_diagnostics(self) -> None:
        assert recovery_phase_name(StopTimeDiagnostics()) == "StopTimeDiagnostics"

    def test_notify_humans(self) -> None:
        assert recovery_phase_name(NotifyHumans(state_before="test")) == "NotifyHumans"

    def test_recovery_done(self) -> None:
        assert recovery_phase_name(RecoveryDone()) == "RecoveryDone"


# ===================================================================
# build_phase_history
# ===================================================================


class TestBuildPhaseHistory:
    def test_empty_history(self) -> None:
        assert build_phase_history([]) == []

    def test_non_recovering_states_excluded(self) -> None:
        history: list[SubsystemState] = [DetectingAnomaly(), DetectingAnomaly()]
        assert build_phase_history(history) == []

    def test_single_recovery_phase(self) -> None:
        history: list[SubsystemState] = [
            Recovering(
                recovery=RealtimeChecks(),
                trigger="crash",
                recovery_start_time=_now(),
            ),
        ]
        assert build_phase_history(history) == ["RealtimeChecks"]

    def test_duplicate_consecutive_phases_deduplicated(self) -> None:
        history: list[SubsystemState] = [
            Recovering(
                recovery=RealtimeChecks(),
                trigger="crash",
                recovery_start_time=_now(),
            ),
            Recovering(
                recovery=RealtimeChecks(),
                trigger="crash",
                recovery_start_time=_now(),
            ),
        ]
        assert build_phase_history(history) == ["RealtimeChecks"]

    def test_different_phases_preserved(self) -> None:
        history: list[SubsystemState] = [
            Recovering(
                recovery=RealtimeChecks(),
                trigger="crash",
                recovery_start_time=_now(),
            ),
            Recovering(
                recovery=StopTimeDiagnostics(),
                trigger="crash",
                recovery_start_time=_now(),
            ),
            Recovering(
                recovery=NotifyHumans(state_before="test"),
                trigger="crash",
                recovery_start_time=_now(),
            ),
        ]
        assert build_phase_history(history) == [
            "RealtimeChecks",
            "StopTimeDiagnostics",
            "NotifyHumans",
        ]

    def test_mixed_detecting_and_recovering(self) -> None:
        history: list[SubsystemState] = [
            DetectingAnomaly(),
            Recovering(
                recovery=RealtimeChecks(),
                trigger="crash",
                recovery_start_time=_now(),
            ),
            DetectingAnomaly(),
            Recovering(
                recovery=StopTimeDiagnostics(),
                trigger="hang",
                recovery_start_time=_now(),
            ),
        ]
        assert build_phase_history(history) == [
            "RealtimeChecks",
            "StopTimeDiagnostics",
        ]


# ===================================================================
# build_controller_status
# ===================================================================


def _make_controller_sm(
    state: SubsystemState,
    state_history: list[SubsystemState] | None = None,
) -> StateMachine[MainState, MainContext]:
    """Build a MainState SM wrapping a training SubsystemState SM."""
    main_sm: StateMachine = StateMachine(
        initial_state=state,
        stepper=StateMachineStepper(handler_map={}),
    )
    if state_history:
        for s in state_history:
            main_sm._state_history.append(s)

    training_entry = SubsystemEntry(
        name="training",
        state_machine=main_sm,
        actuator=MagicMock(),
    )
    controller_state = NormalState(subsystems={"training": training_entry})
    return StateMachine(
        initial_state=controller_state,
        stepper=StateMachineStepper(handler_map={}),
    )


class TestBuildControllerStatus:
    def test_monitoring_mode(self) -> None:
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomaly()),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=5,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_phase is None
        assert status.bad_nodes == []
        assert status.recovery_in_progress is False
        assert status.bad_nodes_confirmed is False

    def test_recovery_mode_with_bad_nodes(self) -> None:
        recovery = EvictingAndRestarting.evict_and_restart_next_stop_time_diag(bad_node_ids=["node-1"])
        state = Recovering(
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
        assert status.recovery_phase == "Evicting"
        assert status.bad_nodes == ["node-1"]
        assert status.recovery_in_progress is True

    def test_latest_iteration_none_when_no_data(self) -> None:
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomaly()),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.latest_iteration is None

    def test_latest_iteration_from_mini_wandb(self) -> None:
        wandb = make_fake_mini_wandb(steps={10: {"iteration": 42.0}})
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomaly()),
            mini_wandb=wandb,
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.latest_iteration == 42

    def test_bad_nodes_confirmed_in_notify_humans(self) -> None:
        state = Recovering(
            recovery=NotifyHumans(state_before="test"),
            trigger="crash",
            recovery_start_time=_now(),
        )
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(state),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.bad_nodes_confirmed is True

    def test_bad_nodes_confirmed_in_recovery_done(self) -> None:
        state = Recovering(
            recovery=RecoveryDone(),
            trigger="crash",
            recovery_start_time=_now(),
        )
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(state),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.bad_nodes_confirmed is True

    def test_phase_history_populated(self) -> None:
        past = Recovering(
            recovery=RealtimeChecks(),
            trigger="crash",
            recovery_start_time=_now(),
        )
        current = DetectingAnomaly()
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(current, state_history=[past]),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=5,
        )

        assert status.phase_history == ["RealtimeChecks"]

    def test_phase_history_none_when_empty(self) -> None:
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomaly()),
            mini_wandb=MiniWandb(),
            training_rank_roster=TrainingRankRoster(),
            tick_count=0,
        )

        assert status.phase_history is None

    def test_active_run_id_from_training_rank_roster(self) -> None:
        roster = TrainingRankRoster(run_id="run-42")
        status = build_controller_status(
            controller_state_machine=_make_controller_sm(DetectingAnomaly()),
            mini_wandb=MiniWandb(),
            training_rank_roster=roster,
            tick_count=0,
        )

        assert status.active_run_id == "run-42"
