from __future__ import annotations

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.context import MainContext
from miles.utils.ft.controller.state_machines.main.models import MainState, NormalState, RestartingMainJobState
from miles.utils.ft.controller.state_machines.subsystem import SubsystemContext, SubsystemState, Recovering, get_known_bad_nodes
from miles.utils.ft.controller.state_machines.recovery import (
    EvictingAndRestarting,
    NotifyHumans,
    RecoveryDone,
    RecoveryState,
)
from miles.utils.ft.controller.types import ControllerMode, ControllerStatus
from miles.utils.ft.utils.state_machine import StateMachine


def recovery_phase_name(recovery: RecoveryState) -> str:
    if isinstance(recovery, EvictingAndRestarting):
        return type(recovery.restart).__name__
    return type(recovery).__name__


def build_phase_history(state_history: list[SubsystemState]) -> list[str]:
    phases: list[str] = []
    for past_state in state_history:
        if isinstance(past_state, Recovering):
            name = recovery_phase_name(past_state.recovery)
            if not phases or phases[-1] != name:
                phases.append(name)
    return phases


def _extract_training_sm(
    controller_state: MainState,
) -> StateMachine[SubsystemState, SubsystemContext] | None:
    if isinstance(controller_state, NormalState):
        training = controller_state.subsystems.get("training")
        if training is not None:
            return training.state_machine
    return None


def _extract_rollout_subsystem_states(
    controller_state: MainState,
) -> dict[str, str] | None:
    if not isinstance(controller_state, NormalState):
        return None
    result: dict[str, str] = {}
    for name, entry in controller_state.subsystems.items():
        if name.startswith("rollout_"):
            result[name] = type(entry.state_machine.state).__name__
    return result if result else None


def build_controller_status(
    *,
    controller_state_machine: StateMachine[MainState, MainContext],
    mini_wandb: MiniWandb,
    training_rank_roster: TrainingRankRoster,
    tick_count: int,
) -> ControllerStatus:
    controller_state = controller_state_machine.state
    iteration_val = mini_wandb.latest(metric_name="iteration")
    latest_iteration = int(iteration_val) if iteration_val is not None else None
    rollout_states = _extract_rollout_subsystem_states(controller_state)

    # Extract training sub-SM state
    training_sm = _extract_training_sm(controller_state)

    if isinstance(controller_state, RestartingMainJobState):
        return ControllerStatus(
            mode=ControllerMode.RECOVERY,
            recovery_phase="RestartingMainJob",
            phase_history=None,
            tick_count=tick_count,
            active_run_id=training_rank_roster.run_id,
            bad_nodes=[],
            recovery_in_progress=True,
            bad_nodes_confirmed=False,
            latest_iteration=latest_iteration,
            rollout_subsystem_states=rollout_states,
        )

    if training_sm is None:
        return ControllerStatus(
            mode=ControllerMode.MONITORING,
            recovery_phase=None,
            phase_history=None,
            tick_count=tick_count,
            active_run_id=training_rank_roster.run_id,
            bad_nodes=[],
            recovery_in_progress=False,
            bad_nodes_confirmed=False,
            latest_iteration=latest_iteration,
            rollout_subsystem_states=rollout_states,
        )

    state = training_sm.state
    phase_history = build_phase_history(training_sm.state_history)

    if isinstance(state, Recovering):
        recovery = state.recovery
        mode = ControllerMode.RECOVERY
        recovery_phase_str = recovery_phase_name(recovery)
        bad_nodes = sorted(get_known_bad_nodes(recovery))
        bad_nodes_confirmed = bool(bad_nodes) or isinstance(recovery, (NotifyHumans, RecoveryDone))
    else:
        mode = ControllerMode.MONITORING
        recovery_phase_str = None
        bad_nodes = []
        bad_nodes_confirmed = False

    return ControllerStatus(
        mode=mode,
        recovery_phase=recovery_phase_str,
        phase_history=phase_history if phase_history else None,
        tick_count=tick_count,
        active_run_id=training_rank_roster.run_id,
        bad_nodes=bad_nodes,
        recovery_in_progress=isinstance(state, Recovering),
        bad_nodes_confirmed=bad_nodes_confirmed,
        latest_iteration=latest_iteration,
        rollout_subsystem_states=rollout_states,
    )
