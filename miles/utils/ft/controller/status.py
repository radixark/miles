from __future__ import annotations

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalState, RestartingMainJobState
from miles.utils.ft.controller.state_machines.subsystem import SubsystemState, Recovering, get_known_bad_nodes
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


def _extract_training_state(controller_state: MainState) -> SubsystemState | None:
    if isinstance(controller_state, NormalState):
        return controller_state.subsystems.get("training")
    return None


def _extract_rollout_subsystem_states(
    controller_state: MainState,
) -> dict[str, str] | None:
    if not isinstance(controller_state, NormalState):
        return None
    result: dict[str, str] = {}
    for name, sub_state in controller_state.subsystems.items():
        if name.startswith("rollout_"):
            result[name] = type(sub_state).__name__
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

    training_state = _extract_training_state(controller_state)

    if training_state is None:
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

    if isinstance(training_state, Recovering):
        recovery = training_state.recovery
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
        phase_history=None,
        tick_count=tick_count,
        active_run_id=training_rank_roster.run_id,
        bad_nodes=bad_nodes,
        recovery_in_progress=isinstance(training_state, Recovering),
        bad_nodes_confirmed=bad_nodes_confirmed,
        latest_iteration=latest_iteration,
        rollout_subsystem_states=rollout_states,
    )
