from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class _RecoveryInfo:
    mode: ControllerMode
    recovery_phase: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool


_MONITORING_INFO = _RecoveryInfo(
    mode=ControllerMode.MONITORING,
    recovery_phase=None,
    bad_nodes=[],
    recovery_in_progress=False,
    bad_nodes_confirmed=False,
)


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

    match controller_state:
        case RestartingMainJobState():
            info = _RecoveryInfo(
                mode=ControllerMode.RECOVERY,
                recovery_phase="RestartingMainJob",
                bad_nodes=[],
                recovery_in_progress=True,
                bad_nodes_confirmed=False,
            )
        case NormalState(subsystems=subs):
            training_state = subs.get("training")
            match training_state:
                case Recovering(recovery=recovery):
                    info = _classify_recovery(recovery)
                case _:
                    info = _MONITORING_INFO
        case _:
            info = _MONITORING_INFO

    return ControllerStatus(
        mode=info.mode,
        recovery_phase=info.recovery_phase,
        phase_history=None,
        tick_count=tick_count,
        active_run_id=training_rank_roster.run_id,
        bad_nodes=info.bad_nodes,
        recovery_in_progress=info.recovery_in_progress,
        bad_nodes_confirmed=info.bad_nodes_confirmed,
        latest_iteration=latest_iteration,
        rollout_subsystem_states=rollout_states,
    )


def _classify_recovery(recovery: RecoveryState) -> _RecoveryInfo:
    bad_nodes = sorted(get_known_bad_nodes(recovery))
    match recovery:
        case NotifyHumans() | RecoveryDone():
            bad_nodes_confirmed = True
        case _:
            bad_nodes_confirmed = bool(bad_nodes)

    return _RecoveryInfo(
        mode=ControllerMode.RECOVERY,
        recovery_phase=recovery_phase_name(recovery),
        bad_nodes=bad_nodes,
        recovery_in_progress=True,
        bad_nodes_confirmed=bad_nodes_confirmed,
    )


def recovery_phase_name(recovery: RecoveryState) -> str:
    match recovery:
        case EvictingAndRestarting(restart=restart):
            return type(restart).__name__
        case _:
            return type(recovery).__name__


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
