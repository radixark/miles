from __future__ import annotations

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalSt, RestartingMainJobSt
from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt, get_known_bad_nodes
from miles.utils.ft.controller.state_machines.recovery import (
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RecoveryDoneSt,
    RecoveryState,
)
from miles.utils.ft.controller.types import ControllerStatus, RecoveryInfo
from miles.utils.ft.utils.state_machine import StateMachine


def recovery_phase_name(recovery: RecoveryState) -> str:
    if isinstance(recovery, EvictingAndRestartingSt):
        return type(recovery.restart).__name__
    return type(recovery).__name__


def _classify_recovery(recovery: RecoveryState) -> RecoveryInfo:
    bad_nodes = sorted(get_known_bad_nodes(recovery))
    match recovery:
        case NotifyHumansSt() | RecoveryDoneSt():
            bad_nodes_confirmed = True
        case _:
            bad_nodes_confirmed = bool(bad_nodes)

    return RecoveryInfo(
        phase=recovery_phase_name(recovery),
        bad_nodes=bad_nodes,
        bad_nodes_confirmed=bad_nodes_confirmed,
    )


def _extract_rollout_subsystem_states(
    controller_state: MainState,
) -> dict[str, str] | None:
    if not isinstance(controller_state, NormalSt):
        return None
    result: dict[str, str] = {}
    for name, state in controller_state.subsystems.items():
        if name.startswith("rollout_"):
            result[name] = type(state).__name__
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

    match controller_state:
        case RestartingMainJobSt():
            recovery = RecoveryInfo(
                phase="RestartingMainJobSt",
                bad_nodes=[],
                bad_nodes_confirmed=False,
            )
            training_subsystem_state = None

        case NormalSt(subsystems=subs):
            training_state = subs.get("training")
            training_subsystem_state = type(training_state).__name__ if training_state is not None else None
            match training_state:
                case RecoveringSt(recovery=rec):
                    recovery = _classify_recovery(rec)
                case _:
                    recovery = None

        case _:
            recovery = None
            training_subsystem_state = None

    return ControllerStatus(
        tick_count=tick_count,
        active_run_id=training_rank_roster.run_id,
        latest_iteration=latest_iteration,
        training_subsystem_state=training_subsystem_state,
        rollout_subsystem_states=rollout_states,
        recovery=recovery,
    )
