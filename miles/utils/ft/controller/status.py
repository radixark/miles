from __future__ import annotations

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.subsystem_hub import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState, NormalSt, RestartingMainJobSt
from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt
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


def _classify_recovery(state: RecoveringSt) -> RecoveryInfo:
    bad_nodes = list(sorted(state.known_bad_node_ids))
    match state.recovery:
        case RecoveryDoneSt():
            bad_nodes_confirmed = True
        case _:
            bad_nodes_confirmed = bool(bad_nodes)

    return RecoveryInfo(
        phase=recovery_phase_name(state.recovery),
        bad_nodes=bad_nodes,
        bad_nodes_confirmed=bad_nodes_confirmed,
    )


def _collect_recoveries(subsystems: dict[str, object]) -> dict[str, RecoveryInfo]:
    recoveries: dict[str, RecoveryInfo] = {}
    for name, sub_state in subsystems.items():
        if isinstance(sub_state, RecoveringSt):
            recoveries[name] = _classify_recovery(sub_state)
    return recoveries


def build_controller_status(
    *,
    controller_state_machine: StateMachine[MainState, MainContext],
    mini_wandb: MiniWandb,
    training_rank_roster: TrainingRankRoster | None,
    tick_count: int,
) -> ControllerStatus:
    controller_state = controller_state_machine.state
    iteration_val = mini_wandb.latest(metric_name="iteration")
    latest_iteration = int(iteration_val) if iteration_val is not None else None

    match controller_state:
        case RestartingMainJobSt():
            recoveries = {
                controller_state.requestor_name: RecoveryInfo(
                    phase=type(controller_state).__name__,
                    bad_nodes=[],
                    bad_nodes_confirmed=False,
                ),
            }
            subsystem_states: dict[str, str] = {}

        case NormalSt(subsystems=subs):
            subsystem_states = {name: type(state).__name__ for name, state in subs.items()}
            recoveries = _collect_recoveries(subs)

        case _:
            recoveries = {}
            subsystem_states = {}

    return ControllerStatus(
        tick_count=tick_count,
        active_run_id=training_rank_roster.run_id if training_rank_roster is not None else None,
        latest_iteration=latest_iteration,
        subsystem_states=subsystem_states,
        recoveries=recoveries,
    )
