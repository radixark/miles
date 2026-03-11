from __future__ import annotations

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_roster import RankRoster
from miles.utils.ft.controller.state_machines.main import MainContext, MainState, Recovering, get_known_bad_nodes
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


def build_phase_history(state_history: list[MainState]) -> list[str]:
    phases: list[str] = []
    for past_state in state_history:
        if isinstance(past_state, Recovering):
            name = recovery_phase_name(past_state.recovery)
            if not phases or phases[-1] != name:
                phases.append(name)
    return phases


def build_controller_status(
    *,
    state_machine: StateMachine[MainState, MainContext],
    mini_wandb: MiniWandb,
    rank_roster: RankRoster,
    tick_count: int,
) -> ControllerStatus:
    state = state_machine.state
    iteration_val = mini_wandb.latest(metric_name="iteration")
    latest_iteration = int(iteration_val) if iteration_val is not None else None

    phase_history = build_phase_history(state_machine.state_history)

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
        active_run_id=rank_roster.run_id,
        bad_nodes=bad_nodes,
        recovery_in_progress=isinstance(state, Recovering),
        bad_nodes_confirmed=bad_nodes_confirmed,
        latest_iteration=latest_iteration,
    )
