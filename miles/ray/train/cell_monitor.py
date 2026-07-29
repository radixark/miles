from typing import TYPE_CHECKING

from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
    StatePending,
    StateStopped,
)
from miles.utils.ft_utils.control_server.models import CellCondition, CellStatus, TriState
from miles.utils.ft_utils.health_checker import SimpleHealthChecker, SimpleHealthCheckerConfig

if TYPE_CHECKING:
    from miles.ray.train.cell import RayTrainCell


def create_trainer_cell_health_checker(
    *,
    cell: "RayTrainCell",
    config: SimpleHealthCheckerConfig,
) -> SimpleHealthChecker:
    async def _check() -> None:
        # Cell health is liveness, not training progress: the heartbeat RPC runs on
        # a dedicated concurrency group and returns even while the training thread is
        # blocked in a (legitimately waiting) cross-cell collective. A returned result
        # proves the process is alive; an RayActorError or RPC timeout proves it is not.
        if not cell.is_alive:
            return

        await cell.execute("get_heartbeat_status", kill_on_failure=False)

    return SimpleHealthChecker(
        name=f"trainer-cell-{cell.cell_index}",
        check_fn=_check,
        config=config,
    )


def compute_cell_status(state: CellState, health_checker_status: TriState) -> CellStatus:
    match state:
        case StateAllocatedAlive():
            match health_checker_status:
                case TriState.FALSE:
                    healthy = CellCondition.healthy(TriState.FALSE, reason="HealthCheckFailed")
                case TriState.UNKNOWN:
                    healthy = CellCondition.healthy(TriState.UNKNOWN, reason="HealthCheckUnknown")
                case TriState.TRUE:
                    healthy = CellCondition.healthy(TriState.TRUE)
            return CellStatus(phase="Running", conditions=[CellCondition.allocated(TriState.TRUE), healthy])

        case StateAllocatedUninitialized():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.TRUE),
                ],
            )

        case StateAllocatedErrored():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.FALSE, reason="ExecutionErrored"),
                ],
            )

        case StatePending():
            return CellStatus(phase="Pending", conditions=[CellCondition.allocated(TriState.FALSE)])

        case StateStopped():
            return CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])

        case _:
            raise NotImplementedError(f"Unknown state: {state}")
