from collections.abc import Callable
from typing import TYPE_CHECKING

from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
)
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.ft_utils.health_checker import ActiveAndEpoch, SimpleHealthChecker, SimpleHealthCheckerConfig

if TYPE_CHECKING:
    from miles.ray.train.cell import TrainerCell


def create_trainer_cell_health_checker(
    *,
    cell: "TrainerCell",
    config: SimpleHealthCheckerConfig,
    get_activeness: Callable[[], ActiveAndEpoch],
) -> SimpleHealthChecker:
    async def _check() -> None:
        # Cell health is liveness, not training progress: the heartbeat RPC runs on
        # a dedicated concurrency group and returns even while the training thread is
        # blocked in a (legitimately waiting) cross-cell collective. A returned result
        # proves the process is alive; a WorkerUnreachableError or RPC timeout proves it is not.
        #
        # One answer is enough. Requiring every worker would let a single one blocked in a
        # collective time the whole cell out, and a worker that really died is reported by the
        # worker manager's own liveness scan, which asks ray rather than the application.
        if not cell.is_alive:
            return

        await cell.probe_liveness()

    return SimpleHealthChecker(
        name=f"trainer-cell-{cell.cell_id}",
        check_fn=_check,
        get_activeness=get_activeness,
        config=config,
    )


def compute_cell_status(state: CellState, health_checker_status: TriState, *, workers_hash: str) -> CellStatus:
    match state:
        case StateAllocatedAlive():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.from_health_checker_status(health_checker_status),
                ],
                workers_hash=workers_hash,
            )

        case StateAllocatedUninitialized():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.TRUE),
                ],
                workers_hash=workers_hash,
            )

        case StateAllocatedErrored():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.FALSE, reason="ExecutionErrored"),
                ],
                workers_hash=workers_hash,
            )

        case _:
            raise NotImplementedError(f"Unknown state: {state}")
