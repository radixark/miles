from typing import Literal

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources

CELL_RESUME_OBSERVED_TIMEOUT_SECONDS: float = 300.0


class StopCellAction(BaseFaultAction):
    kind: Literal["stop_cell"] = "stop_cell"
    cell_id: str

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        assert resources.cell_operations is not None, "Stopping a cell needs the controller's cell operations"
        await resources.cell_operations.suspend(cell_id=self.cell_id)


class StartCellAction(BaseFaultAction):
    kind: Literal["start_cell"] = "start_cell"
    cell_id: str

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        controller, operations = resources.controller, resources.cell_operations
        assert controller is not None and operations is not None, "Starting a cell needs the controller"
        await operations.resume(cell_id=self.cell_id)

        async def _check(_remaining: float) -> None:
            if self.cell_id not in controller.cell_ids:
                raise TimeoutError(f"{self.cell_id} was resumed but is not observed yet")

        await retry_until_deadline(
            _check,
            total_seconds=CELL_RESUME_OBSERVED_TIMEOUT_SECONDS,
            retry_on=TimeoutError,
            initial_delay=1.0,
            max_delay=5.0,
            log_fields=dict(tag="ft", op="wait_cell_observed", cell=self.cell_id),
        )
