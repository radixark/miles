import json
import logging
from typing import TYPE_CHECKING, Literal

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.test_utils.fault_injector.actions.base import (
    BaseFaultAction,
    FaultHookContext,
    FaultHookResources,
    _assert_synchronous_orchestrator,
)

if TYPE_CHECKING:
    from miles.utils.workers.cell_operations.base import BaseCellOperations
    from miles.utils.workers.worker_handle import BaseWorkerHandle
    from miles.utils.workers.worker_provider.base import CellInfo

logger = logging.getLogger(__name__)

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


class RestartRolloutCellAction(BaseFaultAction):
    kind: Literal["restart_rollout_cell"] = "restart_rollout_cell"
    cell_id: str

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        args = resources.args
        assert args is not None, "Restarting a rollout cell needs orchestration arguments"
        _assert_synchronous_orchestrator(args, trainer_model_id=context.trainer_model_id, action=self.kind)
        assert "rollout" in args.ft_components, "Rollout restart testing requires rollout fault tolerance"
        # Only this test action needs the backend and serving dependencies.
        from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE, create_inference_controller_handle
        from miles.ray.wiring import compute_backend_capability
        from miles.utils.workers.types import ClusterBackend

        capability = compute_backend_capability(args)
        receipt = await _restart_rollout_cell(
            self.cell_id,
            operations=capability.cell_operations(),
            controller=create_inference_controller_handle(capability=capability),
            category=POOL_CATEGORY_INFERENCE_ENGINE,
            auto_resume=ClusterBackend(args.cluster_backend) == ClusterBackend.KUBERNETES,
        )
        logger.warning("[ft test rollout restart] %s", json.dumps({"at_rollout": context.rollout_id, **receipt}))


async def _restart_rollout_cell(
    cell_id: str,
    operations: "BaseCellOperations",
    controller: "BaseWorkerHandle",
    category: str,
    auto_resume: bool,
) -> dict[str, object]:
    """Restart before the update lock is taken, then wait for the fresh cell to be registered and ready."""
    before = await operations.cell_infos(pool_ids=None, category=category)
    assert cell_id in before, f"{cell_id} is not an observed rollout cell"
    original = before[cell_id]
    assert original.alive, f"{cell_id} was not alive before the test restart"
    await operations.suspend(cell_id=cell_id)
    if not auto_resume:
        await operations.resume(cell_id=cell_id)

    async def _check(_remaining: float) -> dict[str, "CellInfo"]:
        after = await operations.cell_infos(pool_ids=None, category=category)
        replacement = after.get(cell_id)
        statuses = await controller.get_cell_statuses()
        status = statuses.get(cell_id)
        if (
            replacement is None
            or not replacement.alive
            or replacement.workers_hash == original.workers_hash
            or status is None
            or status.workers_hash != replacement.workers_hash
            or status.phase != "Running"
        ):
            raise TimeoutError(f"{cell_id} has not registered a ready replacement")
        return after

    after = await retry_until_deadline(
        _check,
        total_seconds=3600,
        retry_on=TimeoutError,
        initial_delay=1.0,
        max_delay=5.0,
        log_fields=dict(tag="ft", op="wait_rollout_replacement", cell=cell_id),
    )

    def identity(cell: "CellInfo") -> dict[str, object]:
        return dict(
            workers_hash=cell.workers_hash,
            worker_names=cell.worker_names,
            gpu_offset=cell.meta["gpu_offset"],
            model_id=cell.meta["model_id"],
        )

    return dict(
        cell_id=cell_id,
        original=identity(original),
        replacement=identity(after[cell_id]),
        other_cells={
            name: dict(before=cell.workers_hash, after=after[name].workers_hash if name in after else None)
            for name, cell in before.items()
            if name != cell_id
        },
    )
