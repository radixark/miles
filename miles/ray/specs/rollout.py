from typing import Any, ClassVar, Self

from miles.ray.specs.inference import (
    SESSION_SERVER_POOL_ID,
    compute_inference_controller_provider,
    compute_router_providers,
)
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import RolloutConfig
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import BaseServeSpec, SchedulingSpec, WorkerCtorContext

ROLLOUT_EXECUTOR_POOL_ID = "rollout-executor"
ROLLOUT_EXECUTOR_WORKER_CLASS = "miles.ray.rollout.rollout_executor.RolloutExecutor"


class RolloutExecutorSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "rollout"
    config_class = RolloutConfig
    args: RolloutConfig
    name: str = ROLLOUT_EXECUTOR_POOL_ID
    worker_class: str = ROLLOUT_EXECUTOR_WORKER_CLASS

    @classmethod
    def create(cls, config: RolloutConfig) -> Self:
        return cls(args=config)

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=self.args.pin_rollout_manager_to_head,
        )

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            router_providers=compute_router_providers(ctx.args, capability=ctx.capability),
            session_server_provider=(
                ctx.capability.static_worker_provider(pool_id=SESSION_SERVER_POOL_ID)
                if ctx.args.use_session_server
                else None
            ),
            inference_controller_provider=compute_inference_controller_provider(ctx.args, capability=ctx.capability),
        )


def create_rollout_executor_handle(*, capability: BackendCapability) -> BaseWorkerHandle:
    worker_name = rollout_executor_worker_name()
    provider = capability.static_worker_provider(pool_id=ROLLOUT_EXECUTOR_POOL_ID)
    return provider.get_handle(worker_name)


def rollout_executor_worker_name() -> str:
    return compute_worker_name(pool_id=ROLLOUT_EXECUTOR_POOL_ID)


def rollout_executor_cell_id() -> str:
    return compute_cell_id(pool_id=ROLLOUT_EXECUTOR_POOL_ID, cell_index=0)
