from miles.ray.specs.inference import SESSION_SERVER_SPEC_NAME, compute_router_spec_name
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

ROLLOUT_EXECUTOR_SPEC_NAME = "rollout-executor"
ROLLOUT_EXECUTOR_WORKER_CLASS = "miles.ray.rollout.rollout_executor.RolloutExecutor"


def spec_rollout_executor(args) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=ROLLOUT_EXECUTOR_SPEC_NAME,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        worker_class=ROLLOUT_EXECUTOR_WORKER_CLASS,
        ctor_kwargs=lambda ctx: dict(
            args=args,
            router_provider=ctx.capability.static_worker_provider(spec_name=compute_router_spec_name(0)),
            session_server_provider=(
                ctx.capability.static_worker_provider(spec_name=SESSION_SERVER_SPEC_NAME)
                if args.use_session_server
                else None
            ),
        ),
    )


def create_rollout_executor_handle(*, capability: BackendCapability) -> BaseWorkerHandle:
    worker_name = rollout_executor_worker_name()
    provider = capability.static_worker_provider(spec_name=ROLLOUT_EXECUTOR_SPEC_NAME)
    return provider.get_handle(worker_name)


def rollout_executor_worker_name() -> str:
    return compute_worker_name(spec_name=ROLLOUT_EXECUTOR_SPEC_NAME)


def rollout_executor_cell_id() -> str:
    return compute_cell_id(spec_name=ROLLOUT_EXECUTOR_SPEC_NAME, cell_index=0)
