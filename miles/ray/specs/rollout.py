from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

ROLLOUT_EXECUTOR_POOL_ID = "rollout-executor"
ROLLOUT_EXECUTOR_WORKER_CLASS = "miles.ray.rollout.rollout_executor.RolloutExecutor"


def spec_rollout_executor(args) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=ROLLOUT_EXECUTOR_POOL_ID,
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
        ctor_kwargs=lambda _ctx: dict(args=args),
    )


def create_rollout_executor_handle() -> BaseWorkerHandle:
    from miles.utils.workers.worker_provider.ray import RayWorkerProvider

    provider = RayWorkerProvider.create()  # TODO inject instance
    return provider.get_handle(rollout_executor_worker_name())


def rollout_executor_worker_name() -> str:
    return compute_worker_name(pool_id=ROLLOUT_EXECUTOR_POOL_ID)


def rollout_executor_cell_id() -> str:
    return compute_cell_id(pool_id=ROLLOUT_EXECUTOR_POOL_ID, cell_index=0)
