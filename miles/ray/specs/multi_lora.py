from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

MULTI_LORA_CONTROLLER_POOL_ID = "multi-lora-controller"
MULTI_LORA_CONTROLLER_WORKER_CLASS = "miles.ray.multi_lora.controller.MultiLoRAController"


def spec_multi_lora_controller(args) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=MULTI_LORA_CONTROLLER_POOL_ID,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1 if is_multi_lora_enabled(args) else 0,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=0,
            # Pinned to the head node so the API sits at a port-forwardable address.
            pin_to_head=True,
        ),
        worker_class=MULTI_LORA_CONTROLLER_WORKER_CLASS,
        ctor_kwargs=lambda _ctx: dict(args=args),
    )


def create_multi_lora_controller_handle() -> BaseWorkerHandle:
    from miles.utils.workers.worker_provider.ray import RayWorkerProvider

    provider = RayWorkerProvider.create()  # TODO inject instance
    return provider.get_handle(multi_lora_controller_worker_name())


def multi_lora_controller_worker_name() -> str:
    return compute_worker_name(pool_id=MULTI_LORA_CONTROLLER_POOL_ID)


def multi_lora_controller_cell_id() -> str:
    return compute_cell_id(pool_id=MULTI_LORA_CONTROLLER_POOL_ID, cell_index=0)
