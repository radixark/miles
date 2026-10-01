from __future__ import annotations

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.workers.worker_spec import (
    BaseCommandSpec,
    BaseServeSpec,
    BaseSpec,
    PortInfo,
    SchedulingSpec,
    SpecMetaFn,
)

_RELEASE = "miles-run-c0ffee"


def make_pool_spec(
    pool_id: str,
    *,
    ports: dict[str, int],
    worker_class: str | None = None,
    meta: SpecMetaFn | None = None,
    workers_per_pod: int = 1,
) -> BaseSpec:
    common = dict(
        name=pool_id,
        port_infos=[PortInfo(name=name, static_port=port) for name, port in ports.items()],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=workers_per_pod,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=workers_per_pod,
        ),
        meta=meta,
    )
    if worker_class is None:
        return BaseCommandSpec(**common, launch_command=lambda context: f"python -m {pool_id}")
    return BaseServeSpec(**common, worker_class=worker_class, ctor_kwargs=lambda context: {})


def make_router_spec() -> BaseCommandSpec:
    return BaseCommandSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


def make_engine_spec() -> BaseCommandSpec:
    return BaseCommandSpec(
        name="engine",
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[PortInfo(name="primary", static_port=8000), PortInfo(name="nccl", static_port=10000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=2,
            num_workers_per_cell=1,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=8,
            num_gpus_per_node=8,
        ),
        launch_command=lambda context: "python -m engine",
    )


def make_trainer_spec(
    *, num_workers_per_cell: int, num_gpus_per_node: int = 8, port_infos: list[PortInfo] | None = None
) -> BaseServeSpec:
    return BaseServeSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=port_infos or [PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
        ),
        worker_class="miles.fake.TrainWorker",
        ctor_kwargs=lambda context: {},
        meta=lambda context: dict(role="actor", cell_index=context.cell_index),
    )
