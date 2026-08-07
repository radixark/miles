from __future__ import annotations

from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

RELEASE = "miles-run-c0ffee"


def make_router_spec() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


def make_engine_spec() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="engine",
        port_infos=[PortInfo(name="primary", static_port=8000), PortInfo(name="nccl", static_port=10000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=2, num_workers_per_cell=1, num_gpus_per_worker=1, num_gpu_slots_per_worker=8
        ),
        launch_command=lambda context: "python -m engine",
    )


def make_trainer_spec(*, num_workers_per_cell: int, port_infos: list[PortInfo] | None = None) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-actor",
        port_infos=port_infos or [PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
        ),
        worker_class="miles.fake.TrainWorker",
        ctor_kwargs=lambda context: {},
        meta=lambda context: dict(role="actor", cell_index=context.cell_index),
    )
