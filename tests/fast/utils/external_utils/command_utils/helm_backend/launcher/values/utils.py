from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release="r",
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
)


def router() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: f"python -m router --host {ctx.self_addrs['primary'].host}",
    )


def engine(
    num_cells: int = 2,
    gpus_per_engine: int = 32,
    name: str = "inference-engine-0-0",
    gpu_offset: int = 0,
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name=name,
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[
            PortInfo(name="primary", static_port=8000),
            PortInfo(name="dist_init", static_port=9000, mode="master"),
            PortInfo(name="engine_info_bootstrap", static_port=12000),
        ],
        env_var=lambda ctx: {"NVSHMEM_DISABLE_NCCL": "1"},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=max(1, gpus_per_engine // 8),
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=min(gpus_per_engine, 8),
            num_gpus_per_node=8,
            pg_slot_offset=gpu_offset,
        ),
        launch_command=lambda ctx: (
            f"python -m sglang.launch_server --node-rank {ctx.worker_in_cell_index} "
            f"--dist-init-addr {ctx.self_addrs['dist_init'].host}:{ctx.self_addrs['dist_init'].port} "
            f"--base-gpu-id {ctx.gpu_ids[0]}"
        ),
    )


def trainer(num_cells: int = 2, gpus_per_cell: int = 16) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda ctx: {"NCCL_CUMEM_ENABLE": "0"},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        ctor_kwargs=lambda ctx: {},
    )


def session_server(num_cells: int) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="session-server",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=1, num_gpus_per_worker=0, num_gpu_slots_per_worker=0
        ),
        launch_command=lambda ctx: "python -m session_server",
    )


def session_client() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="rollout-executor",
        port_infos=[PortInfo(name="primary", static_port=8100)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: "python -m executor --session-servers "
        + ",".join(addrs["primary"].addr for addrs in ctx.spec_addrs["session-server"]),
    )
