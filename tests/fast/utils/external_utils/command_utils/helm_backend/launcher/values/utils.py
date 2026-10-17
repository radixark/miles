from tests.fast.utils.workers.fake_specs import FakeCommandSpec, FakeServeSpec

from miles.backends.sglang_utils.sglang_scaling_config import SglangScalingConfig
from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import (
    build_values,
    compute_static_connections,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import MilesRunChartValues
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import DEFAULT_RPC_PORT_INFO, BaseSpec, PortInfo, SchedulingSpec

LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release="r",
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
)

SCALING = ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}))


def build_values_as_launched(
    specs: list[BaseSpec], plan: LaunchPlan, *, scaling: ScalingConfig = SCALING
) -> MilesRunChartValues:
    return build_values(
        specs, plan, scaling=scaling, static_connections=compute_static_connections(specs, scaling=scaling)
    )


def router() -> FakeCommandSpec:
    return FakeCommandSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_vars=lambda ctx: {},
        fixed_scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        command=lambda ctx: f"python -m router --host {ctx.self_addrs['primary'].host}",
    )


def engine(
    num_cells: int = 2,
    gpus_per_engine: int = 32,
    name: str = "inference-engine-0-0",
    gpu_offset: int = 0,
) -> FakeCommandSpec:
    return FakeCommandSpec(
        name=name,
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[
            PortInfo(name="primary", static_port=8000),
            PortInfo(name="dist_init", static_port=9000, mode="master"),
            PortInfo(name="engine_info_bootstrap", static_port=12000),
        ],
        env_vars=lambda ctx: {"NVSHMEM_DISABLE_NCCL": "1"},
        fixed_scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=max(1, gpus_per_engine // 8),
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=min(gpus_per_engine, 8),
            num_gpus_per_node=8,
            pg_slot_offset=gpu_offset,
        ),
        command=lambda ctx: (
            f"python -m sglang.launch_server --node-rank {ctx.worker_in_cell_index} "
            f"--dist-init-addr {ctx.self_addrs['dist_init'].host}:{ctx.self_addrs['dist_init'].port} "
            f"--base-gpu-id {ctx.gpu_ids[0]}"
        ),
    )


def trainer(num_cells: int = 2, gpus_per_cell: int = 16) -> FakeServeSpec:
    return FakeServeSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master"), DEFAULT_RPC_PORT_INFO],
        env_vars=lambda ctx: {"NCCL_CUMEM_ENABLE": "0"},
        fixed_scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        args=BaseLeafConfig(),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
    )


def session_server(num_cells: int) -> FakeCommandSpec:
    return FakeCommandSpec(
        name="session-server",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_vars=lambda ctx: {},
        fixed_scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=1, num_gpus_per_worker=0, num_gpu_slots_per_worker=0
        ),
        command=lambda ctx: "python -m session_server",
    )


def session_client() -> FakeCommandSpec:
    return FakeCommandSpec(
        name="rollout-executor",
        port_infos=[PortInfo(name="primary", static_port=8100)],
        env_vars=lambda ctx: {},
        fixed_scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        command=lambda ctx: "python -m executor --session-servers "
        + ",".join(addrs["primary"].addr for addrs in ctx.pool_addrs["session-server"]),
    )
