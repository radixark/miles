import os
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import (
    ACTOR_ROLE,
    CRITIC_ROLE,
    compute_trainer_args,
    resolve_megatron_config,
)
from miles.ray.specs.inference import create_inference_controller_handle
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import (
    MASTER_PORT_NAME,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerLaunchContext,
)

POOL_CATEGORY_TRAINER_ENGINE = "trainer_engine"

TRAINER_CONCURRENCY_GROUPS = {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}
TRAINER_METHOD_CONCURRENCY_GROUPS = {
    "get_heartbeat_status": "heartbeat_status",
    "inject_fault": "fault_injector",
    "kill_self": "kill_self",
}

TRAINER_CONTROLLER_WORKER_CLASS = "miles.ray.train.group.TrainerController"

_TRAINER_ACTOR_CLASSES = {
    "megatron": "miles.backends.megatron_utils.actor.MegatronTrainRayActor",
    "fsdp": "miles.backends.fsdp_utils.actor.FSDPTrainRayActor",
}

_NUM_GPUS_PER_TRAINER_WORKER = 0.4


def spec_trainer_controller_actor(args) -> ServeWorkerSpec:
    return _compute_spec_trainer_controller(
        role="actor",
        with_ref=args.kl_coef != 0 or args.use_kl_loss,
        with_opd_teacher=args.use_opd and args.opd_type == "megatron",
        drives_inference=True,
    )


def spec_trainer_controller_critic(args) -> ServeWorkerSpec:
    return _compute_spec_trainer_controller(
        role="critic",
        with_ref=False,
        with_opd_teacher=False,
        drives_inference=False,
    )


def create_trainer_controller_handle(*, capability: BackendCapability, role: str) -> BaseWorkerHandle:
    worker_name = trainer_controller_worker_name(role)
    provider = capability.static_worker_provider(pool_id=compute_trainer_controller_pool_id(role))
    return provider.get_handle(worker_name)


def compute_trainer_controller_pool_id(role: str) -> str:
    return f"trainer-controller-{role}"


def trainer_controller_worker_name(role: str) -> str:
    return compute_worker_name(pool_id=compute_trainer_controller_pool_id(role))


def trainer_controller_cell_id(role: str) -> str:
    return compute_cell_id(pool_id=compute_trainer_controller_pool_id(role), cell_index=0)


def _compute_spec_trainer_controller(
    *,
    role: str,
    with_ref: bool,
    with_opd_teacher: bool,
    drives_inference: bool,
) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=compute_trainer_controller_pool_id(role),
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
        ),
        worker_class=TRAINER_CONTROLLER_WORKER_CLASS,
        ctor_kwargs=lambda ctx: dict(
            role=role,
            with_ref=with_ref,
            with_opd_teacher=with_opd_teacher,
            cell_provider=ctx.capability.dynamic_worker_provider(pool_ids=[compute_trainer_pool_id(role)]),
            cell_operations=ctx.capability.cell_operations(),
            inference_controller=(
                create_inference_controller_handle(capability=ctx.capability) if drives_inference else None
            ),
        ),
    )


def specs_trainer(args) -> list[ServeWorkerSpec]:
    specs = [
        _compute_spec_trainer(
            compute_actor_args(args),
            role="actor",
            num_nodes=args.actor_num_nodes,
            num_gpus_per_node=args.actor_num_gpus_per_node,
        )
    ]
    if args.use_critic:
        specs.append(
            _compute_spec_trainer(
                compute_critic_args(args),
                role="critic",
                num_nodes=args.critic_num_nodes,
                num_gpus_per_node=args.critic_num_gpus_per_node,
            )
        )
    return specs


def compute_trainer_pool_id(role: str) -> str:
    return f"trainer-engine-{role}"


def compute_trainer_num_cells(args, *, role: str) -> int:
    num_nodes, num_gpus_per_node = (
        (args.actor_num_nodes, args.actor_num_gpus_per_node)
        if role == "actor"
        else (args.critic_num_nodes, args.critic_num_gpus_per_node)
    )
    total_gpus = num_nodes * num_gpus_per_node
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1


def compute_actor_args(args):
    [actor_config] = [config for config in resolve_megatron_config(args).trainers if config.role == ACTOR_ROLE]
    return compute_trainer_args(args, actor_config)


def compute_critic_args(args):
    [critic_config] = [config for config in resolve_megatron_config(args).trainers if config.role == CRITIC_ROLE]
    critic_args = compute_trainer_args(args, critic_config)
    critic_args.kl_coef = 0
    critic_args.use_opd = False
    critic_args.disable_param_buffers_cpu_backup = False
    return critic_args


def _compute_spec_trainer(
    args,
    *,
    role: str,
    num_nodes: int,
    num_gpus_per_node: int,
) -> ServeWorkerSpec:
    total_gpus = num_nodes * num_gpus_per_node
    num_cells = compute_trainer_num_cells(args, role=role)
    assert total_gpus % num_cells == 0, f"{total_gpus=} must be divisible by {num_cells=}"
    gpus_per_cell = total_gpus // num_cells

    fp8_scales = (
        x
        if (x := os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES")) is not None
        else default_fp8_block_scaling_fp32_scales()
    )

    return ServeWorkerSpec(
        name=compute_trainer_pool_id(role),
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name=MASTER_PORT_NAME, static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda ctx: compute_trainer_env_vars(args, ctx, fp8_scales=fp8_scales),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_cpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
            pg_name="actor",
        ),
        worker_class=_TRAINER_ACTOR_CLASSES[args.train_backend],
        ctor_kwargs=lambda ctx: dict(
            args=args,
            world_size=gpus_per_cell,
            rank=ctx.worker_in_cell_index,
            role=role,
            cell_index=ctx.cell_index,
        ),
        concurrency_groups=TRAINER_CONCURRENCY_GROUPS if args.use_fault_tolerance else None,
        method_concurrency_groups=TRAINER_METHOD_CONCURRENCY_GROUPS if args.use_fault_tolerance else None,
        meta=lambda ctx: dict(role=role, cell_index=ctx.cell_index),
    )


def compute_trainer_env_vars(args, ctx: WorkerLaunchContext, *, fp8_scales: str) -> dict[str, str]:
    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": fp8_scales,
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **args.train_env_vars,
    }

    if source_patcher_config := args.dumper_source_patcher_config_train:
        env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

    if args.offload_train and args.train_backend == "megatron":
        from torch_memory_saver.utils import get_binary_path_from_package

        dynlib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))

        env_vars["LD_PRELOAD"] = dynlib_path
        env_vars["TMS_INIT_ENABLE"] = "1"
        if args.offload_train_target == "disk":
            assert b"TMS_INIT_ENABLE_DISK_BACKUP" in Path(dynlib_path).read_bytes(), (
                f"{dynlib_path} has no disk backend; reinstall torch_memory_saver at the commit "
                f"docker/Dockerfile pins."
            )
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"
            env_vars["TMS_INIT_ENABLE_DISK_BACKUP"] = "1"
            env_vars["TMS_DISK_BACKUP_CHUNK_MB"] = str(args.offload_train_disk_chunk_mb)
            env_vars["TMS_DISK_BACKUP_DIR"] = os.path.join(
                args.offload_train_disk_dir, f"cell{ctx.cell_index}_rank{ctx.worker_in_cell_index}"
            )
        else:
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    return env_vars
