import copy
import os
from pathlib import Path

from miles.ray.train_actor import TRAINER_CONCURRENCY_GROUPS
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.ft_utils.indep_dp import create_tcp_store
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec, WorkerLaunchContext

MASTER_PORT_NAME = "master"

_TRAINER_ACTOR_CLASSES = {
    "megatron": "miles.backends.megatron_utils.actor.MegatronTrainRayActor",
    "fsdp": "miles.backends.experimental.fsdp_utils.actor.FSDPTrainRayActor",
}

_NUM_GPUS_PER_TRAINER_WORKER = 0.4

_indep_dp_stores: list = []


def specs_trainer(args) -> list[ServeWorkerSpec]:
    specs = [
        _compute_spec_trainer(
            args,
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
    return f"trainer-{role}"


def compute_trainer_num_cells(args, *, role: str) -> int:
    num_nodes, num_gpus_per_node = (
        (args.actor_num_nodes, args.actor_num_gpus_per_node)
        if role == "actor"
        else (args.critic_num_nodes, args.critic_num_gpus_per_node)
    )
    total_gpus = num_nodes * num_gpus_per_node
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1


def compute_critic_args(args):
    critic_args = copy.deepcopy(args)
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

    indep_dp_store_addr = _create_indep_dp_store_addr() if num_cells > 1 else None

    return ServeWorkerSpec(
        name=compute_trainer_pool_id(role),
        port_infos=[PortInfo(name=MASTER_PORT_NAME, static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda ctx: compute_trainer_env_vars(args, ctx),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_cpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_gpu_slots_per_worker=1,
            pg_name="actor",
        ),
        worker_class=_TRAINER_ACTOR_CLASSES[args.train_backend],
        ctor_kwargs=lambda ctx: dict(
            args=args,
            world_size=gpus_per_cell,
            rank=ctx.worker_in_cell_index,
            indep_dp_store_addr=indep_dp_store_addr,
            role=role,
            cell_index=ctx.cell_index,
        ),
        concurrency_groups=TRAINER_CONCURRENCY_GROUPS,
        meta=lambda ctx: dict(role=role, cell_index=ctx.cell_index),
    )


def compute_trainer_env_vars(args, ctx: WorkerLaunchContext) -> dict[str, str]:
    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES", default_fp8_block_scaling_fp32_scales()
        ),
        # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
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


def _create_indep_dp_store_addr() -> str:
    store, addr = create_tcp_store()
    _indep_dp_stores.append(store)
    return addr
