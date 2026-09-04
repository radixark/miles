import os
from pathlib import Path

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus


def allocate_gpus_for_actor(
    args,
    gpus_per_cell: int,
    pg: tuple[PlacementGroup, list[int], list[int]],
    num_gpus_per_actor: float,
    indep_dp_store_addr: str,
    role: str,
    cell_index: int,
):
    world_size = gpus_per_cell

    # Use placement group to lock resources for models of same type
    assert pg is not None
    pg, reordered_bundle_indices, _reordered_gpu_ids = pg

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
        else:
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    backend = args.train_backend
    if backend == "megatron":
        from miles.backends.megatron_utils.actor import MegatronTrainRayActor

        actor_impl = MegatronTrainRayActor

    else:
        from miles.backends.fsdp_utils import FSDPTrainRayActor

        actor_impl = FSDPTrainRayActor

    ft = args.use_fault_tolerance
    TrainRayActor = ray.remote(
        num_gpus=1,
        runtime_env={"env_vars": env_vars},
        **(dict(concurrency_groups={"heartbeat_status": 1, "default": 1, "fault_injector": 1}) if ft else {}),
    )(_with_ft_concurrency_groups(actor_impl) if ft else actor_impl)

    # Create worker actors
    actor_handles = []
    master_addr, master_port = None, None
    for rank in range(world_size):
        options = dict(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[rank],
            ),
        )
        disk_env = _disk_scope_env_vars(args, role=role, cell_index=cell_index, rank=rank)
        if disk_env:
            options["runtime_env"] = {"env_vars": {**env_vars, **disk_env}}
        actor = TrainRayActor.options(**options).remote(
            args=args,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
            indep_dp_store_addr=indep_dp_store_addr,
            role=role,
            cell_index=cell_index,
        )
        if rank == 0:
            master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
        actor_handles.append(actor)

    return actor_handles


def _disk_scope_env_vars(args, *, role: str, cell_index: int, rank: int) -> dict[str, str]:
    """Per-process directory scope for everything a training rank spills to local disk.

    Actor and critic ranks of a shared-GPU PPO job land on the same GPU and both
    start their torch.distributed rank at 0, so neither the rank nor the cell can
    tell their directories apart; the process identity can.
    """
    if args.train_backend != "megatron":
        return {}
    disk_offload = args.offload_train and args.offload_train_target == "disk"
    if not disk_offload and not args.stream_optimizer_state_to_disk:
        return {}
    scope = TrainProcessIdentity(component=role, cell_index=cell_index, rank_within_cell=rank).to_name()
    env = {"MILES_TRAIN_DISK_SCOPE": scope}
    if disk_offload:
        env["TMS_DISK_BACKUP_DIR"] = os.path.join(args.offload_train_disk_dir, scope)
    return env


def _with_ft_concurrency_groups(actor_impl: type) -> type:
    class _FtTrainRayActor(actor_impl):
        @ray.method(concurrency_group="heartbeat_status")
        def get_heartbeat_status(self) -> HeartbeatStatus:
            return super().get_heartbeat_status()

        @ray.method(concurrency_group="fault_injector")
        def inject_fault(self, mode: str) -> None:
            super().inject_fault(mode)

    _FtTrainRayActor.__name__ = actor_impl.__name__
    _FtTrainRayActor.__qualname__ = actor_impl.__qualname__
    return _FtTrainRayActor
