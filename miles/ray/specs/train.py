import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE, CRITIC_ROLE, MegatronTrainerConfig
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.args.runtime import AllConfig, TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import (
    TRAINER_CONTROLLER_POOL_ID_PREFIX,
    compute_cell_id,
    compute_worker_name,
    format_name_index,
)
from miles.utils.workers.types import DeployComponent, DeploymentIdentity, PlatformAccess
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import (
    MASTER_PORT_NAME,
    BaseServeSpec,
    HostAndPort,
    PortInfo,
    SchedulingSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)

TRAINER_CONTROLLER_ADDRS_FLAG = "--trainer-controller-addrs"
POOL_CATEGORY_TRAINER_ENGINE = "trainer_engine"

TRAINER_CONCURRENCY_GROUPS = {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}

TRAINER_CONTROLLER_WORKER_CLASS = "miles.ray.train.group.TrainerController"

_TRAINER_ACTOR_CLASSES = {
    "megatron": "miles.backends.megatron_utils.actor.MegatronTrainRayActor",
    "fsdp": "miles.backends.fsdp_utils.actor.FSDPTrainRayActor",
}

_NUM_GPUS_PER_TRAINER_WORKER = 0.4


@dataclass(kw_only=True)
class TrainerControllerSpec(BaseServeSpec):
    worker_type = "trainer_controller"
    args: TrainerConfig
    deploy_component: DeployComponent = DeployComponent.TRAINER
    platform_access: PlatformAccess = PlatformAccess.READ_DELETE
    worker_class: str = TRAINER_CONTROLLER_WORKER_CLASS

    @property
    def name(self) -> str:
        return compute_trainer_controller_pool_id(self.args.trainer_id)

    @classmethod
    def create(cls, args: Any) -> list[Self]:
        return [cls(args=compute_trainer_config(args, trainer)) for trainer in compute_trainer_configs(args)]

    def scheduling(self) -> SchedulingSpec:
        return SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
        )

    def static_meta(self) -> dict[str, Any]:
        return {}

    def slice_config(self, args: Any) -> TrainerConfig:
        return self.args

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        args = ctx.args
        return dict(
            args=args,
            deployment_identity=DeploymentIdentity(
                run_uuid=args.run_uuid,
                deploy_component=args.deploy_component,
                deploy_instance_id=args.deploy_instance_id,
                trainer_id=args.trainer_id,
            ),
            expected_num_cells=compute_trainer_num_cells(args, role=args.trainer_role),
            trainer_id=args.trainer_id,
            role=args.trainer_role,
            with_ref=(args.trainer_role != CRITIC_ROLE) and (args.kl_coef != 0 or args.use_kl_loss),
            with_opd_teacher=(args.trainer_role != CRITIC_ROLE) and args.use_opd and args.opd_type == "megatron",
            cell_provider=ctx.capability.dynamic_worker_provider(pool_ids=[compute_trainer_pool_id(args.trainer_id)]),
            cell_operations=ctx.capability.cell_operations(),
        )


@dataclass(kw_only=True)
class TrainerSpec(BaseServeSpec):
    worker_type = "trainer"
    args: TrainerConfig
    pg_slot_offset: int = 0
    category: str = POOL_CATEGORY_TRAINER_ENGINE
    deploy_component: DeployComponent = DeployComponent.TRAINER

    @property
    def name(self) -> str:
        return compute_trainer_pool_id(self.args.trainer_id)

    @property
    def worker_class(self) -> str:
        return _TRAINER_ACTOR_CLASSES[self.args.train_backend]

    @property
    def concurrency_groups(self) -> dict[str, int] | None:
        return TRAINER_CONCURRENCY_GROUPS if self.args.use_fault_tolerance else None

    @classmethod
    def create(cls, args: Any) -> list[Self]:
        # TODO: support different sizes after the args refactor
        actor_gpus_per_instance = args.actor_num_nodes * args.actor_num_gpus_per_node
        specs = []
        actor_index = 0
        for trainer in compute_trainer_configs(args):
            pg_slot_offset = 0
            if trainer.role != CRITIC_ROLE:
                pg_slot_offset = actor_index * actor_gpus_per_instance
                actor_index += 1
            specs.append(
                cls(
                    args=compute_trainer_config(args, trainer),
                    pg_slot_offset=pg_slot_offset,
                )
            )
        return specs

    def scheduling(self) -> SchedulingSpec:
        args = self.args
        num_nodes, num_gpus_per_node = (
            (args.critic_num_nodes, args.critic_num_gpus_per_node)
            if args.trainer_role == CRITIC_ROLE
            else (args.actor_num_nodes, args.actor_num_gpus_per_node)
        )
        total_gpus = num_nodes * num_gpus_per_node
        num_cells = compute_trainer_num_cells(args, role=args.trainer_role)
        assert total_gpus % num_cells == 0, f"{total_gpus=} must be divisible by {num_cells=}"
        return SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=total_gpus // num_cells,
            num_gpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_cpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
            pg_name="actor",
            pg_slot_offset=self.pg_slot_offset,
        )

    def port_infos(self) -> list[PortInfo]:
        return [
            PortInfo(name=MASTER_PORT_NAME, static_port=9000, mode="master", allow_dynamic=True),
            *super().port_infos(),
        ]

    def static_meta(self) -> dict[str, Any]:
        return dict(role=self.args.trainer_role, include_cell_index=True)

    def slice_config(self, args: Any) -> TrainerConfig:
        return self.args

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        fp8_scales = (
            x
            if (x := os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES")) is not None
            else default_fp8_block_scaling_fp32_scales()
        )
        return compute_trainer_env_vars(ctx.args, ctx, fp8_scales=fp8_scales)

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return dict(
            args=ctx.args,
            world_size=_compute_trainer_world_size(ctx.args),
            rank=ctx.worker_in_cell_index,
            role=ctx.args.trainer_role,
            cell_index=ctx.cell_index,
        )


def specs_trainer_controller(args: AllConfig) -> list[TrainerControllerSpec]:
    return TrainerControllerSpec.create(args)


def compute_trainer_configs(args: AllConfig) -> list[MegatronTrainerConfig]:
    return args.raw_megatron.trainers


def external_trainer_controller_addrs(args, *, trainer_ids: list[str]) -> dict[str, HostAndPort] | None:
    if (entries := args.trainer_controller_addrs) is None:
        return None
    addrs = {prefix: parse_host_and_port(rest) for prefix, _, rest in (entry.partition("=") for entry in entries)}
    assert sorted(addrs) == sorted(trainer_ids) and len(addrs) == len(entries), (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} must name each of {trainer_ids} exactly once as "
        f"'<trainer_id>=<host:port>' (got {entries})"
    )
    return addrs


def compute_trainer_ids(args) -> list[str]:
    return [config.trainer_id for config in compute_trainer_configs(args)]


def create_trainer_controller_handle(args, *, capability: BackendCapability, trainer_id: str) -> BaseWorkerHandle:
    provider = _compute_trainer_controller_provider(args, capability=capability, trainer_id=trainer_id)
    return provider.get_handle(trainer_controller_worker_name(trainer_id))


def _compute_trainer_controller_provider(
    args, *, capability: BackendCapability, trainer_id: str
) -> BaseWorkerProvider:
    pool_id = compute_trainer_controller_pool_id(trainer_id)
    if (addrs := external_trainer_controller_addrs(args, trainer_ids=compute_trainer_ids(args))) is None:
        return capability.static_worker_provider(pool_id=pool_id)
    return StaticWorkerProvider.of_rpc_addrs(
        pool_id=pool_id, addrs=[addrs[trainer_id]], worker_class=TRAINER_CONTROLLER_WORKER_CLASS
    )


def compute_trainer_controller_pool_id(trainer_id: str) -> str:
    return f"{TRAINER_CONTROLLER_POOL_ID_PREFIX}{trainer_id}"


def trainer_controller_worker_name(trainer_id: str) -> str:
    return compute_worker_name(pool_id=compute_trainer_controller_pool_id(trainer_id))


def trainer_controller_cell_id(trainer_id: str) -> str:
    return compute_cell_id(pool_id=compute_trainer_controller_pool_id(trainer_id), cell_index=0)


def specs_trainer(args: AllConfig) -> list[TrainerSpec]:
    return TrainerSpec.create(args)


def compute_trainer_pool_id(trainer_id: str) -> str:
    return f"trainer-engine-{trainer_id}"


def compute_trainer_num_cells(args, *, role: str) -> int:
    num_nodes, num_gpus_per_node = (
        (args.actor_num_nodes, args.actor_num_gpus_per_node)
        if role == ACTOR_ROLE
        else (args.critic_num_nodes, args.critic_num_gpus_per_node)
    )
    total_gpus = num_nodes * num_gpus_per_node
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1


def _compute_trainer_world_size(args: TrainerConfig) -> int:
    total_gpus = (
        args.critic_num_nodes * args.critic_num_gpus_per_node
        if args.trainer_role == CRITIC_ROLE
        else args.actor_num_nodes * args.actor_num_gpus_per_node
    )
    num_cells = compute_trainer_num_cells(args, role=args.trainer_role)
    assert total_gpus % num_cells == 0, f"{total_gpus=} must be divisible by {num_cells=}"
    return total_gpus // num_cells


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
                args.offload_train_disk_dir,
                f"cell{format_name_index(ctx.cell_index)}_rank{format_name_index(ctx.worker_in_cell_index)}",
            )
        else:
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    return env_vars
