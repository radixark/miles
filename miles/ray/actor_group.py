import asyncio
import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy



class RayTrainGroup:
    """
    A group of ray actors

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        *,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        # Allocate the GPUs for actors w/o instantiating them
        self._actor_handles = self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
            "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }
        if self.args.update_weight_transfer_mode == "rdt":
            # RDT/NIXL: let Ray set CUDA_VISIBLE_DEVICES per training actor so each
            # NCCL rank sees only its assigned GPU. NIXL still works because it uses
            # CUDA driver APIs for physical GPU IDs.
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "0"

        if source_patcher_config := self.args.dumper_source_patcher_config_train:
            env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        backend = self.args.train_backend
        if backend == "megatron":
            from miles.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from miles.backends.experimental.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        runtime_env = {"env_vars": env_vars}
        remote_kwargs = {"num_gpus": 1, "runtime_env": runtime_env}
        if getattr(self.args, "use_rdt_weight_sync", False):
            # RDT/NIXL: update_weights() blocks this actor in ray.get() while it
            # awaits each engine rank's pull_weights, and Ray's tensor-transport
            # threads concurrently serve the NIXL reads of the
            # ray.put(_tensor_transport="nixl") objects owned by this same actor.
            # Raise the actor's concurrency above 1 so the blocking update_weights
            # call does not starve those concurrent serve operations (one per
            # engine rank this source feeds).
            rdt_tp_size = getattr(self.args, "rollout_num_gpus_per_engine", 1)
            remote_kwargs["max_concurrency"] = 1 + rdt_tp_size
            # Each SchedulerActor that maps this trainer's IPC bucket holds a
            # ~520 MiB CUDA context on the trainer GPU; expandable segments
            # reclaim allocator fragmentation so tightly-packed trainers (e.g.
            # the multi_engine validate case) keep enough headroom.
            env_vars.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            # NOTE: do NOT unmask CUDA_VISIBLE_DEVICES here (tried; with 8 GPUs
            # visible, helper threads default to cuda:0 and NCCL group re-init
            # in reloadable_process_group fails with "Duplicate GPU detected").
            #
            # LIBFABRIC-vs-UCX backend probe must validate registration at the
            # size we actually register (the staging bucket), not a token size:
            # EFA accepts small CUDA MRs via its host bounce pool even when
            # GPUDirect is broken, then fails on the real bucket.
            env_vars.setdefault(
                "MILES_RDT_NIXL_VALIDATE_BYTES", str(self.args.update_weight_buffer_size)
            )
        TrainRayActor = ray.remote(**remote_kwargs)(actor_impl)

        # Create worker actors
        actor_handles = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            actor_handles.append(actor)

        return actor_handles

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimizer, local ckpt, etc.
        """
        return await self._broadcast(
            "init", self.args, self.role, with_ref=self.with_ref, with_opd_teacher=self.with_opd_teacher
        )

    async def train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        await self._broadcast("train", rollout_id, rollout_data_ref)

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            await self.rollout_manager.recover_updatable_engines.remote()

        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()

        await self._broadcast("update_weights", info=info)

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def connect(self, critic_group):
        refs = [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=False)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        await self._broadcast("set_rollout_manager", rollout_manager)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
