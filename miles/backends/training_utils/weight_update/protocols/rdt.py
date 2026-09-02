"""Update SchedulerActor engines via RDT/NIXL weight transfer.

Rides the updater's bucketed TP/EP all-gather and HF conversion. Instead of a full GPU replica of the rollout model, each engine rank
is backed by a small reusable GPU bucket: per flush the bucket is re-staged to the
ready params' shapes, ``model_replica.load_weights`` writes the TP-rank-correct
sglang shard into the views, and each ``SchedulerActor`` pulls its shard from
``ray.put(..., _tensor_transport="nixl")`` straight into its ``param.data``.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.server_args import ServerArgs

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.utils.distributed_utils import get_gloo_group

from .p2p_transfer_utils import RemoteTransferPlan, create_server_args_from_dict

logger = logging.getLogger(__name__)


def _staging_span(offset: int, dtype: torch.dtype, nbytes: int) -> tuple[int, int]:
    """(aligned start, end) for placing nbytes of dtype at offset in the uint8 bucket."""
    start = (offset + dtype.itemsize - 1) // dtype.itemsize * dtype.itemsize
    return start, start + nbytes


class _EngineRankBucket:
    """Transfer context for one engine rank: model replica, its fixed-size GPU
    bucket, the param specs used to carve views, and the actors that pull it."""

    def __init__(
        self,
        model_replica: torch.nn.Module,
        params_dict: dict[str, torch.nn.Parameter],
        param_specs: dict[str, tuple[torch.Size, torch.dtype, int]],
        gpu_bucket: torch.Tensor,
        actors: list[ActorHandle],
    ) -> None:
        self.model_replica = model_replica
        self.params_dict = params_dict
        self.param_specs = param_specs
        self.gpu_bucket = gpu_bucket
        self.actors = actors

    def stage(self, names: list[str]) -> list[torch.Tensor]:
        """Re-point each ``params_dict[name].data`` to a contiguous view inside the
        fixed bucket, returning the views in ``names`` order."""
        offset = 0
        views: list[torch.Tensor] = []
        capacity = self.gpu_bucket.numel()
        for name in names:
            shape, dtype, nbytes = self.param_specs[name]
            offset, end = _staging_span(offset, dtype, nbytes)
            assert end <= capacity, (
                f"[RDT] Bucket overflow while staging '{name}': "
                f"need {end} bytes but bucket is {capacity} bytes. "
                f"Increase --update-weight-buffer-size."
            )
            view = self.gpu_bucket[offset:end].view(dtype).reshape(shape)
            self.params_dict[name].data = view
            views.append(view)
            offset = end
        return views


class UpdateWeightFromRDT(WeightTransferProtocol):
    """RDT/NIXL weight transfer on the P2P bucketed all-gather + HF conversion.

    Per HF bucket streamed by the updater: stage GPU bucket -> load_weights
    -> ray.put(views, nixl) -> actor.pull_weights -> ray.get. Ready params that
    overflow the fixed bucket are split into sequential NIXL rounds. One bucket
    per engine rank, so concurrent pulls never clobber each other.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.transfer_plan = RemoteTransferPlan(args)
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._group_name = "miles-rdt"

        self._staged_tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._tensor_update_pending: dict[str, int] = {}
        self._shared_params_dict: dict[str, torch.nn.Parameter] = {}
        self._shared_param_mapper: ParameterMapper | None = None

        # One entry per engine rank this source is responsible for.
        self._engine_rank_buckets: list[_EngineRankBucket] = []
        self._scheduler_actors_cache: dict[int, list[ActorHandle]] = {}

    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Plan transfers, build a GPU replica + fixed bucket per engine rank.

        ``engine_gpu_counts`` must be uniform at ``args.rollout_num_gpus_per_engine``.
        The lock and offsets exist for parity with the broadcast path; one-sided
        RDMA pulls have no collective to deadlock, so they are unused.
        """
        if engine_gpu_counts is not None and any(
            count != self.args.rollout_num_gpus_per_engine for count in engine_gpu_counts
        ):
            raise ValueError(
                f"[RDT] Heterogeneous engine GPU counts {list(engine_gpu_counts)} are not supported; "
                f"every engine must use --rollout-num-gpus-per-engine "
                f"({self.args.rollout_num_gpus_per_engine})."
            )

        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self.is_sender = self.transfer_plan._gathered_dp_rank < self.transfer_plan._rollout_num_gpus
        self._staged_tensors.clear()
        self._tensor_update_pending.clear()
        self._shared_params_dict = {}
        self._shared_param_mapper = None
        self._engine_rank_buckets.clear()
        self._scheduler_actors_cache.clear()

        if not self.is_sender:
            return

        self._group_name = f"miles-rdt_{self.transfer_plan._gathered_dp_rank}"
        targets = self.transfer_plan.plan_p2p()

        # Same engine_rank => same TP shard => same parallelism config + shapes.
        targets_grouped_by_engine_rank: dict[int, list] = {}
        for target in targets:
            targets_grouped_by_engine_rank.setdefault(target.engine_rank, []).append(target)

        first_engine_rank = True
        for engine_rank, rank_targets in targets_grouped_by_engine_rank.items():
            first_target = rank_targets[0]
            parallelism_info = ray.get(
                rollout_engines[first_target.engine_ind].get_parallelism_info.remote(rank=engine_rank)
            )
            server_info = ray.get(rollout_engines[first_target.engine_ind].get_server_info.remote())
            parallelism_config = RankParallelismConfig.from_dict(parallelism_info)
            server_args = create_server_args_from_dict(server_info)

            model_replica, params_dict, param_specs = self.create_gpu_replica(
                parallelism_config, self.args.hf_checkpoint, server_args
            )
            if first_engine_rank:
                self._shared_params_dict = params_dict
                self._shared_param_mapper = ParameterMapper.from_model(model_replica)
                first_engine_rank = False

            # CRITICAL: expandable (VMM) memory cannot export CUDA-IPC handles, so
            # UCX silently drops its cuda_ipc lane and NIXL falls back to emulated
            # RMA over TCP (~0.3 GB/s vs ~150 GB/s). Force the bucket off it.
            expandable = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if expandable:
                torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
            try:
                max_param_nbytes = max((spec[2] for spec in param_specs.values()), default=0)
                if max_param_nbytes > self.args.update_weight_buffer_size:
                    logger.warning(
                        f"[RDT] Largest destination parameter needs {max_param_nbytes} bytes, above "
                        f"--update-weight-buffer-size ({self.args.update_weight_buffer_size}); "
                        f"growing bucket to fit."
                    )
                gpu_bucket = torch.empty(
                    max(self.args.update_weight_buffer_size, max_param_nbytes),
                    dtype=torch.uint8,
                    device=torch.cuda.current_device(),
                )
            finally:
                if expandable:
                    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
            # Pin for the process lifetime: otherwise each ray.put re-registers the
            # bucket and the ref drop bumps the NIXL agent meta version, forcing the
            # engines to re-handshake every flush.
            from ray.experimental import register_nixl_memory

            register_nixl_memory(gpu_bucket)

            actors = []
            for t in rank_targets:
                engine_actors = self._get_engine_scheduler_actors(rollout_engines, t.engine_ind)
                actors.append(engine_actors[t.engine_rank])

            self._engine_rank_buckets.append(
                _EngineRankBucket(model_replica, params_dict, param_specs, gpu_bucket, actors)
            )

    def _get_engine_scheduler_actors(
        self, rollout_engines: Sequence[ActorHandle], engine_ind: int
    ) -> list[ActorHandle]:
        if engine_ind not in self._scheduler_actors_cache:
            actors = ray.get(rollout_engines[engine_ind].get_scheduler_actors.remote())
            # Keep destination registrations alive across repeated RDT pulls.
            ray.get([actor.register_weight_for_rdt.remote() for actor in actors])
            self._scheduler_actors_cache[engine_ind] = actors
        return self._scheduler_actors_cache[engine_ind]

    def create_gpu_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args: ServerArgs,
    ) -> tuple[torch.nn.Module, dict[str, torch.nn.Parameter], dict[str, tuple[torch.Size, torch.dtype, int]]]:
        """Create a GPU model replica that loads the right shard and skips post_load_weights.

        The dummy load allocates the full model momentarily; we record each param's
        (shape, dtype, nbytes) and free the storage, keeping only the weight_loader
        metadata. The bytes live in the fixed bucket re-pointed during staging.
        """
        load_config = LoadConfig(
            load_format="dummy",
            model_loader_extra_config=None,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        # This replica is local even when the rollout deployment spans nodes.
        if server_args.nnodes > 1:
            server_args.override("miles.rdt.local_replica", nnodes=1)
        server_args_module.set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)

        # get_model() calls post_load_weights internally; those kernels belong on the
        # rollout engine after the transfer, where end_weight_update() runs them.
        from sglang.srt.model_loader import loader as model_loader_module

        original_post_load_weights = model_loader_module.post_load_weights
        model_loader_module.post_load_weights = lambda *args, **kwargs: None
        try:
            with ParallelismContext(parallelism_config):
                model = get_model(
                    model_config=ModelConfig(model_path),
                    load_config=load_config,
                    device_config=DeviceConfig(device="cuda"),
                )
        finally:
            model_loader_module.post_load_weights = original_post_load_weights

        # Also patch the instance method for subsequent load_weights() calls.
        if hasattr(model, "post_load_weights"):
            model.post_load_weights = lambda *args, **kwargs: None

        params_dict = dict(model.named_parameters())
        param_specs: dict[str, tuple[torch.Size, torch.dtype, int]] = {}
        for name, param in params_dict.items():
            nbytes = param.data.numel() * param.data.element_size()
            param_specs[name] = (param.data.shape, param.data.dtype, nbytes)
            # Release the dummy allocation; stage() re-points this into the bucket.
            param.data = torch.empty(0, dtype=param.data.dtype, device=param.data.device)

        return model, params_dict, param_specs

    def _get_transfer_ready_params(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Return destination params whose shard set is complete, each mapped to its HF shards.

        Params fused on the rollout side (e.g. Megatron's separate Q/K/V vs sglang's
        qkv_proj) stay in ``self._staged_tensors`` until every shard has arrived, so
        ``load_weights()`` is never called on a partial param.
        """
        transfer_ready_params = []
        params_dict = self._shared_params_dict

        for name, tensor in converted_named_tensors:
            mapped_result = self._shared_param_mapper.map(name)
            mapped, num_shards, num_experts = (
                mapped_result.sglang_name,
                mapped_result.num_shards,
                mapped_result.num_local_experts,
            )
            if mapped not in params_dict:
                logger.warning(f"Parameter {mapped} not found in shared model replica.")
                continue

            if num_experts is not None and num_experts > 0:
                total_expected = num_experts * num_shards
            else:
                total_expected = num_shards

            self._staged_tensors.setdefault(mapped, []).append((name, tensor))

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                if mapped not in self._tensor_update_pending:
                    self._tensor_update_pending[mapped] = total_expected - 1
                else:
                    self._tensor_update_pending[mapped] -= 1
                if self._tensor_update_pending[mapped] == 0:
                    transfer_ready_params.append(mapped)

        ready: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for param_name in transfer_ready_params:
            ready[param_name] = self._staged_tensors.pop(param_name, [])
            self._tensor_update_pending.pop(param_name, None)
        return ready

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Stage incoming tensors; once a param is complete, load it into each engine
        rank's bucket and pull via RDT.

        The GPU bucket is lifetime-registered with NIXL, so an oversized ready set is
        packed into sequential rounds rather than grown. A flush that already fits is
        one round.
        """
        if not self.is_sender or not converted_named_tensors:
            return

        ready = self._get_transfer_ready_params(converted_named_tensors)
        names = list(ready)
        while names and self._engine_rank_buckets:
            # Engines are homogeneous (enforced in connect_rollout_engines), so bucket 0
            # decides the split; stage() still asserts per bucket as the backstop.
            specs = self._engine_rank_buckets[0].param_specs
            capacity = self._engine_rank_buckets[0].gpu_bucket.numel()
            offset = count = 0
            for name in names:
                _, dtype, nbytes = specs[name]
                _, end = _staging_span(offset, dtype, nbytes)
                if count and end > capacity:
                    break
                offset, count = end, count + 1
            chunk, names = names[:count], names[count:]
            hf_chunk = [pair for name in chunk for pair in ready[name]]

            weight_refs = []
            futures = []
            for bucket in self._engine_rank_buckets:
                bucket.stage(chunk)
                bucket.model_replica.load_weights(hf_chunk)
                # The async copies must land before ray.put hands the views to NIXL.
                torch.cuda.synchronize()
                # Re-read post-load in case a weight loader reassigned param.data.
                tensor_views = [bucket.params_dict[name].data for name in chunk]
                weights_ref = ray.put(tensor_views, _tensor_transport="nixl")
                weight_refs.append(weights_ref)
                for actor in bucket.actors:
                    futures.append(actor.pull_weights.remote([weights_ref], chunk))
            ray.get(futures)
            del weight_refs

        converted_named_tensors.clear()

    def after_base_weights(self) -> None:
        """Assert all staged shards were transferred (transfers are awaited inline)."""
        if not self.is_sender:
            return
        assert len(self._tensor_update_pending) == 0 and len(self._staged_tensors) == 0, (
            f"Some tensors were not transferred during RDT weight update. "
            f"Pending: {self._tensor_update_pending}, Staged: {self._staged_tensors}"
        )
