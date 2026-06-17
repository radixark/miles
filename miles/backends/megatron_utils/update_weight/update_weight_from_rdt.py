"""
Update SchedulerActor engines via RDT/NIXL for p2p weight transfer.

Trainer all-gathers params (bucketed) and converts them to HF format using
``DistBucketedWeightUpdateMixin``. Instead of holding
a full GPU replica of the rollout model, each engine rank is backed by a small,
reusable fixed-size GPU bucket. For every flush the bucket is re-staged to the
shapes of the ready params, ``model_replica.load_weights`` writes the
TP-rank-correct sglang shard into the bucket views, and the views are exported
via ``ray.put(..., _tensor_transport="nixl")``. Each rollout ``SchedulerActor``
then pulls its shard via RDT directly into its own ``param.data`` buffers.

Lifecycle (inherited from the mixin):
    1. Pause engines, flush cache, pre-process quantization
    2. Bucketed TP/EP all-gather -> HF convert -> per-bucket transfer
    3. Resume engines, post-process quantization + post_load_weights
"""

from __future__ import annotations

import logging
import os
import time
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

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
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group

from .update_weight_from_distributed.mixin import DistBucketedWeightUpdateMixin
from .update_weight_from_distributed.p2p_transfer_utils import (
    RemoteTransferPlan,
    create_server_args_from_dict,
)

logger = logging.getLogger(__name__)


class _EngineRankBucket:
    """Per-engine-rank transfer context.

    Holds a GPU model replica whose ``param.data`` is re-pointed into a fixed-size
    GPU bucket each flush, the bucket itself (allocated once), the recorded param
    specs needed to carve correctly-shaped views, and the scheduler actor handles
    that pull this engine rank's shard.
    """

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
        """Re-point ``params_dict[name].data`` to a view inside the fixed bucket.

        Lays the ready params out contiguously (FlattenedTensorBucket style) and
        returns the staged views in the same order as ``names``.
        """
        offset = 0
        views: list[torch.Tensor] = []
        capacity = self.gpu_bucket.numel()
        for name in names:
            shape, dtype, nbytes = self.param_specs[name]
            itemsize = torch.empty((), dtype=dtype).element_size()
            offset = ((offset + itemsize - 1) // itemsize) * itemsize
            assert offset + nbytes <= capacity, (
                f"[RDT] Bucket overflow while staging '{name}': "
                f"need {offset + nbytes} bytes but bucket is {capacity} bytes. "
                f"Increase --update-weight-buffer-size."
            )
            view = self.gpu_bucket[offset : offset + nbytes].view(dtype).reshape(shape)
            self.params_dict[name].data = view
            views.append(view)
            offset += nbytes
        return views


class UpdateWeightFromRDT(DistBucketedWeightUpdateMixin):
    """RDT/NIXL weight transfer built on the P2P bucketed all-gather + HF conversion.

    All training GPUs sharing the same PP rank hold a complete weight replica
    after TP/EP all-gather; each source rank transfers to its planned rollout
    ranks. For every HF bucket flushed by the mixin we:
        stage fixed GPU bucket -> load_weights (HF -> sglang shard)
        -> ray.put(views, nixl) -> actor.pull_weights -> ray.get
    using a separate bucket per engine rank so concurrent pulls never clobber.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self.transfer_plan = RemoteTransferPlan(args, model)
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._group_name = "miles-rdt"

        # Staging bookkeeping shared with the P2P readiness logic.
        self._staged_tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._tensor_update_pending: dict[str, int] = {}
        self._shared_params_dict: dict[str, torch.nn.Parameter] = {}
        self._shared_param_mapper: ParameterMapper | None = None

        # One entry per engine rank this source is responsible for.
        self._transfer_meta_list: list[_EngineRankBucket] = []
        self._scheduler_actors_cache: dict[int, list[ActorHandle]] = {}
        # Per-sync wall-clock breakdown of the RDT-specific transfer phases,
        # logged and reset in _finalize_and_resume_engines.
        self._phase_seconds: dict[str, float] = defaultdict(float)

    @property
    def _is_source(self) -> bool:
        """Whether this training rank sends weights to rollout.

        Mirrors the P2P path: all training GPUs sharing the same PP rank hold a
        complete weight replica after TP/EP all-gather. Only the first
        ``_rollout_num_gpus`` ranks (by gathered_dp_rank) are sources; the rest
        are idle during transfer.
        """
        return self.transfer_plan._gathered_dp_rank < self.transfer_plan._rollout_num_gpus

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None = None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """Plan transfers, build a GPU replica + fixed bucket per engine rank.

        ``engine_gpu_counts/offsets`` are accepted for caller-interface parity
        with the NCCL broadcast path but unused here -- NIXL routes via
        SchedulerActor handles resolved from the transfer plan.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._staged_tensors.clear()
        self._tensor_update_pending.clear()
        self._shared_params_dict = {}
        self._shared_param_mapper = None
        self._transfer_meta_list.clear()
        self._scheduler_actors_cache.clear()

        if not self._is_source:
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

            # CRITICAL: the bucket must NOT live in an expandable (VMM /
            # cuMemCreate) segment. VMM memory cannot export legacy CUDA-IPC
            # handles, so UCX's cuda_ipc lane silently drops for it and NIXL
            # falls back to software-emulated RMA over TCP at ~0.3 GB/s (vs
            # ~150 GB/s NVLink) — bisected on 2026-06-10:
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the source
            # alone reproduces the slowdown. Allocate the bucket from a
            # non-expandable segment.
            expandable = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if expandable:
                torch.cuda.memory._set_allocator_settings("expandable_segments:False")
            try:
                gpu_bucket = torch.empty(
                    self.args.update_weight_buffer_size,
                    dtype=torch.uint8,
                    device=torch.cuda.current_device(),
                )
            finally:
                if expandable:
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            # Pin the bucket's NIXL registration for the process lifetime.
            # Otherwise every per-flush ray.put registers the full bucket and
            # the ref drop afterwards deregisters it AND bumps the NIXL agent
            # meta version, which invalidates the engines' cached remote agent
            # and forces a re-handshake on the next flush.
            from ray.experimental import register_nixl_memory

            register_nixl_memory(gpu_bucket)

            actors = []
            for t in rank_targets:
                engine_actors = self._get_engine_scheduler_actors(rollout_engines, t.engine_ind)
                actors.append(engine_actors[t.engine_rank])

            self._transfer_meta_list.append(
                _EngineRankBucket(model_replica, params_dict, param_specs, gpu_bucket, actors)
            )

    def _get_engine_scheduler_actors(
        self, rollout_engines: Sequence[ActorHandle], engine_ind: int
    ) -> list[ActorHandle]:
        if engine_ind not in self._scheduler_actors_cache:
            self._scheduler_actors_cache[engine_ind] = ray.get(
                rollout_engines[engine_ind].get_scheduler_actors.remote()
            )
        return self._scheduler_actors_cache[engine_ind]

    def create_gpu_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args: ServerArgs,
    ) -> tuple[torch.nn.Module, dict[str, torch.nn.Parameter], dict[str, tuple[torch.Size, torch.dtype, int]]]:
        """Create a GPU model replica that loads the right shard and skips post_load_weights.

        The dummy load allocates the full model on GPU momentarily; we record each
        param's (shape, dtype, nbytes) and immediately free the storage so the
        replica only keeps nn.Parameter/weight_loader metadata. The actual weight
        bytes live in a reusable fixed-size bucket re-pointed during staging.
        """
        load_config = LoadConfig(
            load_format="dummy",
            model_loader_extra_config=None,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args_module._global_server_args = server_args
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)

        # Monkey-patch the loader-level post_load_weights to no-op BEFORE get_model,
        # because get_model() calls post_load_weights() internally which may invoke
        # kernels that should only run on the rollout engine after RDMA transfer
        # (post_process_weights(post_load_weights=True)).
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
            # Release the full-model dummy allocation; storage is re-pointed into
            # the fixed-size bucket during staging.
            param.data = torch.empty(0, dtype=param.data.dtype, device=param.data.device)

        return model, params_dict, param_specs

    def _get_transfer_ready_params(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[str], list[tuple[str, torch.Tensor]]]:
        """Determine which sglang params have all shards present, returning their accumulated tensors.

        Some parameters are trained separately on the training side but fused into a
        single tensor on the rollout side (e.g., Q/K/V projections are separate in
        Megatron but merged into one qkv_proj in sglang). This function stages
        incoming HF tensors in self._staged_tensors until all shards for a
        sglang param are collected. Only returns tensors for fully-ready params,
        preventing partial load_weights() calls.

        Return:
            transfer_ready_params: tensors' names for the ones ready to be transferred.
            ready_hf_tensors: corresponding complete tensors ready to be transferred.
        """
        transfer_ready_params = []
        params_dict = self._shared_params_dict

        for name, tensor in converted_named_tensors:
            # map the tensor name of huggingface to the one of sglang.
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

        ready_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for param_name in transfer_ready_params:
            staged = self._staged_tensors.pop(param_name, [])
            ready_hf_tensors.extend(staged)
            self._tensor_update_pending.pop(param_name, None)

        return transfer_ready_params, ready_hf_tensors

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Stage incoming tensors; when all shards for a param are collected, load
        into each engine rank's bucket and pull via RDT.

        Each engine rank uses its own fixed bucket so all pulls can be in flight
        concurrently; we wait for them at the end of the flush before the buckets
        are restaged on the next call.
        """
        if not self._is_source or not converted_named_tensors:
            return

        transfer_ready_params, ready_hf_tensors = self._get_transfer_ready_params(converted_named_tensors)

        if transfer_ready_params and ready_hf_tensors:
            phase = self._phase_seconds
            weight_refs = []
            futures = []
            for meta in self._transfer_meta_list:
                t = time.perf_counter()
                meta.stage(transfer_ready_params)
                phase["stage"] += time.perf_counter() - t

                t = time.perf_counter()
                meta.model_replica.load_weights(ready_hf_tensors)
                # Synchronize so async GPU copy time is attributed to "load",
                # not to the cuda.synchronize inside ray.put.
                torch.cuda.synchronize()
                phase["load"] += time.perf_counter() - t

                # Re-read post-load in case a weight loader reassigned param.data.
                tensor_views = [meta.params_dict[name].data for name in transfer_ready_params]

                t = time.perf_counter()
                weights_ref = ray.put(tensor_views, _tensor_transport="nixl")
                phase["put"] += time.perf_counter() - t
                weight_refs.append(weights_ref)
                t = time.perf_counter()
                for actor in meta.actors:
                    futures.append(actor.pull_weights.remote([weights_ref], transfer_ready_params))
                phase["submit"] += time.perf_counter() - t

            t = time.perf_counter()
            ray.get(futures)
            phase["pull_wait"] += time.perf_counter() - t
            phase["bytes"] += sum(v.numel() * v.element_size() for v in tensor_views) * len(self._transfer_meta_list)
            phase["flushes"] += 1
            del weight_refs
            if pbar is not None:
                pbar.update(len(transfer_ready_params))

        converted_named_tensors.clear()

    def _gather_and_update_expert_weights(self, update_bucket_weight_func, pbar=None):
        """Assert all staged shards were transferred (transfers are awaited inline)."""
        super()._gather_and_update_expert_weights(update_bucket_weight_func, pbar)
        if not self._is_source:
            return
        assert len(self._tensor_update_pending) == 0 and len(self._staged_tensors) == 0, (
            f"Some tensors were not transferred during RDT weight update. "
            f"Pending: {self._tensor_update_pending}, Staged: {self._staged_tensors}"
        )

    def _finalize_and_resume_engines(self):
        if self._is_source and self._phase_seconds:
            phase = dict(self._phase_seconds)
            gib = phase.pop("bytes", 0) / 1024**3
            flushes = int(phase.pop("flushes", 0))
            breakdown = ", ".join(f"{k}={v:.2f}s" for k, v in sorted(phase.items()))
            logger.info(f"[RDT] sync phase breakdown ({flushes} flushes, {gib:.2f} GiB staged): {breakdown}")
            self._phase_seconds.clear()
        # The `update_weight_version` here is necessary because the engine was not
        # aware that the RDMA pull happened. After transfer, some models (e.g.
        # Deepseek-arch) on the rollout side must invoke `post_load_weights` to
        # regenerate params not registered as `model.named_parameters()`.
        if dist.get_rank() == 0:
            ray.get(
                [
                    engine.update_weight_version.remote(weight_version=str(self.weight_version))
                    for engine in self.rollout_engines
                ]
            )
        super()._finalize_and_resume_engines(post_load_weights=True)
