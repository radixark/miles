import logging
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, build_lora_sync_config, is_lora_weight_name
from miles.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict->GPU) -> broadcast PP/EP(GPU NCCL) -> gather TP(GPU NCCL) -> convert HF(GPU) -> send.
    Colocated: GPU->CPU serialize -> gather_object(Gloo CPU) -> Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
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
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.is_lora = is_lora
        self._lora_loaded = False

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=self.is_lora,
        )

        self._lora_config = build_lora_sync_config(args) if self.is_lora else None
        # Create IPC gather groups within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        _t = _WeightUpdateTimer(enabled=(rank == 0))

        _t.start("pause_flush")
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())
        _t.stop("pause_flush")

        _t.start("weights_getter")
        megatron_local_weights = self.weights_getter()
        _t.stop("weights_getter")

        sync_chunk_count = 0
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            _t.start(f"send_chunk_{sync_chunk_count}")
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            results = ray.get(refs)
            _check_weight_sync_results(results, is_lora=self.is_lora)
            del long_lived_tensors
            _t.stop(f"send_chunk_{sync_chunk_count}")
            sync_chunk_count += 1

        if self.is_lora and sync_chunk_count == 0:
            raise RuntimeError(
                "LoRA weight sync failed: the weight iterator produced zero chunks. "
                "No adapter weights were sent to the rollout engine. This usually means "
                "the Megatron-Bridge or SGLang version is incompatible."
            )

        _t.start("final_barrier")
        dist.barrier(group=get_gloo_group())
        _t.stop("final_barrier")

        _t.start("post_process")
        # int4/fp4 post_process, mxfp8 post-process (swizzle MoE scales).
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors",
                "mxfp8",
            ]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        _t.stop("post_process")

        _t.report()

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []
        long_lived_tensors = []

        # Separate LoRA weights from base weights
        if self.is_lora:
            weight_tensors = [(n, t) for n, t in hf_named_tensors if is_lora_weight_name(n)]
            if not weight_tensors:
                raise RuntimeError(
                    "LoRA weight sync failed: no LoRA weights (lora_A/lora_B) found in the "
                    "HF weight chunk produced by the weight iterator. This usually means the "
                    "Megatron-Bridge or SGLang version is incompatible and adapter weights were "
                    "not exported. Check that `megatron_to_hf_mode` and bridge version match."
                )
        else:
            weight_tensors = hf_named_tensors

        kwargs = dict(
            hf_named_tensors=weight_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
        )
        if self.is_lora:
            kwargs |= dict(
                lora_config=self._lora_config,
                lora_name=LORA_ADAPTER_NAME,
                lora_loaded=self._lora_loaded,
            )
        else:
            kwargs |= dict(
                weight_version=self.weight_version,
            )

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(**kwargs)
        all_refs.extend(refs_colocated)

        if self.is_lora:
            self._lora_loaded = True

        if self.is_lora and self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                weight_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
    lora_config: dict | None = None,
    lora_name: str | None = None,
    lora_loaded: bool = False,
) -> tuple[list[ObjectRef], Any]:
    is_lora = lora_config is not None
    long_live_tensors = []
    _is_rank0 = dist.get_rank() == 0

    if _is_rank0:
        _t0 = time.monotonic()
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))
    if _is_rank0:
        logger.info(f"[WU send] serialize={time.monotonic()-_t0:.2f}s")

    if _is_rank0:
        _t0 = time.monotonic()
    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )
    if _is_rank0:
        logger.info(f"[WU send] gloo_gather={time.monotonic()-_t0:.2f}s")

    if _is_rank0:
        _t0 = time.monotonic()
    refs = []
    if dist.get_rank() == ipc_gather_src:
        if is_lora:
            if lora_loaded:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))

            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_tensors=serialized_named_tensors[0][0],
                    load_format="flattened_bucket",
                )
            )

        else:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                }
                refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))
    if _is_rank0:
        logger.info(f"[WU send] ray_ipc_submit={time.monotonic()-_t0:.2f}s")

    return refs, long_live_tensors


class _WeightUpdateTimer:
    """Lightweight wall-clock timer for weight update breakdown."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str):
        if self._enabled:
            torch.cuda.synchronize()
            self._starts[name] = time.monotonic()

    def stop(self, name: str):
        if self._enabled and name in self._starts:
            torch.cuda.synchronize()
            self._timings[name] = time.monotonic() - self._starts.pop(name)

    def report(self):
        if not self._enabled:
            return
        total = sum(self._timings.values())
        parts = " | ".join(f"{k}={v:.1f}s" for k, v in self._timings.items())
        logger.info(f"[WeightUpdate Breakdown] total={total:.1f}s | {parts}")


def _check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    """Validate return values from rollout engine weight-sync RPCs.

    Raises RuntimeError if any engine reports failure, preventing silent
    failures when SGLang versions are incompatible.
    """
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
