import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.megatron_utils.lora_utils import (
    LORA_ADAPTER_NAME,
    build_lora_sync_config,
    is_lora_weight_name,
    lora_base_cpu_backup_enabled,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .common import _check_weight_sync_results, begin_weight_update, end_weight_update
from .hf_weight_iterator_base import HfWeightIteratorBase

from .update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


def _pp_assemble_full_adapter(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Assemble the complete adapter on every PP rank (exporter gathers TP/EP but not PP)."""
    pp_group = get_parallel_state().pp.group
    pp_size = dist.get_world_size(group=pp_group)
    if pp_size == 1:
        return hf_named_tensors
    pp_rank = dist.get_rank(group=pp_group)
    global_ranks = dist.get_process_group_ranks(pp_group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp_size
    dist.all_gather_object(all_meta, local_meta, group=pp_group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src_pp, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src_pp == pp_rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src_pp], group=pp_group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                # tensors replicated across stages resolve to the last copy (identical by construction)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())


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
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=self.is_lora,
        )
        if self.is_lora:
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False
            self._lora_base_synced = False

        self._mm_tower_cache: list[tuple[str, torch.Tensor]] | None = None

        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = min(start_rank + self.args.rollout_num_gpus_per_engine, dist.get_world_size())
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and get_parallel_state().tp.rank == 0
                and get_parallel_state().pp.rank == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, g, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()

        # LoRA never mutates the base, so any path that retains it on the rollout side can skip the base sync.
        colocate_base_persistent = getattr(self.args, "colocate", False) and not getattr(
            self.args, "offload_rollout", True
        )
        skip_base_sync = self.is_lora and (
            self.use_distribute or lora_base_cpu_backup_enabled(self.args) or colocate_base_persistent
        )

        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if not skip_base_sync:
                begin_weight_update(self.rollout_engines)
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        if not skip_base_sync:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors

            mm_tower_tensors = self._mm_tower_named_tensors()
            if mm_tower_tensors is not None:
                mm_tower_tensors = [
                    (name, tensor.to(torch.cuda.current_device())) for name, tensor in mm_tower_tensors
                ]
                refs, long_lived_tensors = self._send_base_params(mm_tower_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors, mm_tower_tensors

        if self.is_lora:
            # SGLang's load_lora_adapter_from_tensors expects the full adapter in
            # one call; drain the bridge's chunker so --update-weight-buffer-size
            # only bounds the base path.
            accumulated_named_tensors: list = []
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="lora"
            ):
                accumulated_named_tensors.extend(hf_named_tensors)

            if not accumulated_named_tensors:
                raise RuntimeError(
                    "LoRA weight sync failed: the weight iterator produced zero chunks. "
                    "No adapter weights were sent to the rollout engine. This usually means "
                    "the Megatron-Bridge or SGLang version is incompatible."
                )

            # Assemble the full adapter on every rank before serializing.
            accumulated_named_tensors = _pp_assemble_full_adapter(accumulated_named_tensors)

            refs, long_lived_tensors = self._send_lora_params(accumulated_named_tensors)
            results = ray.get(refs)
            _check_weight_sync_results(results, is_lora=True)
            del long_lived_tensors
            del accumulated_named_tensors
            # Reclaim producer IPC storages now to avoid OOM on the next step.
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            if not self._lora_base_synced:
                self._lora_base_synced = True

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            # Skip when no fresh base bytes landed (skip_base_sync).
            if not skip_base_sync:
                end_weight_update(self.rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _mm_tower_named_tensors(self) -> list[tuple[str, torch.Tensor]] | None:
        """Frozen vision/audio tower tensors to append to every base sync (see
        __init__ comment). Returns None when the run has no MM towers. EVERY
        gather-group rank contributes the full tower set (read once from its local
        HF checkpoint, the same bytes the engine loaded at boot): the colocated
        send requires homogeneous per-rank bucket counts (num_dtypes is taken from
        rank 0 and indexed into every rank's list), so a src-only contribution
        breaks assembly. The duplicates are ~15MB/rank and load idempotently."""
        if not getattr(self.args, "inkling_mm_towers", False):
            return None
        if self._mm_tower_cache is None:
            if self._ipc_gather_group is not None:
                import json

                from safetensors import safe_open

                ckpt_dir = self.args.hf_checkpoint
                with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
                    weight_map = json.load(f)["weight_map"]
                tower_keys = sorted(
                    k
                    for k in weight_map
                    if ".visual." in f".{k}" or ".audio." in f".{k}" or k.startswith(("visual.", "audio."))
                )
                by_shard: dict[str, list[str]] = {}
                for k in tower_keys:
                    by_shard.setdefault(weight_map[k], []).append(k)
                cache = []
                for shard, keys in by_shard.items():
                    with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
                        for k in keys:
                            cache.append((k, f.get_tensor(k)))
                logger.info(
                    "mm tower sync: caching %d tower tensors from %s: %s",
                    len(cache),
                    ckpt_dir,
                    [k for k, _ in cache],
                )
                self._mm_tower_cache = cache
            else:
                self._mm_tower_cache = []
        return self._mm_tower_cache

    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        return refs or [], long_lived_tensors

    def _send_lora_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        if not any(is_lora_weight_name(n) for n, _ in hf_named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator configuration."
            )
        if self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")

        # Adapter-only sync: the full unsharded adapter goes to the engine, which shards it per TP rank internally.
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            lora_config=self._lora_config,
            lora_name=LORA_ADAPTER_NAME,
            lora_loaded=self._lora_loaded,
            check_equal=getattr(self.args, "check_lora_weight_equal", False),
        )
        self._lora_loaded = True
        return refs or [], long_lived_tensors


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
    check_equal: bool = False,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_lora = lora_config is not None
    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if is_lora:
        # Serialize the named dict directly (no FlattenedTensorBucket); the pickler memoizes storages so IPC shares each flat once.
        payload = dict(hf_named_tensors)
        long_live_tensors.append(payload)
        converted_named_tensors_by_dtypes = {}
        serialized_lora = MultiprocessingSerializer.serialize(payload, output_str=True)
    elif getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors: list = [serialized_lora] if is_lora else []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if is_gather_src:
        if is_lora:
            # Unload before (re)loading: the engine rejects a duplicate adapter name.
            try:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))
            except Exception as _unload_err:  # noqa: BLE001 - first sync: nothing to unload
                logger.debug("lora unload before load skipped: %s", _unload_err)

            # Per-rank transport: engine TP rank j deserializes the bucket of the train rank sharing its GPU.
            _expected_checksums = None
            if check_equal:
                import hashlib

                _expected_checksums = {
                    n: hashlib.sha256(
                        t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for n, t in hf_named_tensors
                }
            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_named_tensors=[t[0] for t in serialized_named_tensors],
                    expected_checksums=_expected_checksums,
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

    return refs, long_live_tensors
