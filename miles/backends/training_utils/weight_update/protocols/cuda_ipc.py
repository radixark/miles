import hashlib
import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle
from sglang.srt.utils import MultiprocessingSerializer

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils.lora import lora_base_cpu_backup_enabled

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

from .broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateWeightFromTensor(WeightTransferProtocol):
    """
    Colocated transfer: every training rank serializes its bucket to CUDA-IPC
    handles and gather_objects them to its engine group's src rank, which hands
    them to the engine over Ray. Hybrid deployments broadcast to the
    non-colocated engine tail over NCCL from the global source rank.
    """

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        """
        Create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        super().__init__(args)
        self._model_update_groups = None
        # Overwritten with "miles" when connect finds a distributed engine tail.
        self._group_name = "miles-colocate"

        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = min(start_rank + self.args.rollout_num_gpus_per_engine, dist.get_world_size())
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

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
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self._selector = selector

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
                parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0 and parallel_state.pp.rank == 0
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

        # Every engine-covered rank sends: the per-engine gather_object is
        # collective among the group's members.
        self.is_sender = self._ipc_gather_group is not None
        self.is_lora_sender = self.is_sender

        # A LoRA sync must re-push the frozen base unless the engines keep it
        # across pauses (CPU backup, persistent GPU copy, or remote engines);
        # the weight checker always needs the full rewrite.
        base_persists = (
            self.use_distribute
            or lora_base_cpu_backup_enabled(self.args)
            or (self.args.colocate and not self.args.offload_rollout)
        )
        self.needs_base_resync_for_lora = self.args.check_weight_update_equal or not base_persists

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]], weight_version: int) -> None:
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=bucket,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=self._selector,
            weight_version=weight_version,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                weight_version,
                self.distributed_rollout_engines,
                bucket,
                selector=self._selector,
            )
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        check_weight_sync_results(ray.get(refs or []), is_lora=False)
        del long_lived_tensors

    def send_adapter(
        self, named_tensors: list[tuple[str, torch.Tensor]], *, lora_name: str, lora_config: dict, upsert: bool
    ) -> None:
        if upsert:
            raise NotImplementedError("multi-LoRA weight sync is not supported for colocated engines")
        if self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")

        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=self._selector,
            lora_config=lora_config,
            lora_name=lora_name,
            check_equal=self.args.check_lora_weight_equal,
            repack_lora_for_ipc=self.args.offload_train,
        )
        check_weight_sync_results(ray.get(refs or []), is_lora=True)
        del long_lived_tensors
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def _repack_onto_fresh_storage(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Copy CUDA tensors into freshly allocated flat buffers and return views onto them.

    ``torch_memory_saver.region()`` is a ``torch.cuda.use_mem_pool`` context, so anything
    allocated while building the model -- the LoRA adapter parameters included -- lives in
    a MemPool the preloaded hook backs with cuMem. cuMem allocations cannot be exported
    over the legacy CUDA IPC API that ``MultiprocessingSerializer`` uses, so handing their
    storages straight to the engine fails with "CUDA error: invalid argument" on the first
    sync. Allocating here, outside any region, gets normal caching-allocator memory whose
    handles export fine. (This is also why the FlattenedTensorBucket path works: its
    flattened tensor is likewise allocated at sync time.)

    One buffer per (dtype, device) rather than one clone per tensor: the direct-dict
    transport relies on the pickler memoizing storages so the engine receives a handful of
    IPC handles instead of one per adapter tensor.
    """
    groups: dict[tuple[torch.dtype, torch.device], list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in named_tensors:
        if tensor.is_cuda:
            groups.setdefault((tensor.dtype, tensor.device), []).append((name, tensor))

    views: dict[str, torch.Tensor] = {}
    for (dtype, device), items in groups.items():
        flat = torch.empty(sum(t.numel() for _, t in items), dtype=dtype, device=device)
        offset = 0
        for name, tensor in items:
            view = flat[offset : offset + tensor.numel()].view(tensor.shape)
            view.copy_(tensor)
            views[name] = view
            offset += tensor.numel()

    return {name: views.get(name, tensor) for name, tensor in named_tensors}


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
    lora_config: dict | None = None,
    lora_name: str | None = None,
    check_equal: bool = False,
    selector: str = "all",
    repack_lora_for_ipc: bool = False,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_lora = lora_config is not None
    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if is_lora:
        payload = _repack_onto_fresh_storage(hf_named_tensors) if repack_lora_for_ipc else dict(hf_named_tensors)
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
            try:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))
            except Exception as _unload_err:
                logger.debug("lora unload before load skipped: %s", _unload_err)

            expected_checksums = None
            if check_equal:
                expected_checksums = {
                    n: hashlib.sha256(
                        t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for n, t in hf_named_tensors
                }

            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_named_tensors=[
                        per_rank[0] if per_rank else None for per_rank in serialized_named_tensors
                    ],
                    expected_checksums=expected_checksums,
                )
            )

        else:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                    "selector": selector,
                }
                refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
