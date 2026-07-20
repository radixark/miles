"""Weight-sync protocol shell, copied from fsdp_utils/update_weight_utils.py (twin file:
mirror fixes into both until the two backends converge into a shared torch_native_utils
module). The wire protocol is byte-identical to the proven shell; the one deliberate
day-one seam is ``named_hf_tensors()``: instead of iterating ``model.state_dict()``
directly, ``UpdateWeight.update_weights()`` consumes an injected
``Iterable[(hf_name, tensor)]`` produced by the titan state-dict adapter. This is the
shape both backends are meant to converge on post-#1469-merge.

Also adopts megatron's begin/end_weight_update bracket around the bucket loop (the
copied fsdp shell omits it) — zero cost for bf16, future-proofs quantized engines.
"""

import abc
import logging
import socket
from argparse import Namespace
from collections.abc import Iterable, Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor

from miles.utils.distributed_utils import get_gloo_group, init_process_group

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

from .dtensor import gather_full_param

logger = logging.getLogger(__name__)


def _layer_grouped_named_params(model, adapter) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield (hf_name, full_tensor) pairs, chunked by decoder layer (plus a top-level
    chunk for embeddings/norm/lm_head). Layer-chunking (not per-tensor) is required by
    any adapter whose to_hf fuses multiple titan tensors into one HF tensor (qkv/conv/
    gate_up fusions): the fusion needs every part present in the same to_hf() call.
    """
    sd = model.state_dict()
    by_layer: dict[str, dict[str, torch.Tensor]] = {}
    top_level: dict[str, torch.Tensor] = {}
    for name, param in sd.items():
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "layers":
            by_layer.setdefault(parts[1], {})[name] = param
        else:
            top_level[name] = param

    def _emit(chunk: dict[str, torch.Tensor]):
        gathered = {name: gather_full_param(param, async_op=True) for name, param in chunk.items()}
        gathered = {name: (p.wait() if hasattr(p, "wait") else p) for name, p in gathered.items()}
        hf_chunk = adapter.to_hf(gathered)
        for hf_name, tensor in hf_chunk.items():
            if isinstance(tensor, DTensor):
                tensor = tensor.to_local()
            yield hf_name, tensor

    yield from _emit(top_level)
    for layer_id in sorted(by_layer, key=int):
        yield from _emit(by_layer[layer_id])


def _assert_full_coverage(pushed_names: set[str], adapter, *, skip_prefixes: tuple[str, ...] = ()) -> None:
    """Per-sync completeness gate: SET EQUALITY, not membership. A membership-only
    check misses silent drops (an adapter's fusion guard that emits nothing when a
    fusion group is incomplete never appears as an "extra" name — it's just absent)."""
    mapping = adapter.fqn_to_index_mapping
    if mapping is None:  # single-file checkpoint (no index.json); nothing to check against
        return
    expected = {k for k in mapping if not k.startswith(skip_prefixes)}
    missing = expected - pushed_names
    extra = pushed_names - expected
    if missing or extra:
        raise RuntimeError(
            f"weight-sync coverage mismatch: missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"
        )


class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module, adapter) -> None:
        self.args = args
        self.model = model
        self.adapter = adapter
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        pass

    def update_weights(self) -> None:
        self.weight_version += 1

        if dist.get_rank() == 0:
            futures = [engine.pause_generation.remote(mode="retract") for engine in self.rollout_engines]
            futures.extend([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get(futures)
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.begin_weight_update.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        pushed_names: set[str] = set()
        bucket, bucket_size = [], 0
        skip_prefixes = tuple(getattr(self.args, "check_weight_update_skip_list", None) or ())
        for hf_name, tensor in _layer_grouped_named_params(self.model, self.adapter):
            pushed_names.add(hf_name)
            tensor_size = tensor.numel() * tensor.element_size()
            if bucket and bucket_size + tensor_size >= self.args.update_weight_buffer_size:
                self.update_bucket_weights(bucket, weight_version=self.weight_version)
                bucket, bucket_size = [], 0
            bucket.append((hf_name, tensor))
            bucket_size += tensor_size
        if bucket:
            self.update_bucket_weights(bucket, weight_version=self.weight_version)

        if getattr(self.args, "check_weight_update_equal", False):
            _assert_full_coverage(pushed_names, self.adapter, skip_prefixes=skip_prefixes)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.end_weight_update.remote() for engine in self.rollout_engines])
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        pass


class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines using CUDA-IPC tensor handles."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                self.tp_rank = dist.get_rank() - start_rank

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        monkey_patch_torch_reductions()
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in named_tensors:
            by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

        serialized_tensors = []
        for _dtype, group in by_dtype.items():
            bucket = FlattenedTensorBucket(named_tensors=group)
            payload = {"flattened_tensor": bucket.get_flattened_tensor(), "metadata": bucket.get_metadata()}
            serialized_tensors.append(MultiprocessingSerializer.serialize(payload, output_str=True))

        gathered = (
            [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
            if dist.get_rank() == self._ipc_gather_src
            else None
        )
        dist.gather_object(
            obj=serialized_tensors, object_gather_list=gathered, dst=self._ipc_gather_src, group=self._ipc_gather_group
        )

        if dist.get_rank() == self._ipc_gather_src:
            num_dtypes = len(gathered[0])
            assert num_dtypes > 0
            for i in range(num_dtypes):
                result = ray.get(
                    self._ipc_engine.update_weights_from_tensor.remote(
                        serialized_named_tensors=[t[i] for t in gathered],
                        load_format="flattened_bucket",
                        flush_cache=False,
                        weight_version=str(weight_version),
                    )
                )
                success = result.get("success", True) if isinstance(result, dict) else getattr(result, "success", True)
                if not success:
                    error_msg = (
                        result.get("error_message") if isinstance(result, dict) else getattr(result, "error_message", "unknown")
                    )
                    raise RuntimeError(f"Weight sync failed on rollout engine: {error_msg}")
            ray.get(self._ipc_engine.flush_cache.remote())


class UpdateWeightFromDistributed(UpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines (non-colocate)."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._is_src_rank = dist.get_rank() == 0
        if not self._is_src_rank:
            return

        self._group_name = "miles_torchtitan"
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        world_size = self.args.rollout_num_gpus + 1

        refs = [
            engine.init_weights_update_group.remote(
                master_address, master_port, i * self.args.rollout_num_gpus_per_engine + 1, world_size,
                self._group_name, backend="nccl",
            )
            for i, engine in enumerate(self.rollout_engines)
        ]
        self._model_update_groups = init_process_group(
            backend="nccl", init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size, rank=0, group_name=self._group_name,
        )
        ray.get(refs)

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        if not self._is_src_rank or not named_tensors:
            return

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[t.dtype for _, t in named_tensors],
                shapes=[t.shape for _, t in named_tensors],
                group_name=self._group_name,
                weight_version=str(weight_version),
            )
            for engine in self.rollout_engines
        ]

        handles = []
        for _name, tensor in named_tensors:
            torch.cuda.empty_cache()
            handles.append(dist.broadcast(tensor.contiguous(), 0, group=self._model_update_groups, async_op=True))
        for h in handles:
            h.wait()
        ray.get(refs)
