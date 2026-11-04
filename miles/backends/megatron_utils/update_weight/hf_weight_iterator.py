"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

import json
import logging
import math
import os
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Iterable, Iterator, Sequence

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)
from miles.utils.lora.utils import is_lora_weight_name

logger = logging.getLogger(__name__)


class MegatronHfWeightIteratorBase(HfWeightIteratorBase):
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packed_weight_basenames = (
            get_packed_weight_basenames(self.args.hf_checkpoint)
            if self.quantization_config is not None
            and self.quantization_config["quant_method"] == "compressed-tensors"
            else None
        )

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.backend.q_lora_rank)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        """Both megatron exporters are PP-local after gathering TP/EP; the PP
        gather runs only where the resolved placement asks for it."""
        chunks = self._iter_pp_local_lora(adapter)
        if self.placement.gather_pp:
            chunks = _stream_pp_full_adapter(chunks)
        num_tensors = num_lora_weights = 0
        # every rank drains the chunks: exporting and gathering them are collectives
        for chunk in chunks:
            num_tensors += len(chunk)
            num_lora_weights += sum(is_lora_weight_name(name) for name, _ in chunk)
            if materialize:
                yield from ([(hf_name, tensor)] for hf_name, tensor in chunk)
        num_tensors, num_lora_weights = _count_full_adapter_tensors(
            num_tensors, num_lora_weights, is_pp_local=not self.placement.gather_pp
        )
        if not materialize:
            return
        if num_tensors == 0:
            raise RuntimeError("LoRA weight sync failed: the adapter export produced zero tensors")
        if num_lora_weights == 0:
            raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B names.")

    @abstractmethod
    def _iter_pp_local_lora(self, adapter) -> Iterable[list[tuple[str, torch.Tensor]]]:
        """Backend hook: the adapter's HF-named tensors in chunks, TP/EP gathered, PP-local."""


def get_hf_weight_iterator(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    cls = {
        "raw": HfWeightIteratorDirect,
        "bridge": HfWeightIteratorBridge,
    }[args.megatron_to_hf_mode]

    return cls.build(
        args,
        model,
        required_placement=required_placement,
        model_name=model_name,
        quantization_config=quantization_config,
    )


def get_packed_weight_basenames(hf_checkpoint: str) -> set[str]:
    """Base names the checkpoint stores as compressed-tensors `weight_packed`; the quantizer
    re-quantizes exactly these, since the published `ignore` list is written for loaders and
    leaves out BF16 weights such as routers, residual projections and the vision tower."""
    with open(os.path.join(hf_checkpoint, "model.safetensors.index.json")) as index_file:
        names = json.load(index_file)["weight_map"]
    return {n.removesuffix(".weight_packed") for n in names if n.endswith(".weight_packed")}


def _count_full_adapter_tensors(num_tensors: int, num_lora_weights: int, *, is_pp_local: bool) -> tuple[int, int]:
    """(tensors, lora_A/lora_B weights) in the whole adapter; a PP-local export is
    summed over PP since a stage may hold no adapted layer."""
    counts = [num_tensors, num_lora_weights]
    pp = get_parallel_state().pp
    if is_pp_local and pp.size > 1:
        counts_tensor = torch.tensor(counts, device=torch.cuda.current_device())
        dist.all_reduce(counts_tensor, group=pp.group)
        counts = counts_tensor.tolist()
    return counts[0], counts[1]


def _stream_pp_full_adapter(
    local_chunks: Iterable[list[tuple[str, torch.Tensor]]],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Stream the complete adapter onto every PP rank, one PP stage after another and one chunk at
    a time: the owner broadcasts each chunk's metadata, then one flat buffer per dtype."""
    pp = get_parallel_state().pp
    if pp.size == 1:
        yield from local_chunks
        return
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()
    # only the owning stage advances its exporter; the others wait in the broadcasts
    local_iter = iter(local_chunks)
    for src in range(pp.size):
        while True:
            local = next(local_iter, None) if src == pp.rank else None
            meta = [None if local is None else [(n, tuple(t.shape), t.dtype) for n, t in local]]
            dist.broadcast_object_list(meta, src=global_ranks[src], group=pp.group)
            if meta[0] is None:
                break
            yield _broadcast_chunk(meta[0], local, src_rank=global_ranks[src], group=pp.group, device=device)


def _broadcast_chunk(meta, local, *, src_rank: int, group, device) -> list[tuple[str, torch.Tensor]]:
    local_by_name = dict(local) if local is not None else {}
    by_dtype: dict = {}
    for n, shape, dtype in meta:
        by_dtype.setdefault(dtype, []).append((n, shape))
    received: dict[str, torch.Tensor] = {}
    for dtype, entries in by_dtype.items():
        flat = torch.empty(sum(math.prod(shape) for _, shape in entries), dtype=dtype, device=device)
        if local is not None:
            torch.cat([local_by_name[n].reshape(-1) for n, _ in entries], out=flat)
        dist.broadcast(flat, src=src_rank, group=group)
        off = 0
        for n, shape in entries:
            k = math.prod(shape)
            received[n] = flat[off : off + k].view(shape)
            off += k
    return [(n, received[n]) for n, _, _ in meta]


def _iter_mm_tower_units(args, *, materialize):
    if (provider := args.custom_model_provider_path) is None or "inkling_mm_model_provider" not in provider.path:
        return
    yield from iter_checkpoint_tower_units(args.hf_checkpoint, materialize=materialize)
