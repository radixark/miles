"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

import json
import logging
import math
import os
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Sequence

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
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.q_lora_rank)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        """Both megatron exporters are PP-local after gathering TP/EP; the PP
        gather runs only where the resolved placement asks for it, one owner stage
        at a time, so a rank holds at most one stage's share of the adapter in
        flight instead of the whole adapter (tens of GB for a large MoE)."""
        named_tensors = self._export_pp_local_lora(adapter)
        # TODO: the PP-local branch is unreachable until validate_native_lora_args lifts its
        # colocate-only guard; add an e2e for native-LoRA disaggregate when it does
        if self.placement.gather_pp:
            layout = _gather_pp_adapter_layout(named_tensors)
            names = [name for stage in layout for name, _, _ in stage]
            stages = _iter_pp_adapter_stages(named_tensors, layout)
        else:
            names = [name for name, _ in named_tensors]
            stages = iter([named_tensors])
        del named_tensors
        if materialize:
            if not names:
                raise RuntimeError("LoRA weight sync failed: the adapter export produced zero tensors")
            if not any(is_lora_weight_name(name) for name in names):
                raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B names.")
        for stage in stages:
            # every rank drains the stages in lockstep: each one is a PP broadcast
            if not materialize:
                continue
            while stage:
                hf_name, tensor = stage.pop(0)
                yield [(hf_name, tensor)]

    @abstractmethod
    def _export_pp_local_lora(self, adapter) -> list[tuple[str, torch.Tensor]]:
        """Backend hook: the adapter's HF-named tensors, TP/EP gathered, PP-local."""


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


def _gather_pp_adapter_layout(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[list[tuple[str, tuple[int, ...], torch.dtype]]]:
    """(name, shape, dtype) of every PP stage's adapter tensors, on every PP rank."""
    pp = get_parallel_state().pp
    if pp.size == 1:
        return [[(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]]
    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp.size
    dist.all_gather_object(all_meta, local_meta, group=pp.group)
    return all_meta


def _iter_pp_adapter_stages(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
    layout: list[list[tuple[str, tuple[int, ...], torch.dtype]]],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Gather the complete adapter onto every PP rank one owner stage at a time
    (one flat broadcast per (owner, dtype)), yielding that stage's tensors sorted
    by name. A stage's buffers are released once its consumer drops the tensors,
    before the next stage is broadcast."""
    pp = get_parallel_state().pp
    if pp.size == 1:
        # the export itself, drained in place by the consumer (export order, no copy)
        yield hf_named_tensors if isinstance(hf_named_tensors, list) else list(hf_named_tensors)
        return
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()
    local_by_name = {n: t for n, t in hf_named_tensors}
    for src, meta in enumerate(layout):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        stage: list[tuple[str, torch.Tensor]] = []
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src == pp.rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src], group=pp.group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                stage.append((n, flat[off : off + k].view(shape)))
                off += k
            del flat
        stage.sort(key=lambda item: item[0])
        yield stage
        del stage


def _iter_mm_tower_units(args, *, materialize):
    if "inkling_mm_model_provider" not in (args.custom_model_provider_path or ""):
        return
    yield from iter_checkpoint_tower_units(args.hf_checkpoint, materialize=materialize)
