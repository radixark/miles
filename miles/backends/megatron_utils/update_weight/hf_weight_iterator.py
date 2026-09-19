"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

import logging
import math
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core.utils import unwrap_model

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)
from miles.utils.lora import is_lora_weight_name

logger = logging.getLogger(__name__)


class MegatronHfWeightIteratorBase(HfWeightIteratorBase):
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trainer_has_mtp = bool(unwrap_model(self.model)[0].config.mtp_num_layers)
        if self.args.sglang_speculative_algorithm and not trainer_has_mtp:
            self.weight_update_selector = "target"

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.q_lora_rank)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        """Both megatron exporters are PP-local after gathering TP/EP; the PP
        gather runs only where the resolved placement asks for it."""
        named_tensors = self._export_pp_local_lora(adapter)
        # TODO: the PP-local branch is unreachable until actor.py lifts its bridge-only guard
        # for distributed LoRA; add an e2e for native-LoRA disaggregate when it does
        if self.placement.gather_pp:
            named_tensors = _gather_pp_full_adapter(named_tensors)
        if not materialize:
            return
        if not named_tensors:
            raise RuntimeError(
                f"LoRA weight sync failed: the adapter export produced zero tensors"
                f"{f' for adapter {adapter!r}' if adapter is not None else ''}. "
                "This usually means the Megatron-Bridge or SGLang version is incompatible."
            )
        if not any(is_lora_weight_name(name) for name, _tensor in named_tensors):
            raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B names.")
        while named_tensors:
            hf_name, tensor = named_tensors.pop(0)
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


def _gather_pp_full_adapter(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Gather the complete adapter onto every PP rank: exchange metadata, then
    one flat broadcast per (owner, dtype)."""
    pp = get_parallel_state().pp
    if pp.size == 1:
        return list(hf_named_tensors)
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp.size
    dist.all_gather_object(all_meta, local_meta, group=pp.group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
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
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())


def _iter_mm_tower_units(args, *, materialize):
    if "inkling_mm_model_provider" not in (args.custom_model_provider_path or ""):
        return
    yield from iter_checkpoint_tower_units(args.hf_checkpoint, materialize=materialize)
