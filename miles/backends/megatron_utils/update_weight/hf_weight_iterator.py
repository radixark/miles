"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

import logging
import math
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.utils.lora import is_lora_weight_name

logger = logging.getLogger(__name__)


class MegatronHfWeightIteratorBase(HfWeightIteratorBase):
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
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
        for hf_name, tensor in named_tensors:
            yield [(f"{lora_name}:{hf_name}", tensor)]

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

    return cls(
        args,
        model,
        placement=resolve_placement(required_placement, cls.forced_placement),
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


_MM_TOWER_CACHE: list[tuple[str, "torch.Tensor"]] | None = None


def _iter_mm_tower_units(args, *, materialize):
    """Transitional: Inkling MM trains only the language model; the frozen
    vision/audio towers are deliberately unregistered trainer-side (ckpt
    compat) and exist only on pre_process ranks, so the boot checkpoint is the
    uniform source every rank can re-send from (loads are idempotent).
    Goes away when the towers become real megatron params (Kimi-style) or the
    engine keeps them across offload."""
    global _MM_TOWER_CACHE
    if "inkling_mm_model_provider" not in (args.custom_model_provider_path or ""):
        return
    if not materialize:
        return
    if _MM_TOWER_CACHE is None:
        import json
        import os

        from safetensors import safe_open

        ckpt_dir = args.hf_checkpoint
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
        logger.info("mm tower sync: caching %d tower tensors from %s", len(cache), ckpt_dir)
        _MM_TOWER_CACHE = cache
    for name, tensor in _MM_TOWER_CACHE:
        yield [(name, tensor.to(torch.cuda.current_device()))]
