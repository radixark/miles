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
from miles.backends.training_utils.weight_update.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)


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

    def _export_lora_named_tensors(self, adapter):
        # Both megatron exporters gather TP/EP but not PP.
        return _gather_pp_full_adapter(self._export_pp_local_lora(adapter))

    @abstractmethod
    def _export_pp_local_lora(self, adapter) -> list[tuple[str, torch.Tensor]]:
        """The adapter's HF-named tensors, TP/EP gathered, PP-local."""


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
