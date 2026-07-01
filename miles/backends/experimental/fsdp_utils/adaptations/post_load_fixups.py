"""Post-load weight fixups for the FSDP backend (re-assert on-disk values clobbered by from_pretrained)."""

import glob
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class PostLoadFixup:
    name: str
    applies_to: Callable  # (hf_config) -> bool
    apply: Callable  # (model, ckpt_path) -> int (count of params re-asserted)


_FIXUPS: list[PostLoadFixup] = []


def register_post_load_fixup(fixup: PostLoadFixup) -> None:
    _FIXUPS.append(fixup)


def apply_post_load_fixups(model, hf_config, ckpt_path) -> list[str]:
    """Run every registered fixup whose arch-predicate matches; return the names that fired."""
    fired = []
    for f in _FIXUPS:
        if f.applies_to(hf_config) and f.apply(model, ckpt_path):
            fired.append(f.name)
    return fired


def _is_mamba_hybrid(hf_config) -> bool:
    """True for Mamba/SSM-hybrid archs whose HF `_init_weights` clobbers loaded weights post-load."""
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    if "nemotron_h" in model_type or "mamba" in model_type:
        return True
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return bool(layer_types) and any("mamba" in str(t).lower() for t in layer_types)


def _reload_clobbered_from_disk(model, ckpt_path, tol=1e-3) -> int:
    """Reload params differing from the on-disk checkpoint by > ``tol`` (meta ranks skipped); return count."""
    try:
        from safetensors import safe_open
    except Exception:  # pragma: no cover
        return 0
    files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not files:
        return 0
    index = os.path.join(ckpt_path, "model.safetensors.index.json")
    shard_of = json.load(open(index))["weight_map"] if os.path.exists(index) else {}

    reloaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                continue
            shards = [os.path.join(ckpt_path, shard_of[name])] if name in shard_of else files
            for f in shards:
                try:
                    with safe_open(f, framework="pt") as sf:
                        if name not in sf.keys():
                            continue
                        disk = sf.get_tensor(name)
                except Exception:
                    continue
                if disk.shape == param.shape:
                    disk = disk.to(param.dtype)
                    if (param.detach() - disk).abs().max().item() > tol:
                        param.copy_(disk)
                        reloaded += 1
                break
    if reloaded:
        logger.info(
            "[fsdp post_load] re-asserted %d checkpoint param(s) that from_pretrained clobbered "
            "post-load (Mamba _init_weights)",
            reloaded,
        )
    return reloaded
