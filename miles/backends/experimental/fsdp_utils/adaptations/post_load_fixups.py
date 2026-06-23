"""Post-load weight fixups for the FSDP backend.

Some stock-HF architectures corrupt loaded weights during ``from_pretrained``: NemotronH's Mamba2
``_init_weights`` runs AFTER weight loading and re-initializes mixer special-init params (every
layer's ``mixer.dt_bias`` + ``mixer.out_proj.weight``). The FSDP backend would then train on those
wrong values and sync them to the rollout. Each affected arch registers a fixup that runs on the
instantiated model + checkpoint path and re-asserts disk ground truth.

This is the FSDP backend's fourth registry, alongside ``weight_bridge`` (train->rollout param
contract), ``class_patches.ModelPatchHook`` (config-time HF-compat patches), and ``packing``
(packed-sequence layout). A new arch registers a fixup here instead of editing the actor.
"""

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
    """Run every registered fixup whose arch-predicate matches. Returns the names that fired.

    Runs only where weights are materialized (rank-0 CPU load); meta-device ranks are skipped
    inside each fixup and get the corrected value via the rank-0 broadcast in
    ``_fsdp2_load_full_state_dict``.
    """
    fired = []
    for f in _FIXUPS:
        if f.applies_to(hf_config) and f.apply(model, ckpt_path):
            fired.append(f.name)
    return fired


# ------------------------------------------------------------------------------------------------
# NemotronH / Mamba2-hybrid: transformers' _init_weights re-inits mixer.dt_bias + out_proj AFTER
# loading; re-assert the on-disk checkpoint (ground truth) over any param it clobbered.
# ------------------------------------------------------------------------------------------------
def _is_mamba_hybrid(hf_config) -> bool:
    """True for Mamba/SSM-hybrid archs whose HF `_init_weights` clobbers loaded weights post-load."""
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    if "nemotron_h" in model_type or "mamba" in model_type:
        return True
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return bool(layer_types) and any("mamba" in str(t).lower() for t in layer_types)


def _reload_clobbered_from_disk(model, ckpt_path, tol=1e-3) -> int:
    """Reload every param whose materialized value differs from the on-disk checkpoint by > ``tol``.

    Gated upstream to Mamba/hybrid archs so it never reverts an intended from_pretrained transform
    elsewhere. Meta-device ranks are skipped (they get the corrected value via the rank-0 broadcast).
    Returns the count of params re-asserted.
    """
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


# ``_is_mamba_hybrid`` + ``_reload_clobbered_from_disk`` are reusable mechanism; the arch that needs
# them (NemotronH) registers the fixup in its spec (adaptations/specs/nemotron_h.py).
