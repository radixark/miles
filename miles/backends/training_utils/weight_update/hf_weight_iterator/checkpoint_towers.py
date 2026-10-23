import functools
import glob
import json
import logging
import os

import torch
from safetensors import safe_open

from miles.platforms import current_platform

logger = logging.getLogger(__name__)

_TOWERS = ("visual", "audio")

_INKLING_MM_PROVIDER = "inkling_mm_model_provider"
_INKLING_MM_TOWERS = ("visual", "audio")
_warned_implicit_mm_tower_sync = False


def mm_tower_names(args) -> tuple[str, ...]:
    """Module names of the frozen multimodal towers re-sent on every weight
    update; empty when tower sync is off. ``--mm-tower-sync`` decides. The
    Inkling MM provider implies its two towers so launches that predate the
    flag keep syncing. Any other vision model (Qwen3-VL's ``visual``) opts in
    with the flag, so its engines can release every weight during training
    like a text run (radixark/miles#3159)."""
    global _warned_implicit_mm_tower_sync
    names = getattr(args, "mm_tower_sync", None)
    if names is not None:
        return tuple(names)
    if _INKLING_MM_PROVIDER in (getattr(args, "custom_model_provider_path", None) or ""):
        if not _warned_implicit_mm_tower_sync:
            logger.warning(
                "mm tower sync: enabled implicitly by the %s provider; pass --mm-tower-sync %s",
                _INKLING_MM_PROVIDER,
                " ".join(_INKLING_MM_TOWERS),
            )
            _warned_implicit_mm_tower_sync = True
        return _INKLING_MM_TOWERS
    return ()


def is_tower_key(name: str, towers: tuple[str, ...] = _TOWERS) -> bool:
    """`model.visual.blocks.0.attn.proj.weight` and `visual.merger.ln_q.weight`
    both belong to the `visual` tower."""
    return any(part in towers for part in name.split(".")[:-1])


def _tower_keys_by_shard(hf_checkpoint: str, towers: tuple[str, ...] = _TOWERS) -> dict[str, list[str]]:
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for key, shard in json.load(f)["weight_map"].items():
                if is_tower_key(key, towers):
                    by_shard.setdefault(shard, []).append(key)
        return by_shard
    for path in sorted(glob.glob(os.path.join(hf_checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if is_tower_key(k, towers)]
        if keys:
            by_shard[os.path.basename(path)] = keys
    return by_shard


@functools.cache
def _load_towers(hf_checkpoint: str, towers: tuple[str, ...] = _TOWERS) -> tuple[tuple[str, torch.Tensor], ...]:
    loaded = []
    for shard, keys in sorted(_tower_keys_by_shard(hf_checkpoint, towers).items()):
        with safe_open(os.path.join(hf_checkpoint, shard), framework="pt", device="cpu") as f:
            loaded.extend((key, f.get_tensor(key)) for key in sorted(keys))
    return tuple(loaded)


def iter_checkpoint_tower_units(hf_checkpoint: str, *, materialize: bool, towers: tuple[str, ...] = _TOWERS):
    """The frozen tower weights of ``hf_checkpoint`` as weight-update units,
    one tensor each. ``towers`` are the module names matched against the
    checkpoint keys as path parts (``--mm-tower-sync``)."""
    if not materialize:
        return
    for name, tensor in _load_towers(hf_checkpoint, tuple(towers)):
        yield [(name, tensor.to(current_platform.current_device()))]
