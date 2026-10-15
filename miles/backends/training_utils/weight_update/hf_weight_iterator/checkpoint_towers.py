import functools
import glob
import json
import os

import torch
from safetensors import safe_open

_TOWERS = ("visual", "audio")


def is_tower_key(name: str) -> bool:
    return any(part in _TOWERS for part in name.split(".")[:-1])


def _tower_keys_by_shard(hf_checkpoint: str, prefixes: tuple[str, ...] | None) -> dict[str, list[str]]:
    def selected(key: str) -> bool:
        return key.startswith(prefixes) if prefixes is not None else is_tower_key(key)

    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for key, shard in json.load(f)["weight_map"].items():
                if selected(key):
                    by_shard.setdefault(shard, []).append(key)
        return by_shard
    for path in sorted(glob.glob(os.path.join(hf_checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if selected(k)]
        if keys:
            by_shard[os.path.basename(path)] = keys
    return by_shard


@functools.cache
def _load_towers(hf_checkpoint: str, prefixes: tuple[str, ...] | None) -> tuple[tuple[str, torch.Tensor], ...]:
    towers = []
    for shard, keys in sorted(_tower_keys_by_shard(hf_checkpoint, prefixes).items()):
        with safe_open(os.path.join(hf_checkpoint, shard), framework="pt", device="cpu") as f:
            towers.extend((key, f.get_tensor(key)) for key in sorted(keys))
    return tuple(towers)


def iter_checkpoint_tower_units(hf_checkpoint: str, *, materialize: bool, prefixes: tuple[str, ...] | None = None):
    """Re-send the checkpoint's tower tensors: names starting with ``prefixes``, or ``visual``/``audio`` modules."""
    if not materialize:
        return
    for name, tensor in _load_towers(hf_checkpoint, prefixes):
        yield [(name, tensor.to(torch.cuda.current_device()))]
