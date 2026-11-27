import functools
import glob
import json
import os

import torch
from safetensors import safe_open

from miles.platforms import current_platform

_TOWERS = ("visual", "audio")


def is_tower_key(name: str) -> bool:
    return any(part in _TOWERS for part in name.split(".")[:-1])


def _keys_by_shard(hf_checkpoint: str, select) -> dict[str, list[str]]:
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for key, shard in json.load(f)["weight_map"].items():
                if select(key):
                    by_shard.setdefault(shard, []).append(key)
        return by_shard
    for path in sorted(glob.glob(os.path.join(hf_checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if select(k)]
        if keys:
            by_shard[os.path.basename(path)] = keys
    return by_shard


@functools.cache
def _load(hf_checkpoint: str, select) -> tuple[tuple[str, torch.Tensor], ...]:
    loaded = []
    for shard, keys in sorted(_keys_by_shard(hf_checkpoint, select).items()):
        with safe_open(os.path.join(hf_checkpoint, shard), framework="pt", device="cpu") as f:
            loaded.extend((key, f.get_tensor(key)) for key in sorted(keys))
    return tuple(loaded)


def load_checkpoint_towers(hf_checkpoint: str) -> tuple[tuple[str, torch.Tensor], ...]:
    return _load(hf_checkpoint, is_tower_key)


def iter_checkpoint_units(hf_checkpoint: str, *, select, unit_key=None, convert=None, materialize: bool, keep=None):
    """Checkpoint tensors the trainer never exports, sent on every sync (read once, kept on CPU).

    ``unit_key`` names the load-call unit of a key (tensors serving fuses in one load share it); default one each.
    ``convert`` maps a unit to the tensors serving's update path expects; default as stored.
    ``keep`` narrows the cached selection per call (e.g. to one pipeline stage's layers); default all."""
    if not materialize:
        return
    # Group on CPU and move one unit at a time, so a sync holds one unit on the device, not the whole selection.
    units: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in _load(hf_checkpoint, select):
        if keep is not None and not keep(name):
            continue
        units.setdefault(unit_key(name) if unit_key else name, []).append((name, tensor))
    device = current_platform.current_device()
    for unit in units.values():
        unit = [(name, tensor.to(device)) for name, tensor in unit]
        yield convert(unit) if convert else unit


def iter_checkpoint_tower_units(hf_checkpoint: str, *, materialize: bool):
    yield from iter_checkpoint_units(hf_checkpoint, select=is_tower_key, materialize=materialize)
