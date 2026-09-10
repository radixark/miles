"""Multimodal tower tensors streamed from the HF checkpoint files.

A text-only trainer of a multimodal checkpoint holds no vision or audio tower,
but a protocol that seeds the engine from the stream needs every tensor the
engine holds, and ``--ci-test`` compares the engine against the checkpoint. The
towers are untrained, so the checkpoint is the source every rank re-sends from.
"""

import functools
import glob
import json
import os

import torch
from safetensors import safe_open

_TOWERS = ("visual", "audio")


def is_tower_key(name: str) -> bool:
    return any(part in _TOWERS for part in name.split(".")[:-1])


def _tower_keys_by_shard(hf_checkpoint: str) -> dict[str, list[str]]:
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for key, shard in json.load(f)["weight_map"].items():
                if is_tower_key(key):
                    by_shard.setdefault(shard, []).append(key)
        return by_shard
    for path in sorted(glob.glob(os.path.join(hf_checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if is_tower_key(k)]
        if keys:
            by_shard[os.path.basename(path)] = keys
    return by_shard


@functools.cache
def _load_towers(hf_checkpoint: str) -> tuple[tuple[str, torch.Tensor], ...]:
    towers = []
    for shard, keys in sorted(_tower_keys_by_shard(hf_checkpoint).items()):
        with safe_open(os.path.join(hf_checkpoint, shard), framework="pt", device="cpu") as f:
            towers.extend((key, f.get_tensor(key)) for key in sorted(keys))
    return tuple(towers)


def iter_checkpoint_tower_units(hf_checkpoint: str, *, materialize: bool):
    if not materialize:
        return
    for name, tensor in _load_towers(hf_checkpoint):
        yield [(name, tensor.to(torch.cuda.current_device()))]
