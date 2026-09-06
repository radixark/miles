"""Multimodal tower tensors streamed from the HF checkpoint files.

A text-only trainer of a multimodal checkpoint holds no vision or audio tower,
but a protocol that seeds the engine from the stream needs every tensor the
engine holds, and ``--ci-test`` compares the engine against the checkpoint. The
towers are untrained, so the checkpoint is the uniform source every rank can
re-send from; loads are idempotent.
"""

import glob
import json
import os

import torch
from safetensors import safe_open

_CACHE: dict[str, list[tuple[str, torch.Tensor]]] = {}


def is_tower_key(name: str) -> bool:
    return name.startswith(("visual.", "audio.")) or ".visual." in f".{name}" or ".audio." in f".{name}"


def checkpoint_shards(hf_checkpoint: str) -> dict[str, list[str]]:
    """Tower keys grouped by the safetensors shard that holds them."""
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


def iter_checkpoint_tower_units(hf_checkpoint: str, *, materialize: bool):
    if not materialize:
        return
    cache = _CACHE.get(hf_checkpoint)
    if cache is None:
        cache = []
        for shard, keys in sorted(checkpoint_shards(hf_checkpoint).items()):
            with safe_open(os.path.join(hf_checkpoint, shard), framework="pt", device="cpu") as f:
                for key in sorted(keys):
                    cache.append((key, f.get_tensor(key)))
        _CACHE[hf_checkpoint] = cache
    for name, tensor in cache:
        yield [(name, tensor.to(torch.cuda.current_device()))]
