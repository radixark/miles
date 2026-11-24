"""Write the trained model as an HF checkpoint the rollout engine can load (--save-hf)."""

import json
import logging
import os
import shutil
from collections.abc import Iterator

import safetensors
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from miles.backends.torchtitan_utils.hf_weight_iterator import checkpoint_dtypes, checkpoint_shard_files, hf_weights

logger = logging.getLogger(__name__)

_SHARD_BYTES = 4 << 30


def save_hf_checkpoint(trainer, *, hf_checkpoint: str, path: str) -> None:
    """Collective: every rank gathers the weights, rank 0 writes a complete checkpoint to ``path``.

    Trained tensors keep the source checkpoint's dtypes; tensors the trainer does not hold (frozen towers,
    MTP) and the non-weight files are copied from ``hf_checkpoint``.
    """
    weights = hf_weights(trainer, complete_across_pp=True)
    if dist.get_rank() != 0:
        for _ in weights:
            pass
        dist.barrier()
        return
    os.makedirs(path, exist_ok=True)
    weight_map = _write_trained_shards(weights, dtypes=checkpoint_dtypes(hf_checkpoint), path=path)
    num_trained = len(weight_map)
    weight_map.update(_copy_untouched_tensors(hf_checkpoint, trained=set(weight_map), path=path))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    _copy_files_outside_the_index(hf_checkpoint, path=path)
    logger.info(f"saved HF checkpoint to {path}: {num_trained} trained, {len(weight_map) - num_trained} copied")
    dist.barrier()


def _write_trained_shards(
    weights: Iterator[tuple[str, torch.Tensor]], *, dtypes: dict[str, torch.dtype], path: str
) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    num_shards = 0
    for name, tensor in weights:
        shard[name] = tensor.detach().to(dtypes.get(name, tensor.dtype)).cpu().contiguous()
        shard_bytes += shard[name].nbytes
        if shard_bytes >= _SHARD_BYTES:
            weight_map.update(_save_shard(shard, path=path, file_name=f"model-trained-{num_shards:05d}.safetensors"))
            shard, shard_bytes, num_shards = {}, 0, num_shards + 1
    if shard:
        weight_map.update(_save_shard(shard, path=path, file_name=f"model-trained-{num_shards:05d}.safetensors"))
    return weight_map


def _copy_untouched_tensors(hf_checkpoint: str, *, trained: set[str], path: str) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    for index, source_file in enumerate(checkpoint_shard_files(hf_checkpoint)):
        with safetensors.safe_open(os.path.join(hf_checkpoint, source_file), framework="pt") as source:
            untouched = {name: source.get_tensor(name) for name in source.keys() if name not in trained}
        if untouched:
            weight_map.update(_save_shard(untouched, path=path, file_name=f"model-frozen-{index:05d}.safetensors"))
    return weight_map


def _save_shard(tensors: dict[str, torch.Tensor], *, path: str, file_name: str) -> dict[str, str]:
    save_file(tensors, os.path.join(path, file_name), metadata={"format": "pt"})
    return {name: file_name for name in tensors}


def _copy_files_outside_the_index(hf_checkpoint: str, *, path: str) -> None:
    # a weight file the index does not list (the MTP head's mtp.safetensors) is copied whole
    indexed = {*checkpoint_shard_files(hf_checkpoint), "model.safetensors.index.json"}
    for entry in os.listdir(hf_checkpoint):
        if entry in indexed:
            continue
        source = os.path.join(hf_checkpoint, entry)
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(path, entry), dirs_exist_ok=True)
        else:
            shutil.copy2(source, os.path.join(path, entry))
