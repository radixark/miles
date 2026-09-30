"""Collective HF snapshots using TorchTitan's model adapter and checkpointer."""

import json
import logging
import shutil
from pathlib import Path

import safetensors
import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import is_tower_key
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER

logger = logging.getLogger(__name__)

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


@torch.no_grad()
def export_hf(checkpointer, *, hf_checkpoint: str, path: str) -> None:
    """All training ranks participate; `.complete` is written after assets are copied."""
    source, destination = Path(hf_checkpoint).resolve(), Path(path).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("HF export directory must be separate from --hf-checkpoint; choose another export path.")
    if _has_tower_weights(source):
        raise NotImplementedError(
            "TorchTitan HF export does not support multimodal checkpoints yet; "
            "TorchTitan must export all vision and audio weights."
        )
    if dist.get_rank() == 0:
        (destination / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)
        # Titan consolidates every intermediate shard, including leftovers from a failed save.
        if (destination / "sharded").exists():
            shutil.rmtree(destination / "sharded")
    dist.barrier()
    state_dict = checkpointer.states["model"].state_dict()
    checkpointer.dcp_save(state_dict, checkpoint_id=str(destination), async_mode="disabled", to_hf=True)
    if dist.get_rank() == 0:
        _complete_export(source, destination)
        logger.info(f"Exported torchtitan HF snapshot to {destination}")


def _has_tower_weights(source: Path) -> bool:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        return any(is_tower_key(name) for name in json.loads(index_path.read_text())["weight_map"])
    for shard in source.glob("*.safetensors"):
        with safetensors.safe_open(shard, framework="pt", device="cpu") as weights:
            if any(is_tower_key(name) for name in weights.keys()):
                return True
    return False


def _complete_export(
    source: Path,
    destination: Path,
) -> None:
    index_path = destination / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for asset in source.iterdir():
        if (
            asset.is_file()
            and asset.suffix not in _WEIGHT_SUFFIXES
            and not asset.name.endswith(".index.json")
            and asset.name != HF_EXPORT_COMPLETE_MARKER
        ):
            shutil.copy2(asset, destination / asset.name)
    # Inference loaders may glob shards rather than follow the HF index.
    shard_names = set(index["weight_map"].values())
    for shard in destination.glob("*.safetensors"):
        if shard.name not in shard_names:
            shard.unlink()
    shutil.rmtree(destination / "sharded", ignore_errors=True)
    (destination / HF_EXPORT_COMPLETE_MARKER).touch()
