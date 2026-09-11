"""Per-slot checkpoints in megatron's torch_dist format.

Every tensor carries its global shard coordinates and a slot-agnostic key,
so a checkpoint reloads under any tp/pp/ep/world-size layout and into any
free slot."""

from collections.abc import Sequence
from pathlib import Path

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.checkpoint_io import run_with_failure_collective, write_checkpoint_dir

_WEIGHTS_KEY = "adapter_weights"
_OPTIM_KEY = "adapter_optimizer"


def _slot_weights_sharded_state_dict(model: Sequence[DDP], slot: int) -> dict:
    sharded: dict = {}
    for chunk in unwrap_model(model):
        sharded |= chunk.sharded_state_dict()
    marker = f".adapters.{slot}."
    slot_sharded = {key: value for key, value in sharded.items() if marker in key}
    assert slot_sharded, f"slot {slot} exposed no adapter tensors"
    return slot_sharded


def _canonicalize_slot_keys(tree: dict, slot: int) -> dict:
    """Rewrite storage keys so a checkpoint saved from slot i loads into slot j."""
    marker, canonical = f".adapters.{slot}.", ".adapter."
    for sharded in nested_values(tree):
        if hasattr(sharded, "key"):
            sharded.key = sharded.key.replace(marker, canonical)
    return tree


def save_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str) -> None:
    weights = _slot_weights_sharded_state_dict(model, slot_optimizer.slot)
    sharded = {_WEIGHTS_KEY: weights, _OPTIM_KEY: slot_optimizer.sharded_state(weights, is_loading=False)}
    _canonicalize_slot_keys(sharded, slot_optimizer.slot)
    write_checkpoint_dir(path, lambda tmp_dir: dist_checkpointing.save(sharded, str(tmp_dir)))


def load_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str, load_optimizer: bool) -> None:
    checkpoint_dir = str(Path(path))
    weights = _slot_weights_sharded_state_dict(model, slot_optimizer.slot)
    shells = {_WEIGHTS_KEY: weights}
    if load_optimizer:
        shells[_OPTIM_KEY] = slot_optimizer.sharded_state(weights, is_loading=True)
    _canonicalize_slot_keys(shells, slot_optimizer.slot)

    def apply_shards():
        # weight tensors load in place; the optimizer state comes back as a dict
        loaded = dist_checkpointing.load(shells, checkpoint_dir)
        if load_optimizer:
            slot_optimizer.load_sharded_state(loaded[_OPTIM_KEY])
        else:
            # weights-only load keeps the fresh Adam state the slot init just created
            slot_optimizer.reload_masters()

    run_with_failure_collective(apply_shards)
