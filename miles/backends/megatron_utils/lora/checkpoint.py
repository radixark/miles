"""Per-slot checkpoints: adapter weights plus the slot's optimizer state.

Weight shards are (tp, pp, ep)-addressed and slot-agnostic (saved under
expose_adapter_slot). Optimizer state is per global rank because LayerWise
scatters whole params across ranks; resume requires the same world topology.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.megatron_utils.lora.slots import adapter_shard_topology, megatron_shard_name
from miles.backends.training_utils.checkpoint_io import run_checkpoint_phase, write_checkpoint_dir
from miles.backends.training_utils.parallel import get_parallel_state


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _weight_shard_name() -> str:
    parallel_state = get_parallel_state()
    return megatron_shard_name(
        parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank, parallel_state.ep.size
    )


def _optim_shard_name() -> str:
    return f"optim_rank{_rank()}.pt"


def save_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str) -> None:
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    is_shard_writer, _ = adapter_shard_topology()

    def write_shards(tmp_dir: Path):
        if is_shard_writer:
            with expose_adapter_slot(model, slot_optimizer.slot):
                shard = {
                    name: param.data.cpu()
                    for model_chunk in model
                    for name, param in model_chunk.named_parameters()
                    if ".adapter." in name
                }
            assert shard, f"slot {slot_optimizer.slot} exposed no adapter tensors"
            torch.save(shard, tmp_dir / _weight_shard_name())
        torch.save(slot_optimizer.state(), tmp_dir / _optim_shard_name())

    write_checkpoint_dir(path, write_shards)


def load_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str, load_optimizer: bool) -> None:
    from megatron.bridge.peft.multi_lora_layers import load_adapter

    checkpoint_dir = Path(path)
    shards: dict = {}

    def read_shards():
        shards["weights"] = torch.load(checkpoint_dir / _weight_shard_name(), map_location="cpu", weights_only=True)
        if load_optimizer:
            optim_state = torch.load(checkpoint_dir / _optim_shard_name(), map_location="cpu", weights_only=True)
            slot_optimizer.validate_state(optim_state)
            shards["optim"] = optim_state

    def apply_shards():
        loaded = load_adapter(model, slot_optimizer.slot, shards["weights"])
        assert loaded > 0, f"loaded 0 adapter tensors from {checkpoint_dir / _weight_shard_name()}"
        slot_optimizer.reload_masters()
        if load_optimizer:
            slot_optimizer.load_state(shards["optim"])
        # weights-only load keeps the fresh Adam state the slot init just created

    # every rank validates its shards before any rank touches the live slot
    run_checkpoint_phase(read_shards)
    run_checkpoint_phase(apply_shards)
