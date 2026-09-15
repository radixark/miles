"""Slot mechanics for multi-LoRA: model instrumentation, shard addressing, and
per-slot optimizer state hygiene. Kept deliberately free of any control-plane
or data-plane logic."""

import logging
from argparse import Namespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

# Cached by adapter_shard_topology(); the topology is fixed for the run.
_shard_topology: tuple[bool, tuple[tuple[int, int, int], ...]] | None = None


def create_multi_lora_instance(args: Namespace):
    """Create a MultiLoRA instance from training args."""
    from megatron.bridge.peft.multi_lora import MultiLoRA

    from miles.backends.megatron_utils.lora.utils import convert_target_modules_to_megatron

    lora_type_name = getattr(args, "lora_type", "lora").lower()
    if lora_type_name == "canonical_lora":
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA

        lora_cls = CanonicalLoRA
    else:
        from megatron.bridge.peft.lora import LoRA

        lora_cls = LoRA

    # exclude_modules was already folded into target_modules during arg validation.
    return MultiLoRA(
        target_modules=convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls),
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def megatron_shard_name(tp_rank: int, pp_rank: int, ep_rank: int, ep_size: int) -> str:
    """Adapter shard name for one (tp, pp, ep) coordinate; EP ranks hold different local
    experts. The ep suffix is omitted at ep_size == 1 so legacy checkpoints stay loadable."""
    name = f"adapter_megatron_tp{tp_rank}_pp{pp_rank}"
    if ep_size > 1:
        name += f"_ep{ep_rank}"
    return name + ".pt"


def adapter_shard_topology() -> tuple[bool, tuple[tuple[int, int, int], ...]]:
    """Return ``(this_rank_writes_its_shard, realized (tp, pp, ep) coords)`` via one cached gloo all-gather."""
    global _shard_topology
    if _shard_topology is not None:
        return _shard_topology
    parallel_state = get_parallel_state()
    coords = (parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank)
    if not dist.is_initialized():
        _shard_topology = (True, (coords,))
        return _shard_topology

    current_rank = dist.get_rank()
    group = get_gloo_group()
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, (coords, current_rank), group=group)
    is_writer = current_rank == min(rank for entry_coords, rank in gathered if entry_coords == coords)
    _shard_topology = (is_writer, tuple(sorted({entry_coords for entry_coords, _ in gathered})))
    return _shard_topology


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Trim a max-rank-padded LoRA tensor to ``adapter_rank`` on the rank axis, addressed
    from the end so packed grouped-expert exports are not sliced on the expert axis."""
    if "lora_A" in hf_name:
        rank_dim = tensor.ndim - 2
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_A padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    if "lora_B" in hf_name:
        rank_dim = tensor.ndim - 1
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_B padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    return tensor
