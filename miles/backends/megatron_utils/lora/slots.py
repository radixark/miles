"""Slot mechanics for multi-LoRA: model instrumentation, shard addressing, and
per-slot optimizer state hygiene. Kept deliberately free of any control-plane
or data-plane logic."""

import logging
from argparse import Namespace

import torch

logger = logging.getLogger(__name__)


def create_multi_lora_instance(args: Namespace, *, target_modules):
    from megatron.bridge.peft.multi_lora import MultiLoRA

    return MultiLoRA(
        target_modules=target_modules,
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=args.lora_A_init_method,
        lora_B_init_method=args.lora_B_init_method,
    )


def _lora_rank_dim(hf_name: str, ndim: int) -> int | None:
    """The rank axis of a LoRA tensor, addressed from the end so packed grouped-expert
    tensors are not sliced on the expert axis; None for a non-LoRA name."""
    if "lora_A" in hf_name:
        return ndim - 2
    if "lora_B" in hf_name:
        return ndim - 1
    return None


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Trim a max-rank-padded LoRA tensor to ``adapter_rank`` on the rank axis."""
    rank_dim = _lora_rank_dim(hf_name, tensor.ndim)
    if rank_dim is None or adapter_rank >= tensor.shape[rank_dim]:
        return tensor
    remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
    assert remainder.abs().max() == 0, (
        f"LoRA padded dims are non-zero: {hf_name}, "
        f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
    )
    return tensor.narrow(rank_dim, 0, adapter_rank)


def pad_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int, max_rank: int) -> torch.Tensor:
    """Inverse of ``slice_lora_to_rank``: zero-pad a rank-``adapter_rank`` LoRA tensor to ``max_rank``."""
    rank_dim = _lora_rank_dim(hf_name, tensor.ndim)
    assert (
        rank_dim is not None and tensor.shape[rank_dim] == adapter_rank
    ), f"{hf_name} with shape {tuple(tensor.shape)} is not a rank-{adapter_rank} LoRA tensor"
    padding = [0, 0] * (tensor.ndim - 1 - rank_dim) + [0, max_rank - adapter_rank]
    return torch.nn.functional.pad(tensor, padding)
