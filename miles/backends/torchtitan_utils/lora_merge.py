"""The frozen base with the trained adapter merged in, which --lora-merge-for-rollout engines serve."""

import functools
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torchtitan.models.common.linear import GroupedLinear, Linear

from miles.backends.fsdp_utils.dtensor import gather_full_param


def merged_host_weights(model_parts: Sequence[nn.Module], *, scaling: float) -> dict[str, Callable[[], torch.Tensor]]:
    """``{host weight key: () -> full W + scaling * B @ A}`` for every adapted projection.

    Each call gathers the host weight the way the base export does and adds the adapter in place on
    that full tensor, so the merge costs no memory beyond the gather.
    """
    return {
        f"{name}.weight": functools.partial(_merged_full_weight, module, scaling)
        for part in model_parts
        for name, module in part.named_modules()
        if isinstance(module, (Linear, GroupedLinear)) and "lora_b" in dict(module.named_children())
    }


@torch.no_grad()
def _merged_full_weight(module: nn.Module, scaling: float) -> torch.Tensor:
    weight = gather_full_param(module.weight)
    if weight.data_ptr() == _local(module.weight).data_ptr():
        weight = weight.clone()  # an unsharded weight gathers to itself, and the base must stay frozen
    lora_a = gather_full_param(module.lora_a.weight).to(weight.dtype)
    lora_b = gather_full_param(module.lora_b.weight).to(weight.dtype)
    # One batched update over the expert axis (a dense projection is one expert): the host flattens
    # to [E, (n *) O, I]; a per-expert factor keeps E, a shared one is expanded to it.
    grouped = isinstance(module, GroupedLinear)
    experts = weight.shape[0] if grouped else 1
    host_EOI = weight.flatten(1, -2) if grouped else weight.flatten(0, -2).unsqueeze(0)
    if grouped and lora_b.dim() == weight.dim():
        lora_b_EOL = lora_b.flatten(1, -2)
    else:
        lora_b_EOL = lora_b.flatten(0, -2).unsqueeze(0).expand(experts, -1, -1)
    lora_a_ELI = lora_a if lora_a.dim() == 3 else lora_a.unsqueeze(0).expand(experts, -1, -1)
    host_EOI.baddbmm_(lora_b_EOL, lora_a_ELI, alpha=scaling)
    return weight


def _local(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if isinstance(param, DTensor) else param
