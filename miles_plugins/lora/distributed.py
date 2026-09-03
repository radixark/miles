"""TP/EP/SP communication and gradient handling for Miles-native LoRA."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from miles_plugins.lora.spec.base import AttachContext


def rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool = False,
) -> torch.Tensor:
    """Recompute RMSNorm fused into a TE column-parallel linear."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    weight = gamma.float() + 1.0 if zero_centered_gamma else gamma.float()
    return (normed * weight).to(x.dtype)


def apply_lora_dropout(x: torch.Tensor, context: AttachContext, training: bool) -> torch.Tensor:
    if context.dropout and training:
        return F.dropout(x, p=context.dropout, training=True)
    return x


def branch_input(x: torch.Tensor, module: nn.Module, context: AttachContext) -> torch.Tensor:
    """Return the input expected by a column-parallel LoRA branch.

    This mirrors the wrapped MCore/TE module's fused RMSNorm and TP/SP input
    mapping without moving those architecture details into the attachment spec.
    """
    gamma = getattr(module, "layer_norm_weight", None)
    if gamma is not None:
        x = rmsnorm(x, gamma, context.eps, context.zero_centered_gamma)
    if context.sequence_parallel:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        x = gather_from_sequence_parallel_region(x)
    elif context.tp_size > 1:
        from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region

        x = copy_to_tensor_model_parallel_region(x)
    return apply_lora_dropout(x, context, module.training)


def reduce_row_parallel(partial: torch.Tensor, context: AttachContext) -> torch.Tensor:
    """Complete a row-parallel adapter branch from each rank's partial sum."""
    if context.tp_size <= 1:
        return partial
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    if context.sequence_parallel:
        return reduce_scatter_to_sequence_parallel_region(partial)
    return reduce_from_tensor_model_parallel_region(partial)


class ParallelGather:
    """Batch per-tensor all-gathers into one flat collective per parallel group.

    ``group`` selects the domain the tensor is sharded over: ``"tp"`` for the
    attention tensor-parallel group, ``"ep"`` for the expert-parallel group
    (expert-axis adapter tensors).
    """

    _GROUPS = ("tp", "ep")

    def __init__(self):
        self._requests: dict[str, list[tuple[torch.Tensor, int]]] = {kind: [] for kind in self._GROUPS}
        self._resolved: dict[str, list[torch.Tensor]] | None = None

    def request(self, local: torch.Tensor, cat_dim: int, group: str = "tp") -> Callable[[], torch.Tensor]:
        assert group in self._GROUPS, f"unknown gather group {group!r}"
        index = len(self._requests[group])
        self._requests[group].append((local, cat_dim))
        return lambda: self._resolve(group, index)

    def _resolve(self, group: str, index: int) -> torch.Tensor:
        assert self._resolved is not None, "ParallelGather.flush() must run before resolving requests"
        return self._resolved[group][index]

    def flush(self) -> None:
        self._resolved = {}
        if not dist.is_initialized():
            for kind in self._GROUPS:
                self._resolved[kind] = [local for local, _ in self._requests[kind]]
            return

        from megatron.core import parallel_state as ps

        domains = {
            "tp": (ps.get_tensor_model_parallel_group, ps.get_tensor_model_parallel_world_size),
            "ep": (ps.get_expert_model_parallel_group, ps.get_expert_model_parallel_world_size),
        }
        for kind in self._GROUPS:
            requests = self._requests[kind]
            get_group, get_world = domains[kind]
            world = get_world() if requests else 1
            if not requests or world == 1:
                self._resolved[kind] = [local for local, _ in requests]
                continue
            assert len({local.dtype for local, _ in requests}) == 1, "mixed adapter dtypes"
            flats = [local.detach().contiguous().reshape(-1) for local, _ in requests]
            sizes = [flat.numel() for flat in flats]
            local_flat = torch.cat(flats)
            gathered = local_flat.new_empty(world * local_flat.numel())
            dist.all_gather_into_tensor(gathered, local_flat, group=get_group())
            per_rank = gathered.view(world, -1)

            resolved = []
            offset = 0
            for (local, cat_dim), size in zip(requests, sizes, strict=True):
                shards = [per_rank[rank, offset : offset + size].view(local.shape) for rank in range(world)]
                resolved.append(torch.cat(shards, dim=cat_dim))
                offset += size
            self._resolved[kind] = resolved


TensorParallelGather = ParallelGather


def reduce_marked_lora_grads(model: Sequence[nn.Module]) -> None:
    """Sum partial gradients for replicated native-LoRA parameters."""
    from megatron.core import parallel_state as ps

    if not model:
        return
    marked = getattr(model[0], "_miles_lora_marked_grad_params", None)
    if marked is None:
        marked = []
        for chunk in model:
            for param in chunk.parameters():
                group_name = getattr(param, "_lora_grad_sum_group", None)
                if group_name is not None and param.requires_grad:
                    marked.append((param, group_name))
        model[0]._miles_lora_marked_grad_params = marked
    if not marked:
        return

    groups = {
        "tp": (ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_world_size()),
        "ep": (ps.get_expert_model_parallel_group(), ps.get_expert_model_parallel_world_size()),
    }
    for group_name in ("tp", "ep"):
        group, size = groups[group_name]
        if size <= 1:
            continue
        grads = []
        for param, parameter_group_name in marked:
            if parameter_group_name != group_name:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads.append(grad)
        for dtype in {grad.dtype for grad in grads}:
            matching_grads = [grad for grad in grads if grad.dtype == dtype]
            if len(matching_grads) == 1:
                dist.all_reduce(matching_grads[0], op=dist.ReduceOp.SUM, group=group)
                continue
            flat = torch._utils._flatten_dense_tensors(matching_grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            reduced = torch._utils._unflatten_dense_tensors(flat, matching_grads)
            for grad, reduced_grad in zip(matching_grads, reduced, strict=False):
                grad.copy_(reduced_grad)
