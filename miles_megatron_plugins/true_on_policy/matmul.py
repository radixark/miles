from __future__ import annotations

from collections.abc import Iterable

import torch

from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)

_ROW_LINEAR_INV_BLOCK_K = 128


def _fixed_tree_sum_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum tensors in the same fixed pairwise order as SGLang."""
    partials = list(tensors)
    if not partials:
        raise ValueError("at least one tensor is required")

    while len(partials) > 1:
        next_partials = []
        for index in range(0, len(partials), 2):
            if index + 1 < len(partials):
                next_partials.append(partials[index] + partials[index + 1])
            else:
                next_partials.append(partials[index])
        partials = next_partials

    return partials[0]


def _safe_group_size(group: torch.distributed.ProcessGroup | None) -> int:
    if group is not None:
        return group.size()
    try:
        from megatron.core.parallel_state import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _safe_tensor_context_parallel_size() -> int:
    try:
        from megatron.core.parallel_state import get_tensor_and_context_parallel_world_size

        return get_tensor_and_context_parallel_world_size()
    except Exception:
        return _safe_group_size(None)


def _rollout_row_parallel_partition_k(input_: torch.Tensor, tp_group: torch.distributed.ProcessGroup | None) -> int:
    train_tp_size = _safe_group_size(tp_group)
    rollout_tp_size = _safe_tensor_context_parallel_size()
    global_k_size = input_.shape[-1] * train_tp_size
    if rollout_tp_size <= 0 or global_k_size % rollout_tp_size != 0:
        return input_.shape[-1]
    return global_k_size // rollout_tp_size


def _should_use_sglang_tp_invariant_row_linear(
    input_: torch.Tensor, row_parallel: bool, tp_group: torch.distributed.ProcessGroup | None
) -> bool:
    rollout_partition_k = _rollout_row_parallel_partition_k(input_, tp_group)
    return (
        row_parallel
        and rollout_partition_k >= _ROW_LINEAR_INV_BLOCK_K
        and rollout_partition_k % _ROW_LINEAR_INV_BLOCK_K == 0
    )


def _sglang_row_parallel_matmul(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """SGLang's row-linear TP-invariant matmul contract.

    SGLang chunks the K dimension into 128-wide products, casts each product to
    the input dtype, then combines those partials with a fixed binary tree.
    Mirroring that order is required before the TP tree all-reduce can be
    bitwise identical.
    """
    input_shape = input_.shape
    input_2d = input_.reshape(-1, input_shape[-1])
    weight_t = weight.t()
    partials = []

    for start in range(0, input_2d.shape[1], _ROW_LINEAR_INV_BLOCK_K):
        end = min(start + _ROW_LINEAR_INV_BLOCK_K, input_2d.shape[1])
        partials.append(input_2d[:, start:end] @ weight_t[start:end, :])

    output = _fixed_tree_sum_tensors(partials).to(input_.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], weight.shape[0])


def _sglang_rollout_partition_row_parallel_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    """Mirror SGLang rollout row-linear shards when train TP is smaller than rollout TP."""
    rollout_partition_k = _rollout_row_parallel_partition_k(input_, tp_group)
    if (
        rollout_partition_k <= 0
        or rollout_partition_k >= input_.shape[-1]
        or input_.shape[-1] % rollout_partition_k != 0
    ):
        return _linear_reference_matmul(input_, weight, bias)

    input_shape = input_.shape
    input_2d = input_.reshape(-1, input_shape[-1])
    weight_t = weight.t()
    partials = []

    for start in range(0, input_2d.shape[1], rollout_partition_k):
        end = start + rollout_partition_k
        partials.append(input_2d[:, start:end] @ weight_t[start:end, :])

    output = _fixed_tree_sum_tensors(partials).to(input_.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], weight.shape[0])


def _linear_reference_matmul(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    output = input_.reshape(-1, input_.shape[-1]) @ weight.t()
    output = output.reshape(*input_.shape[:-1], weight.shape[0])
    if bias is not None:
        output = output + bias
    return output


def sglang_reference_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None = None,
    wgrad_deferral_limit: int | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    row_parallel: bool = False,
) -> torch.Tensor:
    """Reference TP matmul entrypoint for the SGLang-compatible backend.

    PR 6 keeps Megatron on the same local numerical path by default and introduces a
    single surface that later PRs can specialize for TP-invariant ordering. The
    implementation intentionally delegates to the existing Megatron kernels so enabling
    the backend flag does not yet change the training contract.
    """

    if input_.dtype != weight.dtype:
        input_ = input_.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)

    if _should_use_sglang_tp_invariant_row_linear(input_, row_parallel, tp_group):
        return _sglang_row_parallel_matmul(input_, weight, bias)
    if row_parallel:
        return _sglang_rollout_partition_row_parallel_matmul(input_, weight, bias, tp_group=tp_group)

    if weight.requires_grad:
        return linear_with_grad_accumulation_and_async_allreduce(
            input=input_,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            grad_output_buffer=grad_output_buffer,
            wgrad_deferral_limit=wgrad_deferral_limit or 0,
            tp_group=tp_group,
        )

    return linear_with_frozen_weight(
        input=input_,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=sequence_parallel,
        grad_output_buffer=grad_output_buffer,
        wgrad_deferral_limit=wgrad_deferral_limit,
        tp_group=tp_group,
    )
