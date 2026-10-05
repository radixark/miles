"""Megatron-facing adapter for fused NVFP4 fake QAT."""

from __future__ import annotations

import os
import weakref

import torch

NVFP4_FAKE_QAT_FLAG = "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"


def _grouped_weight_storage(weight: torch.Tensor) -> torch.Tensor:
    # Import TE only for native grouped weights.
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    if not isinstance(weight, GroupedTensor):
        raise TypeError("Packed NVFP4 fake QAT requires a native TE GroupedTensor.")
    groups, rows, columns = weight.shape
    if (
        weight.quantizer is not None
        or weight.num_tensors != groups
        or weight.logical_shape != (groups * rows, columns)
        or weight.first_dims is not None
        or weight.last_dims is not None
        or weight.tensor_offsets is not None
        or weight.tensor_shapes != [(rows, columns)] * groups
        or weight.offsets not in (None, [i * rows * columns for i in range(groups + 1)])
    ):
        raise ValueError("Packed NVFP4 fake QAT requires uniform, unquantized, densely ordered TE weights.")
    storage = weight.rowwise_data
    if (
        storage is None
        or storage.dtype != weight.dtype
        or storage.device != weight.device
        or storage.numel() != groups * rows * columns
        or not storage.is_contiguous()
    ):
        raise ValueError("Packed NVFP4 fake QAT requires complete contiguous BF16/FP16 rowwise storage.")
    return storage.view(groups, rows, columns)


class _GroupedWeightQDQSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, config):
        from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

        from miles.utils.fused_nvfp4_qdq import compute_grouped_nvfp4_amax, fused_grouped_nvfp4_qdq

        storage = _grouped_weight_storage(weight)
        output = fused_grouped_nvfp4_qdq(storage, compute_grouped_nvfp4_amax(storage), config)
        grouped_output = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=storage.shape[0],
            tensor_shape=tuple(storage.shape[1:]),
            rowwise_data=output,
            dtype=output.dtype,
        )
        # Megatron's leaf hook must see TE's fused-wgrad completion marker.
        for name in ("grad_added_to_main_grad", "zero_out_wgrad", "overwrite_main_grad", "get_main_grad"):
            if hasattr(weight, name):
                setattr(grouped_output, name, getattr(weight, name))
        ctx.weight = weight
        ctx.grouped_output = weakref.ref(grouped_output)
        ctx.set_materialize_grads(False)
        return grouped_output

    @staticmethod
    def backward(ctx, grad_output):
        grouped_output = ctx.grouped_output()
        if grouped_output is not None and getattr(grouped_output, "grad_added_to_main_grad", False):
            ctx.weight.grad_added_to_main_grad = True
        return grad_output, None


def _fake_quantize_grouped_weight(weight, config):
    if hasattr(weight, "rowwise_data"):
        output = _GroupedWeightQDQSTE.apply(weight, config)
        if hasattr(weight, "main_grad"):
            output.main_grad = weight.main_grad
        return output

    from miles.utils.fused_nvfp4_qdq import fake_grouped_nvfp4_quantization_ste

    return fake_grouped_nvfp4_quantization_ste(weight, config)


def maybe_fake_quantize_nvfp4_weight_tensors(
    weight_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Apply fake QAT to discrete or native packed TE weights."""
    if os.getenv(NVFP4_FAKE_QAT_FLAG, "0") != "1":
        return weight_tensors

    # Keep CuTe DSL optional for every process that does not enable this path.
    from miles.utils.fused_nvfp4_qdq import current_nvfp4_qdq_config, fake_nvfp4_quantization_ste

    qdq_config = current_nvfp4_qdq_config()
    return [
        (
            _fake_quantize_grouped_weight(weight, qdq_config)
            if weight.ndim == 3
            else fake_nvfp4_quantization_ste(weight, qdq_config)
        )
        for weight in weight_tensors
    ]


__all__ = ["NVFP4_FAKE_QAT_FLAG", "maybe_fake_quantize_nvfp4_weight_tensors"]
