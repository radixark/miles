"""Inference-side MoE arithmetic used by the DSV4 TOP source contract."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch

_CONTRACT_TOKENS = 96
_CONTRACT_TOPK = 6
_CONTRACT_HIDDEN_SIZE = 4096
_CONTRACT_NUM_EXPERTS = 32
_FP8_BLOCK_SIZE = 128
_GROUPED_GEMM_ALIGNMENT = 16


@dataclass
class Dsv4TopMoeContext:
    """Request-local state shared with SGLang's fused-MoE call stack."""

    active: bool
    topk_ids: torch.Tensor | None = None
    routed_unscaled: torch.Tensor | None = None


_CURRENT_MOE_CONTEXT: ContextVar[Dsv4TopMoeContext | None] = ContextVar(
    "miles_dsv4_top_moe_context",
    default=None,
)


@contextmanager
def dsv4_top_moe_context(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> Iterator[Dsv4TopMoeContext]:
    """Activate the version-pinned four-layer/96-token MoE contract."""
    active = layer_id in (0, 1, 2, 3) and hidden_states.shape == (_CONTRACT_TOKENS, _CONTRACT_HIDDEN_SIZE)
    if active and topk_ids.shape != (
        _CONTRACT_TOKENS,
        _CONTRACT_TOPK,
    ):
        raise RuntimeError(
            "DSV4 TOP MoE contract expected top-k ids with shape "
            f"{(_CONTRACT_TOKENS, _CONTRACT_TOPK)}, got "
            f"{tuple(topk_ids.shape)}"
        )

    context = Dsv4TopMoeContext(
        active=active,
        topk_ids=topk_ids.clone() if active else None,
    )
    token = _CURRENT_MOE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_MOE_CONTEXT.reset(token)


def get_dsv4_top_moe_context() -> Dsv4TopMoeContext | None:
    return _CURRENT_MOE_CONTEXT.get()


def _validate_expert_ids(
    expert_ids: torch.Tensor,
    *,
    num_experts: int,
) -> None:
    valid = (expert_ids == -1) | ((expert_ids >= 0) & (expert_ids < num_experts))
    if not bool(torch.all(valid).item()):
        raise RuntimeError(
            "DSV4 TOP MoE contract received invalid expert ids: "
            f"min={expert_ids.min().item()}, max={expert_ids.max().item()}, "
            f"num_experts={num_experts}"
        )


def _wrap_block_fp8(
    data: torch.Tensor,
    scale_inv: torch.Tensor,
    quantizer,
    *,
    is_2d_scaled: bool,
):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockwiseQTensor,
    )

    data = data.contiguous()
    scale_inv = scale_inv.contiguous()
    if data.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"DSV4 TOP expected FP8 E4M3 data, got {data.dtype}")
    if scale_inv.dtype != torch.float32:
        raise RuntimeError("DSV4 TOP expected FP32 inverse scales, got " f"{scale_inv.dtype}")

    expected_scale_shape = tuple(quantizer.get_scale_shape(data.shape, columnwise=False))
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise RuntimeError(
            "DSV4 TOP FP8 scale shape mismatch: "
            f"data={tuple(data.shape)}, scale={tuple(scale_inv.shape)}, "
            f"expected={expected_scale_shape}"
        )

    return Float8BlockwiseQTensor(
        shape=tuple(data.shape),
        dtype=torch.bfloat16,
        device=data.device,
        requires_grad=False,
        rowwise_data=data.view(torch.uint8),
        rowwise_scale_inv=scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=quantizer,
        is_2D_scaled=is_2d_scaled,
    )


def _te_grouped_gemm_by_slot(
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run trainer-compatible TE grouped GEMM and restore slot order."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_grouped_gemm,
    )
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    if inputs.ndim != 2 or weights.ndim != 3:
        raise RuntimeError(
            "DSV4 TOP grouped GEMM expected 2D inputs and 3D weights, "
            f"got {tuple(inputs.shape)} and {tuple(weights.shape)}"
        )
    num_slots, input_size = inputs.shape
    num_experts, output_size, weight_input_size = weights.shape
    if input_size != weight_input_size:
        raise RuntimeError("DSV4 TOP grouped GEMM input/weight K mismatch: " f"{input_size} vs {weight_input_size}")
    if num_experts != _CONTRACT_NUM_EXPERTS:
        raise RuntimeError("DSV4 TOP grouped GEMM expected " f"{_CONTRACT_NUM_EXPERTS} experts, got {num_experts}")
    if input_size % _FP8_BLOCK_SIZE or output_size % _FP8_BLOCK_SIZE:
        raise RuntimeError(
            "DSV4 TOP grouped GEMM dimensions must be divisible by "
            f"{_FP8_BLOCK_SIZE}: input={input_size}, output={output_size}"
        )
    if expert_ids.shape != (num_slots,):
        raise RuntimeError(
            "DSV4 TOP grouped GEMM expert-id shape mismatch: "
            f"expected {(num_slots,)}, got {tuple(expert_ids.shape)}"
        )
    if weights.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"DSV4 TOP expected FP8 weights, got {weights.dtype}")
    expected_weight_scale_shape = (
        num_experts,
        output_size // _FP8_BLOCK_SIZE,
        input_size // _FP8_BLOCK_SIZE,
    )
    if weight_scale.shape != expected_weight_scale_shape:
        raise RuntimeError(
            "DSV4 TOP weight-scale shape mismatch: "
            f"expected {expected_weight_scale_shape}, got "
            f"{tuple(weight_scale.shape)}"
        )
    if weight_scale.dtype != torch.float32:
        raise RuntimeError(f"DSV4 TOP expected FP32 weight scales, got {weight_scale.dtype}")
    _validate_expert_ids(expert_ids, num_experts=num_experts)

    quantized_inputs, input_scales = sglang_per_token_group_quant_fp8(
        inputs.contiguous(),
        _FP8_BLOCK_SIZE,
        scale_ue8m0=True,
    )
    expected_input_scale_shape = (
        num_slots,
        input_size // _FP8_BLOCK_SIZE,
    )
    if input_scales.shape != expected_input_scale_shape:
        raise RuntimeError(
            "DSV4 TOP input-scale shape mismatch: "
            f"expected {expected_input_scale_shape}, got "
            f"{tuple(input_scales.shape)}"
        )
    if input_scales.dtype != torch.float32:
        raise RuntimeError(f"DSV4 TOP expected FP32 input scales, got {input_scales.dtype}")

    input_quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=True,
        block_scaling_dim=1,
    )
    input_quantizer.internal = True
    input_quantizer.optimize_for_gemm = True
    weight_quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=True,
        block_scaling_dim=2,
    )
    weight_quantizer.internal = True

    input_qtensors = []
    weight_qtensors = []
    slot_indices_by_expert = []
    counts = []
    padded_counts = []
    for expert in range(num_experts):
        slot_indices = torch.nonzero(
            expert_ids == expert,
            as_tuple=False,
        ).flatten()
        count = int(slot_indices.numel())
        padded_count = (count + _GROUPED_GEMM_ALIGNMENT - 1) // _GROUPED_GEMM_ALIGNMENT * _GROUPED_GEMM_ALIGNMENT
        counts.append(count)
        padded_counts.append(padded_count)
        slot_indices_by_expert.append(slot_indices)

        expert_inputs = torch.zeros(
            (padded_count, input_size),
            dtype=torch.float8_e4m3fn,
            device=inputs.device,
        )
        expert_scale_shape = tuple(
            input_quantizer.get_scale_shape(
                expert_inputs.shape,
                columnwise=False,
            )
        )
        expected_expert_scale_shape = (
            input_size // _FP8_BLOCK_SIZE,
            padded_count,
        )
        if expert_scale_shape != expected_expert_scale_shape:
            raise RuntimeError(
                "DSV4 TOP padded input-scale shape mismatch: "
                f"expected {expected_expert_scale_shape}, got "
                f"{expert_scale_shape}"
            )
        expert_scales = torch.ones(
            expert_scale_shape,
            dtype=torch.float32,
            device=inputs.device,
        )
        if count:
            expert_inputs[:count].copy_(quantized_inputs.index_select(0, slot_indices))
            expert_scales[:, :count].copy_(
                input_scales.index_select(
                    0,
                    slot_indices,
                ).transpose(0, 1)
            )
        input_qtensors.append(
            _wrap_block_fp8(
                expert_inputs,
                expert_scales,
                input_quantizer,
                is_2d_scaled=False,
            )
        )
        weight_qtensors.append(
            _wrap_block_fp8(
                weights[expert],
                weight_scale[expert],
                weight_quantizer,
                is_2d_scaled=True,
            )
        )

    if sum(counts) <= 0:
        raise RuntimeError("DSV4 TOP grouped GEMM received no valid routes")

    padded_output = torch.empty(
        (sum(padded_counts), output_size),
        dtype=torch.bfloat16,
        device=inputs.device,
    )
    result = general_grouped_gemm(
        A=weight_qtensors,
        B=input_qtensors,
        out=[padded_output],
        quantization_params=[None] * num_experts,
        out_dtype=torch.bfloat16,
        layout="TN",
        m_splits=padded_counts,
        gelu=False,
        grad=False,
        accumulate=False,
        bias=None,
        use_bias=False,
        use_split_accumulator=True,
        D_dtype=None,
        single_output=True,
    )
    if result[0][0].data_ptr() != padded_output.data_ptr():
        raise RuntimeError("DSV4 TOP TE grouped GEMM did not write into the supplied output")

    output_by_slot = torch.zeros(
        (num_slots, output_size),
        dtype=torch.bfloat16,
        device=inputs.device,
    )
    cursor = 0
    for count, padded_count, slot_indices in zip(
        counts,
        padded_counts,
        slot_indices_by_expert,
        strict=True,
    ):
        if count:
            output_by_slot.index_copy_(
                0,
                slot_indices,
                padded_output[cursor : cursor + count],
            )
        cursor += padded_count
    if cursor != padded_output.shape[0]:
        raise RuntimeError("DSV4 TOP grouped GEMM output cursor mismatch: " f"{cursor} vs {padded_output.shape[0]}")
    return output_by_slot


def replace_fused_moe_fc1(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    intermediate_cache1: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    use_fp8_w8a8: bool,
    block_shape,
    bias: torch.Tensor | None,
) -> None:
    """Replace SGLang FC1 with the trainer-compatible TE grouped GEMM."""
    if not use_fp8_w8a8 or block_shape != [128, 128] or bias is not None:
        raise RuntimeError("DSV4 TOP FC1 requires FP8 block scaling without bias")
    if hidden_states.shape != (
        _CONTRACT_TOKENS,
        _CONTRACT_HIDDEN_SIZE,
    ):
        raise RuntimeError(f"DSV4 TOP FC1 hidden shape mismatch: {hidden_states.shape}")
    if topk_ids.shape != (_CONTRACT_TOKENS, _CONTRACT_TOPK):
        raise RuntimeError(f"DSV4 TOP FC1 top-k shape mismatch: {topk_ids.shape}")

    num_slots = _CONTRACT_TOKENS * _CONTRACT_TOPK
    flat_hidden = (
        hidden_states.unsqueeze(1)
        .expand(
            _CONTRACT_TOKENS,
            _CONTRACT_TOPK,
            _CONTRACT_HIDDEN_SIZE,
        )
        .reshape(num_slots, _CONTRACT_HIDDEN_SIZE)
        .contiguous()
    )
    output_by_slot = _te_grouped_gemm_by_slot(
        flat_hidden,
        topk_ids.reshape(-1).long(),
        w1,
        w1_scale,
    )
    if intermediate_cache1.shape[0] < num_slots:
        raise RuntimeError("DSV4 TOP FC1 cache is too small: " f"{tuple(intermediate_cache1.shape)}")
    intermediate_cache1[:num_slots].copy_(output_by_slot)


def replace_fused_moe_fc2(
    *,
    topk_ids: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    intermediate_cache3: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    use_fp8_w8a8: bool,
    block_shape,
    bias: torch.Tensor | None,
) -> None:
    """Replace SGLang FC2 with the trainer-compatible TE grouped GEMM."""
    if not use_fp8_w8a8 or block_shape != [128, 128] or bias is not None:
        raise RuntimeError("DSV4 TOP FC2 requires FP8 block scaling without bias")
    if topk_ids.shape != (_CONTRACT_TOKENS, _CONTRACT_TOPK):
        raise RuntimeError(f"DSV4 TOP FC2 top-k shape mismatch: {topk_ids.shape}")
    expected_output_shape = (
        _CONTRACT_TOKENS,
        _CONTRACT_TOPK,
        _CONTRACT_HIDDEN_SIZE,
    )
    if intermediate_cache3.shape != expected_output_shape:
        raise RuntimeError(
            "DSV4 TOP FC2 output-cache shape mismatch: "
            f"expected {expected_output_shape}, got "
            f"{tuple(intermediate_cache3.shape)}"
        )

    num_slots = _CONTRACT_TOKENS * _CONTRACT_TOPK
    if intermediate_cache2.shape[0] < num_slots:
        raise RuntimeError("DSV4 TOP FC2 input cache is too small: " f"{tuple(intermediate_cache2.shape)}")
    output_by_slot = _te_grouped_gemm_by_slot(
        intermediate_cache2[:num_slots].contiguous(),
        topk_ids.reshape(-1).long(),
        w2,
        w2_scale,
    )
    intermediate_cache3.copy_(output_by_slot.view(expected_output_shape))


def record_fused_moe_trainer_order(
    intermediate_cache3: torch.Tensor,
) -> None:
    """Reduce TP slots and reproduce the trainer's stable expert ordering."""
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.tp_invariant_ops import tree_all_reduce_sum

    context = get_dsv4_top_moe_context()
    if context is None or not context.active or context.topk_ids is None:
        raise RuntimeError("DSV4 TOP routed combine ran without an active MoE context")
    expected_shape = (
        _CONTRACT_TOKENS,
        _CONTRACT_TOPK,
        _CONTRACT_HIDDEN_SIZE,
    )
    if intermediate_cache3.shape != expected_shape:
        raise RuntimeError(
            "DSV4 TOP routed combine shape mismatch: "
            f"expected {expected_shape}, got "
            f"{tuple(intermediate_cache3.shape)}"
        )

    global_weighted_by_slot = tree_all_reduce_sum(
        intermediate_cache3,
        device_group=get_tp_group().device_group,
    )
    token_ids = (
        torch.arange(
            _CONTRACT_TOKENS,
            dtype=torch.long,
            device=context.topk_ids.device,
        )
        .unsqueeze(1)
        .expand(_CONTRACT_TOKENS, _CONTRACT_TOPK)
        .reshape(-1)
    )
    expert_ids = context.topk_ids.reshape(-1).long()
    order = torch.argsort(
        expert_ids * _CONTRACT_TOKENS + token_ids,
        stable=True,
    )
    sorted_tokens = token_ids[order]
    sorted_weighted = global_weighted_by_slot.reshape(
        _CONTRACT_TOKENS * _CONTRACT_TOPK,
        _CONTRACT_HIDDEN_SIZE,
    )[order]
    trainer_order = torch.zeros(
        (_CONTRACT_TOKENS, _CONTRACT_HIDDEN_SIZE),
        dtype=sorted_weighted.dtype,
        device=sorted_weighted.device,
    )

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        trainer_order.index_add_(
            0,
            sorted_tokens,
            sorted_weighted,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    context.routed_unscaled = trainer_order
