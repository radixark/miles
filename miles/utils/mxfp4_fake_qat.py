"""Megatron-facing adapter for MXFP4 fake QAT of grouped MoE expert weights.

Megatron's ``TEGroupedLinear._get_weight_tensors`` calls :func:`fake_quantize_mxfp4_weight_tensors`
when ``TransformerConfig.moe_expert_fake_quant`` is ``"mxfp4"``. The expert weights are rounded onto
the MXFP4 grid in place. Megatron's mixed-precision optimizer updates fp32 master weights and copies
them into these BF16 parameters after every step, so rounding the parameters changes what the GEMMs
see but not what the optimizer updates: the weight gradient reaches the master weights unchanged, as
with a straight-through estimator. Transformer Engine receives the parameters themselves, so fused
weight-gradient accumulation into ``main_grad`` works as for any other weight and no rounded copy is
kept alive.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence

import torch

from miles.kernels.quant.fake_quant import fake_quant_fp4_forward

MOE_EXPERT_FAKE_QUANT_FIELD = "moe_expert_fake_quant"
MXFP4 = "mxfp4"
_OTHER_FAKE_QAT_FLAGS = ("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "OPEN_TRAINING_INT4_FAKE_QAT_FLAG")


def fake_quantize_mxfp4_weight_tensors(module, weight_tensors: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """Round a grouped linear's expert weights onto the MXFP4 grid in place, once per forward-backward pass."""
    return _round_in_place_once_per_pass(module, weight_tensors, round_in_place=_round_to_mxfp4_in_place)


def validate_moe_expert_fake_quant_config(config) -> None:
    """Reject model configurations in which rounding the grouped-expert weights would be skipped or wrong."""
    if MOE_EXPERT_FAKE_QUANT_FIELD not in {field.name for field in dataclasses.fields(config)}:
        raise ValueError(
            f"This Megatron-LM's TransformerConfig has no {MOE_EXPERT_FAKE_QUANT_FIELD} field, so its "
            "TEGroupedLinear cannot apply routed-expert fake QAT."
        )
    if config.moe_expert_fake_quant != MXFP4:
        raise ValueError(f"Unsupported {MOE_EXPERT_FAKE_QUANT_FIELD}={config.moe_expert_fake_quant!r}.")
    if not getattr(
        config, "moe_grouped_gemm", False
    ):  # config-access-exempt: Megatron-Core versions differ in which TransformerConfig fields exist
        raise ValueError("MXFP4 fake QAT requires grouped-GEMM routed experts (moe_grouped_gemm).")
    if getattr(
        config, "use_transformer_engine_op_fuser", False
    ):  # config-access-exempt: Megatron-Core versions differ in which TransformerConfig fields exist
        raise ValueError("MXFP4 fake QAT does not support the Transformer Engine op fuser.")
    if (
        getattr(config, "cuda_graph_impl", "none") != "none"
        or getattr(config, "enable_cuda_graph", False)
        or getattr(config, "external_cuda_graph", False)
    ):  # config-access-exempt: Megatron-Core versions differ in which TransformerConfig fields exist
        raise ValueError(
            "MXFP4 fake QAT does not support CUDA graphs: whether to round is decided in Python on each "
            "forward, which a captured graph does not re-evaluate when it replays."
        )
    if config.fp8 is None and config.fp4 is None:
        raise ValueError(
            "MXFP4 fake QAT rounds the weights on the first microbatch of each forward-backward pass, "
            "which Megatron marks only when an FP8 or FP4 training recipe is configured."
        )
    enabled_flags = [flag for flag in _OTHER_FAKE_QAT_FLAGS if os.getenv(flag, "0") == "1"]
    if enabled_flags:
        raise ValueError(f"MXFP4 fake QAT cannot be combined with {', '.join(enabled_flags)}.")


def _round_to_mxfp4_in_place(weight: torch.Tensor) -> None:
    fake_quant_fp4_forward(weight, out=weight)


def _round_in_place_once_per_pass(
    module, weight_tensors: Sequence[torch.Tensor], *, round_in_place: Callable[[torch.Tensor], None]
) -> Sequence[torch.Tensor]:
    # Megatron sets is_first_microbatch on the first microbatch of every forward-backward pass, which is
    # also when Transformer Engine refreshes its cached quantized weights. The weights do not change
    # within a pass and rounding is idempotent, so rounding then covers the whole pass, including the
    # forwards that activation recomputation repeats.
    if not module.is_first_microbatch:
        return weight_tensors
    if any(
        hasattr(weight, "__fsdp_param__") for weight in weight_tensors
    ):  # config-access-exempt: Megatron FSDP marks its weights with this attribute
        raise RuntimeError("MXFP4 fake QAT does not support Megatron FSDP weights.")
    with torch.no_grad():
        for weight in weight_tensors:
            round_in_place(weight.detach())
    return weight_tensors


__all__ = [
    "MOE_EXPERT_FAKE_QUANT_FIELD",
    "MXFP4",
    "fake_quantize_mxfp4_weight_tensors",
    "validate_moe_expert_fake_quant_config",
]
