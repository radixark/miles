from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps

import torch
import triton
import triton.language as tl

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE

_CONFIG_ATTR = "dsv4_mxfp4_qat"
_PATCH_ATTR = "_miles_dsv4_mxfp4_qat_original_get_weight_tensors"
_GROUPS_PER_PROGRAM = 64

logger = logging.getLogger(__name__)


@triton.jit
def _mxfp4_quantize_dequantize_kernel(
    weight_ptr,
    output_ptr,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    group_offsets = tl.program_id(0) * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    value_offsets = tl.arange(0, GROUP_SIZE)
    offsets = group_offsets[:, None] * GROUP_SIZE + value_offsets[None, :]
    mask = group_offsets[:, None] < num_groups

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(weight), axis=1)[:, None]
    exponent = tl.ceil(tl.log2(amax / 6.0))
    exponent = tl.maximum(exponent, -127.0)
    exponent = tl.where(amax > 0.0, exponent, -127.0)
    scale = tl.exp2(exponent)

    magnitude = tl.abs(weight) / scale
    code = (magnitude > 0.25).to(tl.int32)
    code += (magnitude > 0.75).to(tl.int32)
    code += (magnitude > 1.25).to(tl.int32)
    code += (magnitude > 1.75).to(tl.int32)
    code += (magnitude > 2.5).to(tl.int32)
    code += (magnitude > 3.5).to(tl.int32)
    code += (magnitude > 5.0).to(tl.int32)

    quantized = tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            0.5,
            tl.where(
                code == 2,
                1.0,
                tl.where(
                    code == 3,
                    1.5,
                    tl.where(code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )
    quantized = tl.where(weight < 0.0, -quantized, quantized)
    tl.store(output_ptr + offsets, quantized * scale, mask=mask)


def mxfp4_quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    """Round a CUDA tensor to the routed-expert MXFP4 grid."""
    if not weight.is_cuda:
        raise ValueError("MXFP4 QAT requires a CUDA weight tensor.")
    if not weight.is_contiguous():
        raise ValueError("MXFP4 QAT requires contiguous grouped-expert weights.")
    if weight.shape[-1] % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {weight.shape[-1]} must be divisible by {MXFP4_GROUP_SIZE} for MXFP4 QAT.")

    output = torch.empty_like(weight)
    num_groups = weight.numel() // MXFP4_GROUP_SIZE
    grid = (triton.cdiv(num_groups, _GROUPS_PER_PROGRAM),)
    _mxfp4_quantize_dequantize_kernel[grid](
        weight,
        output,
        num_groups,
        GROUP_SIZE=MXFP4_GROUP_SIZE,
        GROUPS_PER_PROGRAM=_GROUPS_PER_PROGRAM,
        num_warps=4,
    )
    return output


# Megatron DDP attributes that Transformer Engine reads from the weights it receives.
_DDP_WEIGHT_ATTRS = ("main_grad", "grad_added_to_main_grad", "zero_out_wgrad")


class _MXFP4FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, wgrad_in_main_grad: bool) -> torch.Tensor:
        ctx.weight = weight if wgrad_in_main_grad else None
        return mxfp4_quantize_dequantize(weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        if ctx.weight is not None:
            # Transformer Engine accumulated the weight gradient into the shared
            # main_grad and returned a placeholder; DDP must not add it again.
            ctx.weight.grad_added_to_main_grad = True
        return grad_output, None


def mxfp4_fake_quantize_ste(weight: torch.Tensor, *, wgrad_in_main_grad: bool = False) -> torch.Tensor:
    """Fake-quantize a weight while passing its gradient through unchanged.

    The output carries the weight's DDP attributes, so a GEMM that fuses weight-gradient
    accumulation adds into the weight's main_grad. With ``wgrad_in_main_grad`` the
    backward marks that gradient as already added, as Transformer Engine does for the
    weights it receives directly.
    """
    wgrad_in_main_grad = wgrad_in_main_grad and hasattr(weight, "grad_added_to_main_grad")
    output = _MXFP4FakeQuantizeSTE.apply(weight, wgrad_in_main_grad)
    for attr in _DDP_WEIGHT_ATTRS:
        if hasattr(weight, attr):
            setattr(output, attr, getattr(weight, attr))
    return output


def _wrap_get_weight_tensors(original: Callable) -> Callable:
    @wraps(original)
    def get_weight_tensors(module):
        weights = original(module)
        if not getattr(module.config, _CONFIG_ATTR, False):
            return weights
        if any(hasattr(weight, "__fsdp_param__") or hasattr(weight, "get_main_grad") for weight in weights):
            raise RuntimeError("DeepSeek V4 MXFP4 QAT does not support Megatron FSDP weights.")
        wgrad_in_main_grad = getattr(module.config, "gradient_accumulation_fusion", False)
        return [mxfp4_fake_quantize_ste(weight, wgrad_in_main_grad=wgrad_in_main_grad) for weight in weights]

    return get_weight_tensors


def validate_dsv4_mxfp4_qat_config(config) -> None:
    """Reject configurations whose routed experts would not reach the fake-quantization hook."""
    if not getattr(config, "moe_grouped_gemm", False):
        raise ValueError("DeepSeek V4 MXFP4 QAT requires grouped-GEMM routed experts (moe_grouped_gemm).")
    if getattr(config, "use_transformer_engine_op_fuser", False):
        raise ValueError("DeepSeek V4 MXFP4 QAT does not support the Transformer Engine op fuser.")
    if getattr(config, "delay_wgrad_compute", False):
        raise ValueError("DeepSeek V4 MXFP4 QAT does not support delayed weight-gradient computation.")


def install_dsv4_mxfp4_qat() -> None:
    """Install the opt-in routed-expert MXFP4 fake-quantization hook."""
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    if hasattr(TEGroupedLinear, _PATCH_ATTR):
        return

    original = TEGroupedLinear._get_weight_tensors
    setattr(TEGroupedLinear, _PATCH_ATTR, original)
    TEGroupedLinear._get_weight_tensors = _wrap_get_weight_tensors(original)
    logger.info("Installed DeepSeek V4 routed-expert MXFP4 QAT hook")
