"""Stock SGLang residual norm at Megatron's existing norm and BDA seams.

Between seams, [branch, residual] is packed along the last dimension. This is
transport, not addition: checkpointing and PP see one tensor with both gradient
paths. Norm returns the ordinary activation/residual pair the layer accepts.
"""

import torch
from megatron.core.transformer.transformer_layer import TransformerLayer

from miles_plugins.top.kernels.rms_norm_backward import rms_norm_backward


class ResidualTransformerLayer(TransformerLayer):
    """Keep Megatron's forward; declare the first layer's unique unpaired input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layernorm.requires_residual_pair = self.layer_number != 1


class _StockResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, norm):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = norm.variance_epsilon
        # The inference kernel mutates both buffers. Never mutate Megatron's
        # tensors or the operands saved for backward/recomputation.
        return norm(
            x.clone(memory_format=torch.contiguous_format),
            residual.clone(memory_format=torch.contiguous_format),
        )

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        x, residual, weight = ctx.saved_tensors
        dx, dr, dw = rms_norm_backward(
            x, weight, ctx.eps, grad_output,
            residual=residual, grad_residual=grad_residual,
        )
        return dx, dr, dw, None


def delegated_residual_norm(x, residual, norm):
    """BF16 implementation; quantized output requires a separately admitted recipe."""
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("[top] residual norm requires equal-shaped, equal-dtype inputs")
    if x.dtype != torch.bfloat16 or norm.weight.dtype != x.dtype or not x.is_cuda:
        raise NotImplementedError("[top] stock residual norm currently requires CUDA BF16")
    if norm.cast_x_before_out_mul or norm.override_orig_dtype is not None:
        raise NotImplementedError("[top] residual norm requires stock SGLang rounding")
    return _StockResidualNorm.apply(x, residual, norm.weight, norm)


def deferred_residual_add(training, fused):
    """BDA builder: retain operands instead of prematurely rounding their sum."""
    return _pack_residual


def _pack_residual(output_with_bias, residual, probability):
    x, bias = output_with_bias
    if bias is not None or probability != 0:
        raise NotImplementedError("[top] deferred residual add requires no bias and zero dropout")
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("[top] residual operands must have identical shape and dtype")
    return torch.cat((x, residual), dim=-1)


def validate_residual_config(config, *, use_te):
    """Reject unsupported execution paths before constructing model parameters."""
    unsupported = {
        "transformer_engine spec": use_te,
        "FP8 (recipe not yet admitted)": bool(getattr(config, "fp8", None)),
        "FP4": bool(getattr(config, "fp4", None)),
        "FP32 residual": config.fp32_residual_connection,
        "hidden dropout": config.hidden_dropout != 0,
        "linear bias": config.add_bias_linear,
        "hyper connections": getattr(config, "enable_hyper_connections", False),
        "CPU offload": getattr(config, "cpu_offloading", False),
        "MTP": bool(getattr(config, "mtp_num_layers", None)),
        "fused TP inference": getattr(config, "inference_fuse_tp_communication", False),
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise NotImplementedError("[top] stock residual norm does not support: " + ", ".join(rejected))
    if not hasattr(config, "pipeline_hidden_size"):
        raise RuntimeError("[top] Megatron needs pipeline_hidden_size support for residual transport")
    config.pipeline_hidden_size = 2 * config.hidden_size
