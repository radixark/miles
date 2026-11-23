"""Forward delegation to the rollout operations, with analytic backwards.

Each backward uses tensors saved by the forward; none re-runs a forward. The
rollout has no backward to match, so gradients need ordinary accuracy only.
"""

from __future__ import annotations

import torch

from miles_plugins.top.kernels.rms_norm_backward import rms_norm_backward


class _DelegatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, sgl_module):
        out = sgl_module(x)
        if isinstance(out, tuple):
            out = out[0]
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        dx, _, dw = rms_norm_backward(x, weight, ctx.eps, grad_out)
        return dx, dw, None, None


def delegated_rms_norm(x, sgl_module, eps):
    return _DelegatedRMSNorm.apply(x, sgl_module.weight, eps, sgl_module)


def _linear_backward(ctx, grad_out):
    """Ordinary dX/dW/dB for y = x W^T + b from the saved forward operands."""
    x_2d, weight = ctx.saved_tensors
    grad_x = grad_out @ weight if ctx.needs_input_grad[0] else None
    grad_w = grad_out.t() @ x_2d if ctx.needs_input_grad[1] else None
    grad_b = grad_out.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
    return grad_x, grad_w, grad_b


class _DelegatedMatchedTp1Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_2d, weight, bias):
        from sglang.srt.true_on_policy.linear import matched_tp1_row_linear

        ctx.save_for_backward(x_2d, weight)
        ctx.has_bias = bias is not None
        return matched_tp1_row_linear(x_2d, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        return _linear_backward(ctx, grad_out)


def delegated_matched_tp1_linear(x_2d, weight, bias=None):
    return _DelegatedMatchedTp1Matmul.apply(x_2d, weight, bias)


class _DelegatedQKNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Optional CUDA kernel dependency is needed only for this delegated path.
        from sglang.srt.true_on_policy.qk_norm import fused_qk_norm

        # Megatron's norm interface returns fresh storage; preserve its input.
        out = x.clone(memory_format=torch.contiguous_format)
        rows = out.view(-1, 1, out.shape[-1])
        empty = out.new_empty((rows.shape[0], 0, out.shape[-1]))
        fused_qk_norm(rows, empty, weight, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        width = x.shape[-1]
        dx, _, dw = rms_norm_backward(x.reshape(-1, width), weight, ctx.eps, grad_out.reshape(-1, width))
        return dx.view_as(x), dw, None


def delegated_qk_norm(x, weight, eps):
    return _DelegatedQKNorm.apply(x, weight, eps)
