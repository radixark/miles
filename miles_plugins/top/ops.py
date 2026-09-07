"""Delegating ops: the trainer runs the rollout's op object, with an analytic backward.

Backwards are analytic, never recompute-then-autograd; the rollout has no backward to match.
"""

from __future__ import annotations

import sglang.srt.tp_invariant_ops as _tp
import torch


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
        x, w = ctx.saved_tensors
        xf, gf, wf = x.float(), grad_out.float(), w.float()
        h = xf.shape[-1]
        r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + ctx.eps)
        gw = gf * wf
        grad_x = r * gw - (r.pow(3) * xf / h) * (gw * xf).sum(dim=-1, keepdim=True)
        grad_w = (gf * xf * r).reshape(-1, h).sum(dim=0)
        return grad_x.to(x.dtype), grad_w.to(w.dtype), None, None


def delegated_rms_norm(x, sgl_module, eps):
    return _DelegatedRMSNorm.apply(x, sgl_module.weight, eps, sgl_module)


class _DelegatedTpInvMatmul(torch.autograd.Function):
    """sglang's TP-invariant matmul. Degree-invariant, so used at every tp including 1."""

    @staticmethod
    def forward(ctx, x_2d, weight, bias):
        ctx.save_for_backward(x_2d, weight)
        ctx.has_bias = bias is not None
        return _tp.matmul_tp_inv(x_2d, weight.t(), bias=bias)

    @staticmethod
    def backward(ctx, grad_out):
        x_2d, weight = ctx.saved_tensors
        grad_x = grad_out @ weight
        grad_w = grad_out.t() @ x_2d
        grad_b = grad_out.sum(dim=0) if ctx.has_bias else None

        # megatron's wgrad-fusion protocol: accumulate into main_grad, return None
        main_grad = getattr(weight, "main_grad", None)
        if main_grad is not None:
            main_grad.add_(grad_w.to(main_grad.dtype))
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
            grad_w = None
        return grad_x, grad_w, grad_b


def delegated_tp_inv_linear(x_2d, weight, bias=None):
    return _DelegatedTpInvMatmul.apply(x_2d, weight, bias)


class _TreeReduceFromTPRegion(torch.autograd.Function):
    """Deterministic TP all-reduce: all-gather the partials, reduce with a fixed local tree.

    An ordinary all-reduce has no counterpart in a tp=1 rollout, which computes the whole sum in
    one GEMM -- so its order shows up as a divergence on every row linear. Backward is identity,
    matching megatron's _ReduceFromModelParallelRegion.
    """

    @staticmethod
    def forward(ctx, x, group):
        if group is None or group.size() == 1:
            return x
        return _tp.tree_all_reduce_sum(x.contiguous(), device_group=group)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def tree_reduce_from_tp_region(x, group=None):
    return _TreeReduceFromTPRegion.apply(x, group)
