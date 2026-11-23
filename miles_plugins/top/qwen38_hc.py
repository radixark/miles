"""Qwen3.8-Next Gated Residual binding: rollout forward, Miles FP32-chain backward.

The forward delegates to SGLang's pinned TOP program, so trainer scores see the
rollout bytes. The backward differentiates the FP32 chain at the same inputs; it
ignores the forward's BF16 rounding boundaries and is not bitwise qualified.
"""

import torch
from sglang.srt.true_on_policy import hyper_connection as sgl_hc

from miles.kernels.hyper_connection.hc import combine_backward, mix_inject_backward


class _TopHCMixInject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps):
        mixed, normed, rstd = sgl_hc.hc_mix(x2d, norm_weight=weight, w_down=w_down, w_up=w_up, n=n, eps=eps)
        if w_inject is not None:
            h_post = sgl_hc.hc_inject(normed, w_inject=w_inject, n=n)
        else:
            h_post = x2d.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd)
        ctx.n, ctx.eps = n, eps
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        x2d, weight, w_down, w_up, w_inject, rstd = ctx.saved_tensors
        return mix_inject_backward(
            x2d=x2d,
            weight=weight,
            w_down=w_down,
            w_up=w_up,
            w_inject=w_inject,
            rstd=rstd,
            n=ctx.n,
            eps=ctx.eps,
            dmixed=dmixed,
            dh_post=dh_post,
        )


class _TopHCCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual2d, y2d, h_post, n):
        out = sgl_hc.hc_combine(residual2d, y2d=y2d, h_post=h_post, n=n)
        ctx.save_for_backward(y2d, h_post)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, dout):
        y2d, h_post = ctx.saved_tensors
        dy, dh_post = combine_backward(y2d=y2d.contiguous(), h_post=h_post, n=ctx.n, dout=dout)
        return dout, dy, dh_post, None


def top_hc_mix_inject(x, weight, w_down, w_up, w_inject, n: int, eps: float):
    """Same signature and shapes as ``miles.kernels.hyper_connection.hc.hc_mix_inject``."""
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    mixed, h_post = _TopHCMixInject.apply(x2d, weight, w_down, w_up, w_inject, n, eps)
    mixed = mixed.reshape(*lead, -1)
    if w_inject is not None:
        h_post = h_post.reshape(*lead, n)
    return mixed, h_post


def top_hc_combine(residual, block_output, h_post, n: int):
    """Same signature and shapes as ``miles.kernels.hyper_connection.hc.hc_combine``."""
    lead = residual.shape[:-1]
    r2d = residual.reshape(-1, residual.shape[-1]).contiguous()
    y2d = block_output.reshape(-1, block_output.shape[-1]).contiguous()
    hp2d = h_post.reshape(-1, n).float()
    out = _TopHCCombine.apply(r2d, y2d, hp2d, n)
    return out.reshape(*lead, -1)


def install() -> str:
    """Bind the Qwen3.8-Next model's Gated Residual to the rollout program; returns the stamp."""
    from miles_plugins.models.qwen3_8_next import hyper_connection

    hyper_connection.set_hc_ops(mix_inject=top_hc_mix_inject, combine=top_hc_combine)
    return "qwen38_hc=delegate(sglang.true_on_policy.hyper_connection)"
