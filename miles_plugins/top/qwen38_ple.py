"""Qwen3.8-Next N-gram (PLE) binding: rollout forward, Miles FP32-chain backward.

The forward delegates the key/value projections to the shared BF16 BI GEMM and the
increment to SGLang's pinned program with zero history at each packed document.
The backward differentiates the Miles FP32 chain at the same inputs and is not
bitwise qualified.
"""

import torch
from sglang.srt.batch_invariant_ops import matmul_persistent
from sglang.srt.true_on_policy import ple as sgl_ple

from miles.kernels.embedding.ple_gate_conv import _seg_bounds, ple_gate_conv_backward


def _mm_out(a, b, dtype):
    """a @ b through aten::mm.out, which the BF16-only BI override does not replace."""
    out = torch.empty(a.shape[0], b.shape[1], dtype=dtype, device=a.device)
    return torch.mm(a.to(dtype), b.to(dtype), out=out)


class _DelegatedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return matmul_persistent(x.contiguous(), weight.t())

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx = _mm_out(dy, weight, torch.float32).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = _mm_out(dy.t(), x, torch.float32).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw


class _TopPLEIncrement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hc_state, key, value, wk, wq, wc, conv_w, n, eps, dilation, cu_seqlens):
        T, W = hc_state.shape
        dev = hc_state.device
        out, _, (gate, rstdk, rstdq, rstdc, conv_pre) = sgl_ple.ple_increment(
            hc_state=hc_state,
            key=key,
            value=value,
            norm_key_w=wk,
            norm_query_w=wq,
            norm_conv_w=wc,
            conv_weight=conv_w,
            n=n,
            eps=eps,
            dilation=dilation,
            offsets=sgl_ple.segment_offsets(cu_seqlens, T, dev),
        )
        seg_lo, seg_hi = _seg_bounds(T, cu_seqlens, dev)
        Kk = conv_w.shape[-1]
        ctx.save_for_backward(
            hc_state,
            key,
            value,
            wk,
            wq,
            wc,
            conv_w.reshape(W, Kk).contiguous(),
            gate,
            rstdk,
            rstdq,
            rstdc,
            seg_lo,
            seg_hi,
            conv_pre,
        )
        ctx.dims = (n, eps, dilation, Kk, conv_w.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        (hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi, conv_pre) = (
            ctx.saved_tensors
        )
        n, eps, dilation, Kk, conv_w_dtype = ctx.dims
        return ple_gate_conv_backward(
            hc_state=hc_state,
            key=key,
            value=value,
            wk=wk,
            wq=wq,
            wc=wc,
            convw2d=convw2d,
            gate=gate,
            rstdk=rstdk,
            rstdq=rstdq,
            rstdc=rstdc,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            conv_pre=conv_pre,
            n=n,
            eps=eps,
            dilation=dilation,
            Kk=Kk,
            conv_w_dtype=conv_w_dtype,
            dout=dout,
        )


def top_ple_forward(module, hc_state, embeddings, cu_seqlens):
    """``Qwen38NextPLE`` increment [T, n*C] from flat ``hc_state`` and gathered embeddings."""
    embeddings = embeddings.contiguous()
    key = _DelegatedLinear.apply(embeddings, module.key_proj.weight)
    value = _DelegatedLinear.apply(embeddings, module.value_proj.weight)
    return _TopPLEIncrement.apply(
        hc_state.contiguous(),
        key,
        value,
        module.norm_key,
        module.norm_query,
        module.norm_conv,
        module.conv1d_weight,
        module.n,
        module.norm_eps,
        module.conv_dilation,
        cu_seqlens,
    )


def install() -> str:
    """Bind the Qwen3.8-Next model's PLE increment to the rollout program; returns the stamp."""
    from miles_plugins.models.qwen3_8_next.ops import ple

    ple.set_top_ple_forward(top_ple_forward)
    return "qwen38_ple=delegate(sglang.true_on_policy.ple)"
