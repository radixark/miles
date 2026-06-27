# ruff: noqa
"""Block-sparse GQA attention over MSA-selected key/value blocks.

M3 analogue of GLM-5's `ops/sparse_mla.py::SparseMLA`. The contract:

    q:         [N, Hq, D]            bf16   (Hq = num_attention_heads)
    k:         [N, Hkv, D]           bf16   (Hkv = num_key_value_heads == num index heads)
    v:         [N, Hkv, D]           bf16
    block_ids: [N, topk_blocks]      int32  (sequence-local block ids, -1 = unused)
    out:       [N, Hq, D]            bf16

Each query attends only to keys inside its (≤16) selected blocks of `block_size`
tokens plus, implicitly, the forced init/local blocks already baked into
`block_ids` by the indexer. Causality is enforced inside the masked SDPA.

Two implementations:
  * `block_sparse_attention_reference` — dense torch SDPA with a block keep-mask;
    O(N^2) memory, used as the correctness oracle and small-shape fallback.
  * `BlockSparseGQA` — autograd Function hook for a fused varlen kernel
    (flash-attn block-mask or a tilelang kernel mirroring glm5's sparse_mla
    fwd/bwd). Wired to the reference until the fused kernel lands.

NOTE: the fused forward/backward kernels (`tilelang_block_sparse_fwd/bwd`) are
the only pieces still TODO; everything upstream (indexer, selection, spec
wiring, weight layout) is complete. The reference path makes the whole model
runnable and trainable at small/medium sequence lengths today.
"""

from __future__ import annotations

import torch

from .msa_indexer import selected_blocks_to_token_mask


def block_sparse_attention_reference(
    q: torch.Tensor,          # [N, Hq, D]
    k: torch.Tensor,          # [N, Hkv, D]
    v: torch.Tensor,          # [N, Hkv, D]
    block_ids: torch.Tensor,  # [N, topk_blocks]
    cu_seqlens: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference block-sparse GQA. Returns [N, Hq, D]."""
    N, Hq, D = q.shape
    Hkv = k.shape[1]
    g = Hq // Hkv
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    out = torch.empty_like(q)

    for s in range(cu_seqlens.numel() - 1):
        lo, hi = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        n = hi - lo
        if n == 0:
            continue
        keep = selected_blocks_to_token_mask(block_ids[lo:hi], n, block_size)  # [n, n]

        qs = q[lo:hi].float().transpose(0, 1)             # [Hq, n, D]
        ks = k[lo:hi].float().transpose(0, 1)             # [Hkv, n, D]
        vs = v[lo:hi].float().transpose(0, 1)             # [Hkv, n, D]
        ks = ks.repeat_interleave(g, dim=0)               # [Hq, n, D]
        vs = vs.repeat_interleave(g, dim=0)

        attn = torch.einsum("hid,hjd->hij", qs, ks) * softmax_scale  # [Hq, n, n]
        attn = attn.masked_fill(~keep[None], float("-inf"))
        attn = attn.softmax(dim=-1)
        o = torch.einsum("hij,hjd->hid", attn, vs)        # [Hq, n, D]
        out[lo:hi] = o.transpose(0, 1).to(out.dtype)
    return out


class BlockSparseGQA(torch.autograd.Function):
    """Autograd hook for the fused block-sparse GQA kernel.

    Mirrors GLM-5's `SparseMLA.apply(q, kv, topk_indices, scale)` call site so
    the attention module is kernel-agnostic. Swap the forward/backward bodies
    for the tilelang kernels once written; the reference keeps it trainable now.
    """

    @staticmethod
    def forward(ctx, q, k, v, block_ids, cu_seqlens, block_size, softmax_scale):
        out = block_sparse_attention_reference(
            q, k, v, block_ids, cu_seqlens,
            block_size=block_size, softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(q, k, v, block_ids, cu_seqlens)
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Reference: recompute via autograd on the dense-masked path. A fused
        # kernel would return analytic grads for q/k/v directly.
        q, k, v, block_ids, cu_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            out = block_sparse_attention_reference(
                qd, kd, vd, block_ids, cu_seqlens,
                block_size=ctx.block_size, softmax_scale=ctx.softmax_scale,
            )
            gq, gk, gv = torch.autograd.grad(out, (qd, kd, vd), grad_out)
        return gq, gk, gv, None, None, None, None


def block_sparse_gqa(q, k, v, block_ids, cu_seqlens, block_size, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return BlockSparseGQA.apply(q, k, v, block_ids, cu_seqlens, block_size, softmax_scale)
