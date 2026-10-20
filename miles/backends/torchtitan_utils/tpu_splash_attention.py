"""Packed-document causal attention on TPU with JAX's Pallas splash-attention kernel; documents are segment ids."""

# copied from google-pytorch/torchtitan torchtitan/experiments/tpu/kernels/splash_attention.py (338990d6), trimmed

import hashlib
import math

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
from torch_tpu._internal import pallas

_MASK_VALUE = sk.DEFAULT_MASK_VALUE


def _document_aware_mask_info(mask_info, segment_ids, block_q: int, block_kv: int, *, is_dkv: bool):
    """Zero the causal block mask where a q block and a kv block share no document.

    Segment ids are non-decreasing, so a block's documents are the interval [min, max] over it and a tile has a
    same-document pair iff the two intervals overlap.
    """
    if mask_info.block_mask is None or mask_info.data_next is None:
        return mask_info
    seq_len = segment_ids.shape[-1]
    num_q, num_kv = seq_len // block_q, seq_len // block_kv
    if seq_len % block_q or seq_len % block_kv or mask_info.block_mask.shape[-2:] != (num_q, num_kv):
        return mask_info
    q_seg = segment_ids.reshape(num_q, block_q)
    kv_seg = segment_ids.reshape(num_kv, block_kv)
    overlaps = (q_seg.min(-1)[:, None] <= kv_seg.max(-1)[None, :]) & (
        kv_seg.min(-1)[None, :] <= q_seg.max(-1)[:, None]
    )
    block_mask = jnp.where(overlaps[None], mask_info.block_mask, 0).astype(mask_info.block_mask.dtype)
    if is_dkv:
        data_next = jnp.arange(num_q, dtype=jnp.int32)[None, :, None]
    else:
        data_next = jnp.arange(num_kv, dtype=jnp.int32)[None, None, :]
    data_next = jnp.where(block_mask == 0, 0, jnp.broadcast_to(data_next, block_mask.shape))
    return mask_info._replace(block_mask=block_mask, data_next=data_next.astype(mask_info.data_next.dtype))


def _make_kernels(*, seq_len: int, n_heads: int, block_q: int, block_kv: int, block_q_dkv: int, block_kv_dkv: int):
    block_sizes = sk.BlockSizes(
        block_q=min(block_q, seq_len),
        block_kv=min(block_kv, seq_len),
        block_kv_compute=min(block_kv, seq_len),
        block_q_dkv=min(block_q_dkv, seq_len),
        block_kv_dkv=min(block_kv_dkv, seq_len),
        block_kv_dkv_compute=min(block_kv_dkv, seq_len),
        use_fused_bwd_kernel=True,
    )
    mask = sm.MultiHeadMask(masks=(sm.CausalMask(shape=(seq_len, seq_len)),) * n_heads)
    kernel = sk.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
    fwd_info, dkv_info = kernel.fwd_mask_info, kernel.dkv_mask_info
    kw = kernel.kwargs

    def fwd_one(q, k, v, seg):
        ids = sk.SegmentIds(q=seg, kv=seg)
        info = _document_aware_mask_info(fwd_info, seg, block_sizes.block_q, block_sizes.block_kv, is_dkv=False)
        out, (lse,) = sk._splash_attention_forward(
            info,
            q,
            k,
            v,
            segment_ids=ids,
            sinks=None,
            mask_value=kw.get("mask_value", _MASK_VALUE),
            is_mqa=False,
            block_sizes=block_sizes,
            residual_checkpoint_name=kw.get("residual_checkpoint_name"),
            save_residuals=True,
            mask_function=kw.get("mask_function"),
            attn_logits_soft_cap=kw.get("attn_logits_soft_cap"),
            interpret=kw.get("interpret", False),
        )
        return out, lse

    def bwd_one(q, k, v, out, lse, g, seg):
        ids = sk.SegmentIds(q=seg, kv=seg)
        info = _document_aware_mask_info(dkv_info, seg, block_sizes.block_q_dkv, block_sizes.block_kv_dkv, is_dkv=True)
        res = (q, k, v, ids, None, out, lse, None, info)
        grads = sk._splash_attention_bwd(
            False,
            kw.get("mask_value", _MASK_VALUE),
            False,
            block_sizes,
            kw.get("residual_checkpoint_name"),
            kw.get("mask_function"),
            kw.get("attn_logits_soft_cap"),
            kw.get("interpret", False),
            res,
            g,
        )
        return grads[3], grads[4], grads[5]  # dq, dk, dv (q pre-scaled)

    def scaled(x):
        # scale in fp32, as SDPA does: a bf16 1/sqrt(head_dim) is off by ~1e-4
        return (x.astype(jnp.float32) * (1.0 / math.sqrt(x.shape[-1]))).astype(x.dtype)

    @jax.jit
    def splash_fwd(q, k, v, segment_ids):
        return jax.vmap(fwd_one)(scaled(q), k, v, segment_ids)

    @jax.jit
    def splash_bwd(q, k, v, out, lse, g, segment_ids):
        _, (dq, dk, dv) = jax.lax.scan(
            lambda c, xs: (c, bwd_one(*xs)), None, (scaled(q), k, v, out, lse, g, segment_ids)
        )
        return scaled(dq), dk, dv

    return splash_fwd, splash_bwd


# torch.library registers an op name once per process, so each kernel shape is built and registered once
_OPS: dict[tuple, tuple[str, str]] = {}


@torch.compiler.assume_constant_result
def _op_names(**kernel_args) -> tuple[str, str]:
    key = tuple(sorted(kernel_args.items()))
    if key in _OPS:
        return _OPS[key]
    jax_fwd, jax_bwd = _make_kernels(**kernel_args)
    ident = hashlib.sha256(str(key).encode()).hexdigest()[:16]
    fwd_name, bwd_name = f"miles_splash_fwd_{ident}", f"miles_splash_bwd_{ident}"
    fwd = torch.library.custom_op(
        f"pallas::{fwd_name}",
        pallas.custom_jax_kernel(jax_fwd, name="splash_attention"),
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor segment_ids) -> (Tensor, Tensor)",
    )
    fwd.register_fake(
        lambda q, k, v, segment_ids: (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(q.shape[:3], dtype=torch.float32, device=q.device),
        )
    )
    bwd = torch.library.custom_op(
        f"pallas::{bwd_name}",
        pallas.custom_jax_kernel(jax_bwd, name="splash_attention_bwd"),
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor grad_out, Tensor segment_ids)"
        " -> (Tensor, Tensor, Tensor)",
    )
    bwd.register_fake(
        lambda q, k, v, out, lse, grad_out, segment_ids: (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(k.shape, dtype=k.dtype, device=k.device),
            torch.empty(v.shape, dtype=v.dtype, device=v.device),
        )
    )

    def setup_context(ctx, inputs, output):
        q, k, v, segment_ids = inputs
        ctx.save_for_backward(q, k, v, output[0], output[1], segment_ids)

    def backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse, segment_ids = ctx.saved_tensors
        dq, dk, dv = bwd(q, k, v, out, lse, grad_out.contiguous(), segment_ids)
        return dq, dk, dv, None

    fwd.register_autograd(backward, setup_context=setup_context)
    _OPS[key] = (fwd_name, bwd_name)
    return _OPS[key]


def splash_document_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    scale: float | None = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_dkv: int = 1024,
) -> torch.Tensor:
    """q: (B, Nq, L, H); k, v: (B, Nkv, L, H); segment_ids: (B, L) int32, non-decreasing per row."""
    batch, n_heads, seq_len, head_dim = q.shape
    # the kernel reads and writes contiguous (B, N, L, H); the fake outputs say so too
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if scale is not None and abs(scale * math.sqrt(head_dim) - 1.0) > 1e-6:
        q = q * (scale * math.sqrt(head_dim))  # the kernel applies 1/sqrt(head_dim)
    fwd_name, _ = _op_names(
        seq_len=seq_len,
        n_heads=n_heads,
        block_q=block_q,
        block_kv=block_kv,
        block_q_dkv=block_dkv,
        block_kv_dkv=block_dkv,
    )
    out, _ = getattr(torch.ops.pallas, fwd_name)(q, k, v, segment_ids)
    return out
