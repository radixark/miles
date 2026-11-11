"""Causal attention over packed sequences on TPU: JAX's Pallas splash-attention kernel, called through torch_tpu.

The torch ops wrap the JAX functions with torch_tpu's Pallas bridge, as sglang-torchtpu does
(sglang_tpu/attention_pallas.py). Skipping the q/kv blocks that share no segment follows google-pytorch/torchtitan
torchtitan/experiments/tpu/kernels/splash_attention.py (338990d6).
"""

import functools
import math

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
from torch_tpu._internal import pallas

from miles.kernels.tpu import FLASH_ATTENTION_BLOCK as BLOCK


def _skip_blocks_across_segments(mask_info, segment_ids, block_q: int, block_kv: int, *, is_dkv: bool):
    """Zero the causal block mask where a q block and a kv block share no segment.

    Segment ids are non-decreasing, so a block's segments are the interval [min, max] over it, and a tile holds a
    same-segment pair iff the two intervals overlap.
    """
    seq_len = segment_ids.shape[-1]
    num_q, num_kv = seq_len // block_q, seq_len // block_kv
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


@functools.cache
def _kernel(seq_len: int, num_heads: int, block: int):
    block = min(block, seq_len)
    block_sizes = sk.BlockSizes(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        use_fused_bwd_kernel=True,
    )
    mask = sm.MultiHeadMask(masks=(sm.CausalMask(shape=(seq_len, seq_len)),) * num_heads)
    # first built inside a trace: concrete mask metadata, or the cached tracers leak into the other pass's trace
    with jax.ensure_compile_time_eval():
        return block_sizes, sk.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)


def _scaled(x):
    # the kernel takes q pre-scaled; scale in fp32 as SDPA does, a bf16 1/sqrt(head_dim) is off by ~1e-4
    return (x.astype(jnp.float32) * (1.0 / math.sqrt(x.shape[-1]))).astype(x.dtype)


def _forward(
    q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: jax.Array, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    block_sizes, kernel = _kernel(q.shape[2], q.shape[1], BLOCK)
    kw = kernel.kwargs

    def one(q, k, v, seg):
        info = _skip_blocks_across_segments(
            kernel.fwd_mask_info, seg, block_sizes.block_q, block_sizes.block_kv, is_dkv=False
        )
        out, (lse,) = sk._splash_attention_forward(
            info,
            q,
            k,
            v,
            segment_ids=sk.SegmentIds(q=seg, kv=seg),
            sinks=None,
            mask_value=kw.get("mask_value", sk.DEFAULT_MASK_VALUE),
            is_mqa=False,
            block_sizes=block_sizes,
            residual_checkpoint_name=kw.get("residual_checkpoint_name"),
            save_residuals=True,
            mask_function=kw.get("mask_function"),
            attn_logits_soft_cap=kw.get("attn_logits_soft_cap"),
            interpret=interpret,
        )
        return out, lse

    return jax.vmap(one)(_scaled(q), k, v, segment_ids)


def _backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    lse: jax.Array,
    grad_out: jax.Array,
    segment_ids: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    block_sizes, kernel = _kernel(q.shape[2], q.shape[1], BLOCK)
    kw = kernel.kwargs

    def one(q, k, v, out, lse, grad_out, seg):
        info = _skip_blocks_across_segments(
            kernel.dkv_mask_info, seg, block_sizes.block_q_dkv, block_sizes.block_kv_dkv, is_dkv=True
        )
        residuals = (q, k, v, sk.SegmentIds(q=seg, kv=seg), None, out, lse, None, info)
        grads = sk._splash_attention_bwd(
            save_residuals=False,
            mask_value=kw.get("mask_value", sk.DEFAULT_MASK_VALUE),
            is_mqa=False,
            block_sizes=block_sizes,
            residual_checkpoint_name=kw.get("residual_checkpoint_name"),
            mask_function=kw.get("mask_function"),
            attn_logits_soft_cap=kw.get("attn_logits_soft_cap"),
            interpret=interpret,
            res=residuals,
            do=grad_out,
        )
        return grads[3], grads[4], grads[5]

    _, (dq, dk, dv) = jax.lax.scan(
        lambda carry, xs: (carry, one(*xs)), None, (_scaled(q), k, v, out, lse, grad_out, segment_ids)
    )
    return _scaled(dq), dk, dv


# torch_tpu's Pallas bridge traces every parameter of the function it is given, so a static flag such as `interpret`
# would arrive as a tracer: the registered kernels take tensors only
def _forward_on_tpu(q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _forward(q, k, v, segment_ids)


def _backward_on_tpu(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    lse: jax.Array,
    grad_out: jax.Array,
    segment_ids: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _backward(q, k, v, out, lse, grad_out, segment_ids)


# registered at import, outside any compiled region. custom_jax_kernel, not jax_op: jax_op exports the function, and
# the splash mask metadata it closes over then counts as extra inputs
_forward_op = torch.library.custom_op(
    "miles_tpu::flash_attention_fwd",
    pallas.custom_jax_kernel(jax.jit(_forward_on_tpu), name="flash_attention_fwd"),
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor segment_ids) -> (Tensor, Tensor)",
)
_backward_op = torch.library.custom_op(
    "miles_tpu::flash_attention_bwd",
    pallas.custom_jax_kernel(jax.jit(_backward_on_tpu), name="flash_attention_bwd"),
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor grad_out, Tensor segment_ids)"
    " -> (Tensor, Tensor, Tensor)",
)
_forward_op.register_fake(
    lambda q, k, v, segment_ids: (torch.empty_like(q), q.new_empty(q.shape[:3], dtype=torch.float32))
)
_backward_op.register_fake(
    lambda q, k, v, out, lse, grad_out, segment_ids: (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
)


def _setup_context(ctx, inputs, output):
    q, k, v, segment_ids = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], segment_ids)


def _grad(ctx, grad_out, grad_lse):
    q, k, v, out, lse, segment_ids = ctx.saved_tensors
    dq, dk, dv = _backward_op(q, k, v, out, lse, grad_out.contiguous(), segment_ids)
    return dq, dk, dv, None


_forward_op.register_autograd(_grad, setup_context=_setup_context)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segment_ids: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention within each segment, GQA included. q: [B, Hq, L, D]; k, v: [B, Hkv, L, D];
    segment_ids: [B, L] int32, non-decreasing, L a multiple of `BLOCK`."""
    head_dim = q.shape[-1]
    if scale is not None and abs(scale * math.sqrt(head_dim) - 1.0) > 1e-6:
        q = q * (scale * math.sqrt(head_dim))  # the kernel applies 1/sqrt(head_dim)
    out, _ = _forward_op(q.contiguous(), k.contiguous(), v.contiguous(), segment_ids)
    return out
