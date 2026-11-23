"""Trainer KDA core: serving's pinned chunk prefill forward, FLA's chunk_kda backward.

The value is serving's KDA, whose decode follows its chunk prefill bit for bit (SGLang ``true_on_policy.kda_chunked``),
so every packed document is one chunk prefill from zero state. The gradient is that of FLA's ``chunk_kda`` with the
gate in-kernel and the safe gate over the same documents: the two differ only by rounding.
"""

import functools

import torch

from miles_plugins.top.kernels.surrogate import surrogate


def _split(mixed_qkv, heads, head_dim):
    t, width = mixed_qkv.shape[0], heads * head_dim
    return (x.reshape(1, t, heads, head_dim) for x in mixed_qkv.split(width, dim=-1))


def _forward(mixed_qkv, f, b, A_log, dt_bias, *, cu_seqlens, heads, head_dim, lower_bound):
    from sglang.srt.true_on_policy.kda_chunked import top_kda_kernel

    t = mixed_qkv.shape[0]
    # Serving's chunk_kda writes its output into v's buffer: hand it copies.
    q, k, v = (x.clone() for x in _split(mixed_qkv, heads, head_dim))
    docs = cu_seqlens.numel() - 1
    state = mixed_qkv.new_zeros((docs, heads, head_dim, head_dim), dtype=torch.float32)
    return top_kda_kernel().extend(
        q,
        k,
        v,
        f.reshape(1, t, heads, head_dim).contiguous(),
        b.reshape(1, t, heads).contiguous(),
        ssm_states=state,
        cache_indices=torch.arange(docs, device=mixed_qkv.device, dtype=torch.int32),
        query_start_loc=cu_seqlens,
        A_log=A_log.reshape(-1),
        dt_bias=dt_bias.reshape(-1),
        lower_bound=lower_bound,
        beta_is_raw=True,
    )


def _reference(mixed_qkv, f, b, A_log, dt_bias, *, cu_seqlens, heads, head_dim, lower_bound):
    from fla.ops.kda import chunk_kda

    t = mixed_qkv.shape[0]
    q, k, v = _split(mixed_qkv, heads, head_dim)
    out, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=f.reshape(1, t, heads, head_dim),
        beta=b.float().sigmoid().reshape(1, t, heads),
        A_log=A_log.reshape(-1),
        dt_bias=dt_bias.reshape(-1),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens.long(),
    )
    return out


def delegated_kda(mixed_qkv, f, b, A_log, dt_bias, *, cu_seqlens, heads, head_dim, lower_bound):
    """KDA gating and core over packed documents: serving's pinned chunk prefill as the value, FLA's chunk_kda as
    the gradient. ``mixed_qkv`` [T, 3 * heads * head_dim] post-conv; ``f`` [T, heads * head_dim] raw gate; ``b``
    [T, heads] raw beta; ``A_log`` [heads], ``dt_bias`` [heads * head_dim] FP32. Returns [1, T, heads, head_dim]."""
    kw = dict(cu_seqlens=cu_seqlens, heads=heads, head_dim=head_dim, lower_bound=lower_bound)
    return surrogate(
        functools.partial(_forward, **kw),
        functools.partial(_reference, **kw),
        mixed_qkv,
        f,
        b,
        A_log,
        dt_bias,
        name="kda_core",
    )
