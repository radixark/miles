"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Unified Miles DSA operators: sparse attention and lightning indexer, forward and backward.

Public surface (all layouts of :mod:`.layouts`):

* :func:`sparse_attention` -- autograd sparse MQA/MLA attention over top-k indices
  with optional FP32 attention sink; replaces ``glm5.ops.sparse_mla.SparseMLA`` and
  ``deepseek_v4...tilelang_sparse_mla.sparse_attn_tilelang``.
* :func:`indexer_logits`, :func:`select_topk`, :func:`indexer_topk_scores`,
  :func:`lightning_indexer` -- the indexer pipeline over the shared Miles top-k
  backends (:mod:`miles.kernels.attention.dsa.topk`: ``torch``, ``flashinfer``, or ``cake`` =
  the generated deterministic :func:`cake_topk`) or a caller-supplied top-k callable, with an
  explicit ``topk_indices`` override for rollout replay; replaces
  ``glm5.ops.indexer.lighting_indexer`` / ``IndexerFunction`` and
  ``deepseek_v4...tilelang_indexer.V4IndexerFunction``.
* :func:`fused_indexer_topk_scores` -- the ``cake_fused`` switch of :func:`lightning_indexer`
  (``thd``): the generated fused scoring + top-k stages return the selected scores and
  indices without the dense ``[T, T_kv]`` logits; same deterministic backward.
* :func:`sparse_attention_forward` / :func:`sparse_attention_backward` /
  :func:`indexer_backward` -- non-autograd entry points for tests and benchmarks.

Every gradient is bit-reproducible run to run: the kernels use no floating-point
atomics and the key-side gradients are reduced from per-slot partials in an order
fixed by the indices.  Parallelism is *not* handled here: the kernels see local
heads and the local-or-gathered key rows; see :mod:`.parallel` for the
wrapper-side TP/SP/CP collectives.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn

from . import layouts
from .layouts import RowMeta

ATTENTION_TOPK_MULTIPLE = 64
LOG2E = 1.4426950408889634
# Reducer of the lightning-indexer backward: 4 = ``dsa_segmented_reduce_split4`` (fixed 4-way tree per key,
# deterministic FP32); 1 = the serial per-key reducer ``dsa_segmented_reduce``.
INDEXER_REDUCE_SPLIT = 4
# Reducer of the sparse-attention backward: the same fixed 4-way tree (bandwidth-bound; the serial reducer's
# per-key latency chain is the tail of every chunk).  Fixed-order FP32.
ATTENTION_REDUCE_SPLIT = 4

TopkFn = Callable[[torch.Tensor, int], torch.Tensor]
# Not a top-k backend over logits: the ``lightning_indexer`` switch to the fused scoring + top-k stages.
FUSED_TOPK_BACKEND = "cake_fused"


# --------------------------------------------------------------------------- #
# Top-k backends (one registry for both kernel backends: miles.kernels.attention.dsa.topk)
# --------------------------------------------------------------------------- #


def cake_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """The generated deterministic FP32 row top-k (``dsa_topk_fp32``) with the same ``-1`` convention.

    ``logits`` is FP32 ``[rows, num_keys]`` with ``num_keys <= 32768`` and ``topk <= 512`` (SM100a /
    SM103a); ``-inf`` and NaN positions are never selected.  Output layout contract: slot ``j`` of a
    row holds the key with the ``j``-th highest score (ties: the smaller index first) and the unfilled
    tail is ``-1`` (also when a row has fewer than ``topk`` finite scores).  This differs from the
    ascending-index layout (``-1`` first) of the ``torch.topk`` + ``.sort`` paths; every consumer
    gathers by slot and is position-independent, so callers use the output directly and must not
    re-sort it.  Bit-identical run to run (no floating-point atomics).  Limits are enforced with a
    ``ValueError``; there is no silent fallback to another backend.  Registered as ``"cake"`` in
    :mod:`miles.kernels.attention.dsa.topk`.
    """
    from . import _kernels as weave

    return weave.dsa_topk(logits, topk)


def get_topk_fn(backend: str | TopkFn) -> TopkFn:
    """Resolve a top-k backend: a callable (e.g. a replay-manager wrapper) is used as is; a name
    (``torch``, ``flashinfer``, ``cake``) resolves through :func:`miles.kernels.attention.dsa.topk.get_dsa_topk_fn`,
    the registry the TileLang path uses too."""
    if callable(backend):
        return backend
    if backend == FUSED_TOPK_BACKEND:
        raise ValueError(
            f"{backend!r} is not a top-k backend over logits: it is the fused indexer switch "
            f"lightning_indexer(..., topk_backend={backend!r}), which scores and selects in one kernel pair "
            "without logits; the top-k backends are registered in miles.kernels.attention.dsa.topk"
        )
    return get_dsa_topk_fn(backend)


def select_topk(logits: torch.Tensor, topk: int, backend: str | TopkFn = "torch") -> torch.Tensor:
    """Select ``min(topk, num_keys)`` key indices per query row of ``logits[..., num_keys]``.

    Rows are flattened to ``[rows, num_keys]`` for the backend (the Miles replay
    convention) and the result is reshaped to ``[..., topk]``; short key axes are
    padded with ``-1``.
    """
    num_keys = logits.shape[-1]
    count = min(topk, num_keys)
    flat = logits.reshape(-1, num_keys)
    indices = get_topk_fn(backend)(flat, count).to(torch.int32)
    if count < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - count), value=-1)
    return indices.reshape(*logits.shape[:-1], topk)


def extract_topk_scores(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``logits`` at ``topk_indices`` along the last axis; ``-1`` slots become ``-inf``."""
    valid = topk_indices >= 0
    safe = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe)
    return torch.where(valid, scores, torch.full_like(scores, float("-inf")))


# --------------------------------------------------------------------------- #
# Sparse attention: flat kernels
# --------------------------------------------------------------------------- #


def _sink_or_disabled(attn_sink: torch.Tensor | None, heads: int, device) -> torch.Tensor:
    if attn_sink is None:
        return torch.full((heads,), float("-inf"), dtype=torch.float32, device=device)
    if attn_sink.shape != (heads,):
        raise ValueError(f"attn_sink must be [heads={heads}], got {tuple(attn_sink.shape)}")
    return attn_sink.to(torch.float32).contiguous()


def _flat_attention_forward(q, kv, indices, sink, sm_scale, d_v, meta: RowMeta):
    """Run the flat forward; return ``(o_flat, lse_flat, o_layout)``.

    ``o_layout`` is the output in the model layout and is the tensor the wrapper hands back.  For batched
    layouts it is allocated here in layout shape and the kernel writes through a flat view of it, so the
    returned tensor is a fresh allocation rather than a view of an intermediate (callers such as the
    DeepSeek-V4 attention apply the inverse RoPE in place on the output, which autograd rejects for views
    produced inside a custom Function).
    """
    from . import _kernels as weave

    out = None
    o_layout = None
    if meta.layout != "thd":
        o_layout = torch.empty(meta.batch, meta.seq_len, q.shape[1], d_v, dtype=q.dtype, device=q.device)
        out = o_layout.view(q.shape[0], q.shape[1], d_v)
    o_flat, lse_flat = weave.dsa_attention_forward(q, kv, indices, sink, float(sm_scale), d_v=d_v, out=out)
    return o_flat, lse_flat, (o_layout if o_layout is not None else o_flat)


def _flat_attention_backward(q, kv, o, do, indices, lse, sink, sm_scale, d_v, workspace_bytes):
    from . import _kernels as weave

    from . import reduction

    num_rows, heads, d_qk = q.shape
    num_kv = kv.shape[0]
    topk = indices.shape[1]
    do = do.contiguous()
    geo = weave.attention_geometry(q, kv, indices, d_v)
    # delta = rowsum(O * dO) in FP32: one kernel, fixed order (bit-reproducible).
    delta = weave.dsa_attention_delta_launch(o.contiguous(), do)
    dq = torch.empty_like(q)
    dkv32 = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    # FP32 partial rows per (query row, slot): one per head block, or one per row when the two head blocks of a row
    # run as a cluster and sum their contributions on chip (``geo.pair``)
    kv_partials = geo.kv_partials
    chunks = reduction.plan_row_chunks(num_rows, kv_partials * topk, d_qk, reduction.workspace_budget(workspace_bytes))
    rows_per_chunk = chunks[0][1] - chunks[0][0]
    partial = torch.empty(rows_per_chunk * kv_partials * topk, d_qk, dtype=torch.float32, device=q.device)
    # one global stable sort of (key, chunk) composites: per-chunk reducer segments without per-chunk sorts
    segments = reduction.ChunkedSegments(indices, kv_partials, rows_per_chunk, num_kv)
    for chunk, (start, end) in enumerate(chunks):
        rows = end - start
        part = partial[: rows * kv_partials * topk]
        weave.dsa_attention_backward_rows(
            q[start:end],
            kv,
            do[start:end],
            indices[start:end],
            lse[start:end],
            delta[start:end],
            dq[start:end],
            part,
            float(sm_scale),
            geo,
        )
        order_local, seg_start, seg_count = segments.segment(chunk)
        weave.dsa_segmented_reduce_launch(part, order_local, seg_start, seg_count, dkv32, split=ATTENTION_REDUCE_SPLIT)
    dkv = dkv32.to(kv.dtype)
    dsink = None
    if sink is not None:
        # d sink[h] = -sum_rows delta[row, h] * p_sink[row, h], p_sink = exp2(sink * log2 e - lse).
        dsink = -(delta * torch.exp2(sink.unsqueeze(0) * LOG2E - lse)).sum(dim=0)
    return dq, dkv, dsink


# --------------------------------------------------------------------------- #
# Sparse attention: public functional + autograd
# --------------------------------------------------------------------------- #


def _prepare_attention(q, kv, indices, layout):
    q_flat, kv_flat, idx_flat, meta = layouts.flatten_attention_inputs(q, kv, indices, layout)
    idx_flat = layouts.pad_topk(idx_flat, ATTENTION_TOPK_MULTIPLE)
    if q_flat.dtype != torch.bfloat16 or kv_flat.dtype != torch.bfloat16:
        raise TypeError("sparse attention expects BF16 q and kv")
    return q_flat, kv_flat, idx_flat, meta


def sparse_attention_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward only: returns ``(output, lse)`` in the input layout (``lse`` is base-2, sink included)."""
    q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
    if sm_scale is None:
        sm_scale = q_flat.shape[-1] ** -0.5
    sink = _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
    _o_flat, lse_flat, o_layout = _flat_attention_forward(q_flat, kv_flat, idx_flat, sink, sm_scale, d_v, meta)
    return o_layout, meta.rows_to_layout(lse_flat)


def sparse_attention_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    *,
    sm_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward only: returns ``(dq, dkv, d_attn_sink)`` in the input layout (``d_attn_sink`` is ``None`` without a sink)."""
    q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
    if sm_scale is None:
        sm_scale = q_flat.shape[-1] ** -0.5
    sink = None if attn_sink is None else _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
    o_flat, _ = layouts.flatten_rows(o, layout, kind="q")
    do_flat, _ = layouts.flatten_rows(do, layout, kind="q")
    lse_flat, _ = layouts.flatten_rows(lse, layout, kind="q")
    dq, dkv, dsink = _flat_attention_backward(
        q_flat, kv_flat, o_flat, do_flat, idx_flat, lse_flat.float(), sink, sm_scale, d_v, workspace_bytes
    )
    dkv = meta.kv_rows_to_layout(dkv)
    if layout == "thd" and kv.dim() == 3:
        dkv = dkv.unsqueeze(1)
    return meta.rows_to_layout(dq), dkv, dsink


class SparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the unified sparse attention kernels."""

    @staticmethod
    def forward(ctx, q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes):
        q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
        if sm_scale is None:
            sm_scale = q_flat.shape[-1] ** -0.5
        has_sink = attn_sink is not None
        sink = _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
        o_flat, lse_flat, o_layout = _flat_attention_forward(q_flat, kv_flat, idx_flat, sink, sm_scale, d_v, meta)
        if o_layout is not o_flat:
            # o_flat is a view of the returned layout tensor, created here in no-grad mode. DeepSeek-V4
            # applies the inverse RoPE in place on the output before the projection, so save a copy that
            # the in-place update cannot reach (the pinned TileLang function saves o.clone() for the same reason).
            o_flat = o_flat.clone()
        ctx.save_for_backward(q_flat, kv_flat, idx_flat, o_flat, lse_flat, sink)
        ctx.meta = meta
        ctx.sm_scale = float(sm_scale)
        ctx.d_v = int(d_v)
        ctx.workspace_bytes = workspace_bytes
        ctx.has_sink = has_sink
        ctx.kv_had_group_dim = layout == "thd" and kv.dim() == 3
        return o_layout

    @staticmethod
    def backward(ctx, grad_output):
        q_flat, kv_flat, idx_flat, o_flat, lse_flat, sink = ctx.saved_tensors
        meta: RowMeta = ctx.meta
        do_flat, _ = layouts.flatten_rows(grad_output, meta.layout, kind="q")
        dq, dkv, dsink = _flat_attention_backward(
            q_flat,
            kv_flat,
            o_flat,
            do_flat,
            idx_flat,
            lse_flat,
            sink if ctx.has_sink else None,
            ctx.sm_scale,
            ctx.d_v,
            ctx.workspace_bytes,
        )
        dkv = meta.kv_rows_to_layout(dkv)
        if ctx.kv_had_group_dim:
            dkv = dkv.unsqueeze(1)
        return meta.rows_to_layout(dq), dkv, dsink, None, None, None, None, None


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
    workspace_bytes: int | None = None,
) -> torch.Tensor:
    """Sparse MQA/MLA attention over top-k key indices (autograd; gradients for ``q``, ``kv``, ``attn_sink``).

    Args:
        q: BF16 queries, ``[T, H, d_qk]`` (``thd``) or ``[B, S, H, d_qk]`` (``bshd``).
        kv: BF16 latent keys/values, ``[T_kv, d_qk]`` / ``[T_kv, 1, d_qk]`` (``thd``) or ``[B, S_kv, d_qk]``.
        indices: int32 key indices per query, ``-1`` for empty slots; ``[T, topk]`` /
            ``[T, 1, topk]`` (``thd``) or batch-local ``[B, S, topk]`` (``bshd``).
        sm_scale: softmax scale (default ``d_qk ** -0.5``).
        attn_sink: optional FP32 per-head sink logit ``[H]`` (DeepSeek-V4).
        d_v: number of leading latent channels that form the value (``512``).
        workspace_bytes: FP32 partial-buffer budget for the backward's key-side
            partials (default ``MILES_DSA_BWD_WORKSPACE_BYTES`` or 2 GiB).

    Returns the BF16 attention output ``[..., H, d_v]`` in the input layout.
    """
    return SparseAttentionFunction.apply(q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes)


# --------------------------------------------------------------------------- #
# Lightning indexer
# --------------------------------------------------------------------------- #


def indexer_logits(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    layout: str = "thd",
) -> torch.Tensor:
    """Head-summed ReLU indexer logits (no autograd; the indexer gradient flows through :func:`indexer_topk_scores`).

    ``index_q`` BF16 ``[T, H, D]`` / ``[B, S, H, D]`` / ``[S, B, H, D]``, ``index_k`` BF16
    ``[T_kv, D]`` / ``[B, S_kv, D]`` / ``[S_kv, B, D]``, ``weights`` FP32 ``[T, H]`` (or
    ``[T, H, 1]``) / ``[B, S, H]`` / ``[S, B, H]``.  ``cu_seqlen_ks``/``cu_seqlen_ke``
    are int32 per-query key bounds ``[S]`` shared across the batch.  Returns FP32
    logits ``[T, T_kv]`` (``thd``) or ``[B, S, S_kv]``; keys outside a query's bounds
    are ``-inf``.
    """
    from . import _kernels as weave

    q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, layout)
    batch, seq_len, seq_len_kv = meta.batch, meta.seq_len, meta.seq_len_kv
    if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            f"cu_seqlen_ks/ke must be [{seq_len}], got {tuple(cu_seqlen_ks.shape)} / {tuple(cu_seqlen_ke.shape)}"
        )
    if q_flat.dtype != torch.bfloat16 or k_flat.dtype != torch.bfloat16:
        raise TypeError("indexer expects BF16 index_q and index_k")
    ks = cu_seqlen_ks.to(torch.int32).contiguous()
    ke = cu_seqlen_ke.to(torch.int32).contiguous()
    logits = weave.dsa_indexer_logits(
        q_flat, k_flat, w_flat, ks, ke, batch=batch, seq_len=seq_len, seq_len_kv=seq_len_kv
    )
    return logits.squeeze(0) if layout == "thd" else logits


def _flat_indexer_backward(q, k, w, indices, grad_scores, workspace_bytes):
    from . import _kernels as weave

    from . import reduction

    num_rows, heads, dim = q.shape
    num_kv = k.shape[0]
    topk = indices.shape[1]
    dq = torch.empty_like(q)
    dw = torch.empty(num_rows, heads, dtype=torch.float32, device=q.device)
    chunks = reduction.plan_row_chunks(num_rows, topk, dim, reduction.workspace_budget(workspace_bytes))
    rows_per_chunk = chunks[0][1] - chunks[0][0]
    partial = torch.empty(rows_per_chunk * topk, dim, dtype=torch.float32, device=q.device)
    dk32 = None
    for start, end in chunks:
        part = partial[: (end - start) * topk]
        # The rows kernel is the first launch of the call: the host prepares the segment glue of
        # this chunk (and the dk32 fill) while it runs instead of leaving the GPU idle before it.
        weave.dsa_indexer_backward_rows(
            q[start:end],
            k,
            w[start:end],
            indices[start:end],
            grad_scores[start:end],
            dq[start:end],
            dw[start:end],
            part,
        )
        if dk32 is None:
            dk32 = torch.zeros(num_kv, dim, dtype=torch.float32, device=q.device)
        reduction.deterministic_scatter_add(part, indices[start:end].reshape(-1), dk32, split=INDEXER_REDUCE_SPLIT)
    return dq, dw, dk32


def indexer_backward(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    *,
    layout: str = "thd",
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward only: ``(d_index_q BF16, d_weights FP32, d_index_k FP32)`` in the input layout.

    ``topk_indices`` and ``grad_scores`` are ``[T, topk]`` (``thd``) or batch-major
    ``[B, S, topk]`` for both batched layouts; ``-1`` slots contribute nothing.
    """
    q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, layout)
    from . import _kernels as weave

    idx_flat = layouts.pad_topk(
        layouts.flatten_topk_indices(topk_indices, meta), weave.indexer_key_block(q_flat.shape[1])
    )
    grad_flat = layouts.flatten_scores(grad_scores, meta)
    if grad_flat.shape[1] != idx_flat.shape[1]:
        grad_flat = torch.nn.functional.pad(grad_flat, (0, idx_flat.shape[1] - grad_flat.shape[1]), value=0.0)
    # Empty slots carry no gradient (their upstream scores are -inf, so grad may be NaN / inf there):
    # the rows kernel loads an exact 0.0 for idx == -1, so no host-side masking launch is needed.
    dq, dw, dk = _flat_indexer_backward(q_flat, k_flat, w_flat, idx_flat, grad_flat.contiguous(), workspace_bytes)
    dk = meta.kv_rows_to_layout(dk)
    if layout == "thd" and index_k.dim() == 3:
        dk = dk.unsqueeze(1)
    dw = meta.rows_to_layout(dw)
    if layout == "thd" and weights.dim() == 3:
        dw = dw.unsqueeze(-1)
    return meta.rows_to_layout(dq), dw, dk


def _indexer_scores_backward(ctx, grad_scores):
    """The backward shared by the selected-score Functions: ``(d index_q, d index_k, d weights)`` from the saved
    ``(index_q, index_k, weights, topk_indices)`` -- nothing derived from the dense logits is needed."""
    index_q, index_k, weights, topk_indices = ctx.saved_tensors
    dq, dw, dk = indexer_backward(
        index_q,
        index_k,
        weights,
        topk_indices,
        grad_scores,
        layout=ctx.layout,
        workspace_bytes=ctx.workspace_bytes,
    )
    return dq, dk, dw.to(weights.dtype)


class IndexerScoresFunction(torch.autograd.Function):
    """Top-k indexer scores with the unified deterministic backward."""

    @staticmethod
    def forward(ctx, index_q, index_k, weights, topk_indices, logits, layout, workspace_bytes):
        ctx.save_for_backward(index_q, index_k, weights, topk_indices)
        ctx.layout = layout
        ctx.workspace_bytes = workspace_bytes
        return extract_topk_scores(logits, topk_indices)

    @staticmethod
    def backward(ctx, grad_scores):
        dq, dk, dw = _indexer_scores_backward(ctx, grad_scores)
        return dq, dk, dw, None, None, None, None


def indexer_topk_scores(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    logits: torch.Tensor,
    *,
    layout: str = "thd",
    workspace_bytes: int | None = None,
) -> torch.Tensor:
    """Gather the selected indexer scores from ``logits`` (autograd to ``index_q``, ``index_k``, ``weights``).

    ``logits`` is ``[T, T_kv]`` or ``[B, S, S_kv]`` as returned by :func:`indexer_logits`;
    ``topk_indices`` is ``[T, topk]`` or batch-local ``[B, S, topk]``.  Empty slots yield
    ``-inf`` scores and receive no gradient.
    """
    return IndexerScoresFunction.apply(index_q, index_k, weights, topk_indices, logits, layout, workspace_bytes)


def _causal_segments(ks: torch.Tensor, ke: torch.Tensor, num_rows: int, num_kv: int, ratio: int):
    """The segment table ``(cu_seqlens_q, cu_seqlens_k)`` (int32 ``[S + 1]``) behind ``thd`` causal key bounds.

    Segments start where ``cu_seqlen_ks`` changes; segment ``s`` owns the keys ``[ks_s, ks_{s+1})`` (the last one up
    to ``num_kv``).  Row ``u`` of a segment with ``Lq`` queries and ``Lk`` keys must satisfy
    ``ke - ks == min(Lk, max(0, floor((offset + u + 1) / ratio)))`` with ``offset = Lk - Lq`` at ``ratio == 1`` and
    ``0`` otherwise: the causal prefix of GLM-5 / DeepSeek-V3.2 and the compressed-key prefix of DeepSeek-V4.  Any
    other pattern (a sliding-window start, a block of query rows cut before its sequence end, context-parallel row
    shuffles) raises ``ValueError`` naming the first offending row; the dense path serves those calls.  Reads the
    two bound vectors on the host (one device-to-host copy per call).
    """
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
        raise ValueError(f"ratio must be a positive integer, got {ratio!r}")
    if tuple(ks.shape) != (num_rows,) or tuple(ke.shape) != (num_rows,):
        raise ValueError(f"cu_seqlen_ks/ke must be [{num_rows}], got {tuple(ks.shape)} / {tuple(ke.shape)}")
    if num_rows == 0:
        empty = torch.zeros(1, dtype=torch.int32, device=ks.device)
        return empty, empty.clone()
    ks64 = ks.detach().to(device="cpu", dtype=torch.int64)
    ke64 = ke.detach().to(device="cpu", dtype=torch.int64)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), torch.nonzero(ks64[1:] != ks64[:-1]).flatten() + 1])
    ends = torch.cat([starts[1:], torch.tensor([num_rows])])
    kstarts = ks64[starts]
    kends = torch.cat([kstarts[1:], torch.tensor([num_kv])])
    if kstarts[0] < 0 or (kends < kstarts).any():
        raise ValueError(
            "the fused indexer top-k needs key starts that increase with the packed sequences and end within "
            f"index_k ({num_kv} keys), got segment key starts {kstarts[:8].tolist()}"
            f"{'...' if len(kstarts) > 8 else ''}"
        )
    lq, lk = ends - starts, kends - kstarts
    seg = torch.repeat_interleave(torch.arange(starts.numel()), lq)
    u = torch.arange(num_rows) - starts[seg]
    offset = (lk - lq) if ratio == 1 else torch.zeros_like(lk)
    visible = torch.div(offset[seg] + u + 1, ratio, rounding_mode="floor").clamp(min=0)
    visible = torch.minimum(visible, lk[seg])
    mismatch = torch.nonzero(ke64 != ks64 + visible).flatten()
    if mismatch.numel():
        t = int(mismatch[0])
        s = int(seg[t])
        raise ValueError(
            f"row {t} has the key range [{int(ks64[t])}, {int(ke64[t])}) but the fused indexer top-k scores the "
            f"causal prefix [{int(ks64[t])}, {int(ks64[t] + visible[t])}) of its segment {s} ({int(lq[s])} queries, "
            f"{int(lk[s])} keys, ratio {ratio}); use the dense path for other key bounds"
        )
    cu_q = torch.cat([starts, torch.tensor([num_rows])]).to(torch.int32)
    cu_k = torch.cat([kstarts, torch.tensor([num_kv])]).to(torch.int32)
    return cu_q.to(ks.device), cu_k.to(ks.device)


class FusedIndexerScoresFunction(torch.autograd.Function):
    """``thd`` fused indexer scoring + top-k with the backward of :class:`IndexerScoresFunction`."""

    @staticmethod
    def forward(ctx, index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, ratio, workspace_bytes):
        from . import _kernels as weave

        q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, "thd")
        if q_flat.dtype != torch.bfloat16 or k_flat.dtype != torch.bfloat16:
            raise TypeError("indexer expects BF16 index_q and index_k")
        ks = cu_seqlen_ks.to(torch.int32).contiguous()
        ke = cu_seqlen_ke.to(torch.int32).contiguous()
        cu_q, cu_k = _causal_segments(ks, ke, meta.num_rows, meta.num_kv, ratio)
        # softmax_scale = 1.0: the dense path scores relu(q . k) without a scale (callers fold theirs into the inputs).
        local, scores = weave.dsa_indexer_topk(
            q_flat, k_flat, w_flat, cu_q, cu_k, topk=topk, softmax_scale=1.0, ratio=ratio
        )
        # The kernel emits segment-local key ids; the consumers and the backward take global key rows (-1 stays).
        topk_indices = torch.where(local >= 0, local + ks.unsqueeze(1), local)
        ctx.save_for_backward(index_q, index_k, weights, topk_indices)
        ctx.layout = "thd"
        ctx.workspace_bytes = workspace_bytes
        ctx.mark_non_differentiable(topk_indices)
        return scores, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, _grad_indices):
        dq, dk, dw = _indexer_scores_backward(ctx, grad_scores)
        return dq, dk, dw, None, None, None, None, None


def fused_indexer_topk_scores(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    *,
    ratio: int = 1,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused indexer scoring + top-k (``thd``): ``(scores, topk_indices)`` with gradients for q, k and weights.

    The ``cake_fused`` switch of :func:`lightning_indexer`: the generated scan + finalize stages (Blackwell SM100a /
    SM103a) score the keys in FP32 and select each row's top-k on chip, so the dense ``[T, T_kv]`` logits are never
    materialised.  The scores are those of :func:`indexer_logits` at the selection (FP32, no scale; the two kernels
    round their reductions in different orders, so a near-tie may select differently from :func:`cake_topk`).
    ``topk_indices`` is int32 ``[T, topk]`` of global key rows in ascending order with the ``-1`` tail (consumers
    gather by slot), ``scores`` FP32 ``[T, topk]`` with ``-inf`` at the tail.  Limits: ``index_q`` with 32 heads of
    dim 128, ``1 <= topk <= 512``, key bounds that are each packed sequence's causal (``ratio == 1``) or
    compressed-key (``ratio >= 2``) prefix; everything else raises ``ValueError`` -- there is no silent fallback.
    Bit-identical across calls; the backward is :func:`indexer_backward`, as for every other backend.
    """
    return FusedIndexerScoresFunction.apply(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, ratio, workspace_bytes
    )


def lightning_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    *,
    layout: str = "thd",
    topk_backend: str | TopkFn = "torch",
    topk_indices: torch.Tensor | None = None,
    ratio: int = 1,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer logits -> top-k selection -> differentiable selected scores.

    ``topk_backend`` is a registered name or any ``fn(logits, topk)`` callable (a
    replay-manager wrapper, for instance); ``topk_indices`` bypasses selection
    entirely (rollout replay).  ``topk_backend="cake_fused"`` (``thd`` only) replaces
    the three steps by :func:`fused_indexer_topk_scores` -- no dense logits; the key
    bounds must be each sequence's causal / compressed-key prefix (``ratio``) -- except
    under ``topk_indices``, where the dense path scores the given selection as for
    every backend.  Returns ``(scores, topk_indices)``: ``[T, topk]`` (``thd``) or
    batch-major ``[B, S, topk]``.
    """
    fused = isinstance(topk_backend, str) and topk_backend == FUSED_TOPK_BACKEND
    if fused and topk_indices is None:
        if layout != "thd":
            raise ValueError(f"topk_backend={FUSED_TOPK_BACKEND!r} supports layout='thd' only, got {layout!r}")
        return fused_indexer_topk_scores(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, ratio=ratio, workspace_bytes=workspace_bytes
        )
    if ratio != 1 and not fused:
        raise ValueError(f"ratio applies to topk_backend={FUSED_TOPK_BACKEND!r} only, got ratio={ratio!r}")
    logits = indexer_logits(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, layout=layout)
    if topk_indices is None:
        topk_indices = select_topk(logits, topk, topk_backend)
    scores = indexer_topk_scores(
        index_q,
        index_k,
        weights,
        topk_indices,
        logits,
        layout=layout,
        workspace_bytes=workspace_bytes,
    )
    return scores, topk_indices
