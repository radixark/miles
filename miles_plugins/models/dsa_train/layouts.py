"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Layout adapters between the Miles model layouts and the flattened kernel contract.

The kernels see queries as rows ``[num_rows, heads, ...]``, keys as rows
``[num_kv, ...]`` and ``indices[row, slot]`` as *global* key rows.  Three model
layouts map onto that contract:

``"thd"``
    Packed sequences (GLM-5 / DeepSeek-V3.2).  Queries ``[T, H, D]``, keys
    ``[T_kv, D]`` or ``[T_kv, 1, D]`` (single KV group), indices ``[T, topk]`` or
    ``[T, 1, topk]``.  Already flat; the group dimension is squeezed.
``"bshd"``
    Batched sequences (DeepSeek-V4 attention).  Queries ``[B, S, H, D]``, keys
    ``[B, S_kv, D]``, indices ``[B, S, topk]`` local to the batch element.  Rows are
    flattened batch-major and indices are offset by ``b * S_kv``.
``"sbhd"``
    Sequence-first batches (DeepSeek-V4 indexer inputs).  Queries ``[S, B, H, D]``,
    keys ``[S_kv, B, D]``, weights ``[S, B, H]``; indices/scores/logits stay
    batch-major ``[B, S, ...]``.  Rows are permuted to batch-major before flattening.

Padded slots are ``-1`` in every layout and stay ``-1`` after globalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LAYOUTS = ("thd", "bshd", "sbhd")


def _check_layout(layout: str) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def globalize_indices(indices: torch.Tensor, seq_len_kv: int) -> torch.Tensor:
    """Offset batch-local ``[B, S, topk]`` indices by ``b * seq_len_kv`` and flatten to ``[B * S, topk]``.

    ``-1`` padding is preserved.  Always returns a fresh contiguous ``int32`` tensor.
    """
    if indices.dim() != 3:
        raise ValueError(f"batched indices must be [B, S, topk], got {tuple(indices.shape)}")
    batch = indices.shape[0]
    offsets = (torch.arange(batch, device=indices.device, dtype=torch.int64) * seq_len_kv).view(batch, 1, 1)
    flat = torch.where(indices < 0, torch.full_like(indices, -1, dtype=torch.int64), indices.to(torch.int64) + offsets)
    return flat.reshape(batch * indices.shape[1], indices.shape[2]).to(torch.int32).contiguous()


def pad_topk(indices: torch.Tensor, multiple: int) -> torch.Tensor:
    """Right-pad the slot axis of ``[rows, topk]`` indices with ``-1`` to a multiple of ``multiple``."""
    topk = indices.shape[-1]
    padded = max(multiple, (topk + multiple - 1) // multiple * multiple)
    if padded == topk:
        return indices.contiguous()
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1).contiguous()


@dataclass(frozen=True)
class RowMeta:
    """How flattened query rows map back to the model layout."""

    layout: str
    batch: int
    seq_len: int
    seq_len_kv: int

    @property
    def num_rows(self) -> int:
        return self.batch * self.seq_len

    @property
    def num_kv(self) -> int:
        return self.batch * self.seq_len_kv

    def rows_to_layout(self, rows: torch.Tensor) -> torch.Tensor:
        """Map a per-query-row tensor ``[num_rows, ...]`` back to the model layout."""
        if self.layout == "thd":
            return rows
        batched = rows.reshape(self.batch, self.seq_len, *rows.shape[1:])
        if self.layout == "bshd":
            return batched
        return batched.transpose(0, 1).contiguous()  # sbhd

    def kv_rows_to_layout(self, rows: torch.Tensor) -> torch.Tensor:
        """Map a per-key-row tensor ``[num_kv, ...]`` back to the model layout."""
        if self.layout == "thd":
            return rows
        batched = rows.reshape(self.batch, self.seq_len_kv, *rows.shape[1:])
        if self.layout == "bshd":
            return batched
        return batched.transpose(0, 1).contiguous()  # sbhd

    def batch_major_to_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Map a batch-major per-row tensor ``[B, S, ...]`` to the model layout (identity except ``thd``)."""
        if self.layout == "thd":
            return tensor.reshape(self.num_rows, *tensor.shape[2:])
        return tensor


def flatten_rows(tensor: torch.Tensor, layout: str, *, kind: str) -> tuple[torch.Tensor, RowMeta]:
    """Flatten a query-side (``kind="q"``) or key-side (``kind="kv"``) tensor to rows.

    Returns the contiguous flat tensor ``[rows, ...]`` and the row metadata.  For the
    key side the metadata's ``seq_len`` is unknown and set to ``0``.
    """
    _check_layout(layout)
    if layout == "thd":
        flat = tensor.contiguous()
        return flat, RowMeta("thd", 1, flat.shape[0] if kind == "q" else 0, flat.shape[0] if kind == "kv" else 0)
    if layout == "bshd":
        batch, seq = tensor.shape[0], tensor.shape[1]
        flat = tensor.reshape(batch * seq, *tensor.shape[2:]).contiguous()
    else:  # sbhd
        seq, batch = tensor.shape[0], tensor.shape[1]
        flat = tensor.transpose(0, 1).reshape(batch * seq, *tensor.shape[2:]).contiguous()
    return flat, RowMeta(layout, batch, seq if kind == "q" else 0, seq if kind == "kv" else 0)


def flatten_attention_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RowMeta]:
    """Flatten sparse-attention inputs to ``(q[T,H,D], kv[T_kv,D], indices[T,topk], meta)``."""
    _check_layout(layout)
    if layout == "thd":
        if kv.dim() == 3:
            if kv.shape[1] != 1:
                raise ValueError(f"thd KV must have a single group, got {tuple(kv.shape)}")
            kv = kv.squeeze(1)
        if indices.dim() == 3:
            if indices.shape[1] != 1:
                raise ValueError(f"thd indices must have a single group, got {tuple(indices.shape)}")
            indices = indices.squeeze(1)
        if q.dim() != 3 or kv.dim() != 2 or indices.dim() != 2:
            raise ValueError("thd expects q [T, H, D], kv [T_kv, D], indices [T, topk]")
        meta = RowMeta("thd", 1, q.shape[0], kv.shape[0])
        return q.contiguous(), kv.contiguous(), indices.to(torch.int32).contiguous(), meta
    if layout == "sbhd":
        raise ValueError("sparse attention takes 'thd' or 'bshd'; 'sbhd' is an indexer-input layout")
    if q.dim() != 4 or kv.dim() != 3 or indices.dim() != 3:
        raise ValueError("bshd expects q [B, S, H, D], kv [B, S_kv, D], indices [B, S, topk]")
    batch, seq, heads, dim = q.shape
    if kv.shape[0] != batch or indices.shape[0] != batch or indices.shape[1] != seq:
        raise ValueError("bshd batch/sequence extents disagree between q, kv and indices")
    seq_kv = kv.shape[1]
    meta = RowMeta("bshd", batch, seq, seq_kv)
    return (
        q.reshape(batch * seq, heads, dim).contiguous(),
        kv.reshape(batch * seq_kv, kv.shape[2]).contiguous(),
        globalize_indices(indices, seq_kv),
        meta,
    )


def flatten_indexer_inputs(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RowMeta]:
    """Flatten indexer inputs to ``(index_q[T,H,D], index_k[T_kv,D], weights[T,H], meta)``."""
    _check_layout(layout)
    if layout == "thd":
        if index_k.dim() == 3:
            if index_k.shape[1] != 1:
                raise ValueError(f"thd index_k must have a single group, got {tuple(index_k.shape)}")
            index_k = index_k.squeeze(1)
        if weights.dim() == 3:
            if weights.shape[-1] != 1:
                raise ValueError(f"thd weights must be [T, H] or [T, H, 1], got {tuple(weights.shape)}")
            weights = weights.squeeze(-1)
        meta = RowMeta("thd", 1, index_q.shape[0], index_k.shape[0])
        return index_q.contiguous(), index_k.contiguous(), weights.float().contiguous(), meta
    if layout == "bshd":
        batch, seq = index_q.shape[0], index_q.shape[1]
        seq_kv = index_k.shape[1]
        meta = RowMeta("bshd", batch, seq, seq_kv)
        return (
            index_q.reshape(batch * seq, *index_q.shape[2:]).contiguous(),
            index_k.reshape(batch * seq_kv, index_k.shape[2]).contiguous(),
            weights.reshape(batch * seq, weights.shape[2]).float().contiguous(),
            meta,
        )
    seq, batch = index_q.shape[0], index_q.shape[1]
    seq_kv = index_k.shape[0]
    if index_k.shape[1] != batch or weights.shape[0] != seq or weights.shape[1] != batch:
        raise ValueError("sbhd batch/sequence extents disagree between index_q, index_k and weights")
    meta = RowMeta("sbhd", batch, seq, seq_kv)
    return (
        index_q.transpose(0, 1).reshape(batch * seq, *index_q.shape[2:]).contiguous(),
        index_k.transpose(0, 1).reshape(batch * seq_kv, index_k.shape[2]).contiguous(),
        weights.transpose(0, 1).reshape(batch * seq, weights.shape[2]).float().contiguous(),
        meta,
    )


def flatten_topk_indices(indices: torch.Tensor, meta: RowMeta) -> torch.Tensor:
    """Flatten batch-major (or ``thd``) top-k indices to global ``[num_rows, topk]`` rows."""
    if meta.layout == "thd":
        if indices.dim() == 3:
            indices = indices.squeeze(1)
        return indices.to(torch.int32).contiguous()
    return globalize_indices(indices, meta.seq_len_kv)


def flatten_scores(scores: torch.Tensor, meta: RowMeta) -> torch.Tensor:
    """Flatten batch-major (or ``thd``) per-slot scores/gradients to ``[num_rows, topk]``."""
    if meta.layout == "thd":
        if scores.dim() == 3:
            scores = scores.squeeze(1)
        return scores.float().contiguous()
    return scores.reshape(meta.num_rows, scores.shape[-1]).float().contiguous()
