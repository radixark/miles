"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic scatter-add of per-slot FP32 partials into key rows.

Given ``partial[p, :]`` for pairs ``p`` and the key row ``keys[p]`` each pair
belongs to (``-1`` for padded slots), :func:`deterministic_scatter_add` computes
``out[k, :] += sum_{p : keys[p] == k} partial[p, :]`` with a summation order that
is a pure function of ``keys``: pairs are stably sorted by key, so ties keep their
pair order, and one CTA per key row adds them sequentially.  Repeated runs give
bit-identical results; the atomics they replace do not.  No host synchronisation
is needed: segment starts and counts come from ``searchsorted`` on the sorted keys.

Workspace planning (:func:`plan_row_chunks`) bounds the partial buffer to a byte
budget by processing query rows in chunks; chunks are reduced in order, which keeps
the overall accumulation deterministic.
"""

from __future__ import annotations

import os

import torch

DEFAULT_WORKSPACE_BYTES = 2 << 30
_WORKSPACE_ENV = "MILES_DSA_BWD_WORKSPACE_BYTES"


def workspace_budget(workspace_bytes: int | None) -> int:
    """Resolve the partial-buffer budget from the argument or ``MILES_DSA_BWD_WORKSPACE_BYTES``."""
    if workspace_bytes is not None:
        return int(workspace_bytes)
    return int(os.environ.get(_WORKSPACE_ENV, DEFAULT_WORKSPACE_BYTES))


def plan_row_chunks(num_rows: int, partials_per_row: int, dim: int, workspace_bytes: int) -> list[tuple[int, int]]:
    """Split ``num_rows`` into ``(start, end)`` chunks whose FP32 partials fit ``workspace_bytes``."""
    bytes_per_row = partials_per_row * dim * 4
    rows_per_chunk = max(1, min(num_rows, workspace_bytes // max(bytes_per_row, 1)))
    return [(start, min(num_rows, start + rows_per_chunk)) for start in range(0, num_rows, rows_per_chunk)]


def segment_pairs(keys: torch.Tensor, num_out: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable-sort pair keys; return ``(order, seg_start, seg_count)`` indexed by key row.

    ``order[p]`` is the pair index at sorted position ``p``; key ``k`` owns sorted
    positions ``[seg_start[k], seg_start[k] + seg_count[k])``.  Pairs with key ``-1``
    sort first and belong to no key.  Everything is computed on the device without a
    host synchronisation or atomics (``searchsorted`` on the sorted keys).
    """
    if keys.dim() != 1:
        raise ValueError("keys must be a flat [num_pairs] tensor")
    keys32 = keys.to(torch.int32)
    sorted_keys, order = torch.sort(keys32, stable=True)
    probe = torch.arange(num_out, device=keys.device, dtype=torch.int32)
    starts = torch.searchsorted(sorted_keys, probe)
    ends = torch.searchsorted(sorted_keys, probe, right=True)
    return order.to(torch.int32), starts.to(torch.int32).contiguous(), (ends - starts).to(torch.int32).contiguous()


def deterministic_scatter_add(partial: torch.Tensor, keys: torch.Tensor, out: torch.Tensor) -> None:
    """``out[keys[p], :] += partial[p, :]`` in a fixed order (see module docstring).

    ``partial`` is ``[num_pairs, dim]`` FP32, ``keys`` ``[num_pairs]`` integer with
    ``-1`` for pairs to skip, ``out`` ``[num_out, dim]`` FP32 (updated in place).
    """
    from . import _kernels as weave

    if partial.dim() != 2 or out.dim() != 2 or partial.shape[1] != out.shape[1]:
        raise ValueError("partial and out must be [*, dim] with the same dim")
    if partial.shape[0] != keys.numel():
        raise ValueError("keys must have one entry per partial row")
    if partial.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("deterministic scatter-add works on FP32 partials and outputs")
    if partial.shape[1] % 4 != 0:
        raise ValueError(f"segmented reduce needs dim divisible by 4, got {partial.shape[1]}")
    if keys.numel() == 0 or out.shape[0] == 0:
        return
    order, seg_start, seg_count = segment_pairs(keys, int(out.shape[0]))
    weave.dsa_segmented_reduce_launch(partial.contiguous(), order, seg_start, seg_count, out)
