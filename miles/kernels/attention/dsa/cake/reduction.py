"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic scatter-add of per-slot FP32 partials into key rows.

Given ``partial[p, :]`` for pairs ``p`` and the key row ``keys[p]`` each pair
belongs to (``-1`` for padded slots), :func:`deterministic_scatter_add` computes
``out[k, :] += sum_{p : keys[p] == k} partial[p, :]`` with a summation order that
is a pure function of ``keys``: pairs are stably sorted by key, so ties keep their
pair order, and each key row is added either sequentially (``split=1``) or as a
fixed 4-way tree over its segment (``split=4``: four contiguous quarters summed in
pair order, then combined in quarter order).  Repeated runs give bit-identical
results; the atomics they replace do not.  No host synchronisation is needed: the
segment bounds come from one ``searchsorted`` on the sorted keys.

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
    # Narrow sort keys when every key (and the probe ``num_out``) fits int16: half the radix passes,
    # the same stable order (the cast is monotonic on [-1, num_out]).
    sort_keys = keys32.to(torch.int16) if num_out + 1 <= 32767 else keys32
    sorted_keys, order = torch.sort(sort_keys, stable=True)
    # One search over ``0 .. num_out`` gives every segment's start and (as the next start) its end;
    # the final probe ``num_out`` lands past the last key.
    probe = torch.arange(num_out + 1, device=keys.device, dtype=sort_keys.dtype)
    bounds = torch.searchsorted(sorted_keys, probe, out_int32=True)
    return order.to(torch.int32), bounds[:-1], (bounds[1:] - bounds[:-1]).contiguous()


def deterministic_scatter_add(partial: torch.Tensor, keys: torch.Tensor, out: torch.Tensor, *, split: int = 1) -> None:
    """``out[keys[p], :] += partial[p, :]`` in a fixed order (see module docstring).

    ``partial`` is ``[num_pairs, dim]`` FP32, ``keys`` ``[num_pairs]`` integer with
    ``-1`` for pairs to skip, ``out`` ``[num_out, dim]`` FP32 (updated in place).
    ``split`` selects the per-key reduction order: ``1`` sequential, ``4`` the fixed
    4-way tree (``dsa_segmented_reduce_split4``).
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
    weave.dsa_segmented_reduce_launch(partial.contiguous(), order, seg_start, seg_count, out, split=split)


class ChunkedSegments:
    """Per-chunk reducer segments from ONE global stable sort (the chunk loop of the attention backward).

    The attention backward processes query rows in chunks of ``rows_per_chunk`` and reduces each chunk's
    partials ``partial[(row_local * num_head_blocks + hb) * topk + slot, :]`` into ``out[key, :]``.  Sorting each
    chunk's keys separately costs ~14 launches per chunk; instead the composite key ``key * num_chunks + chunk``
    of every pair is stably sorted once.  Within one ``(key, chunk)`` segment the pairs then appear in ascending
    global pair order -- exactly the sequence the per-chunk stable sort produced -- so the reduction is
    bit-identical to the per-chunk plan for the same chunking.  Padded pairs (``key == -1``) map to negative
    composites and sort before every probe.  ``order_local`` is the pair index inside its chunk (``order %
    pairs_per_chunk``: every chunk but the last holds exactly ``pairs_per_chunk`` pairs), the row index the
    reducer uses into the chunk's partial buffer.  Segment starts/counts of all chunks come from two
    ``searchsorted`` calls; :meth:`segment` is launch-free.
    """

    def __init__(self, indices: torch.Tensor, num_head_blocks: int, rows_per_chunk: int, num_out: int):
        num_rows, topk = indices.shape
        device = indices.device
        num_chunks = (num_rows + rows_per_chunk - 1) // rows_per_chunk
        comp_dtype = torch.int32 if num_out * num_chunks < 2**31 else torch.int64
        chunk_of_row = torch.arange(num_rows, device=device, dtype=comp_dtype) // rows_per_chunk
        comp = indices.to(comp_dtype) * num_chunks + chunk_of_row[:, None]
        comp = comp.unsqueeze(1).expand(num_rows, num_head_blocks, topk).reshape(-1)
        sorted_comp, order = torch.sort(comp, stable=True)
        pairs_per_chunk = rows_per_chunk * num_head_blocks * topk
        self.order_local = (order % pairs_per_chunk).to(torch.int32)
        probe = (
            torch.arange(num_chunks, device=device, dtype=comp_dtype)[:, None]
            + (torch.arange(num_out, device=device, dtype=comp_dtype) * num_chunks)[None, :]
        )
        starts = torch.searchsorted(sorted_comp, probe.reshape(-1), out_int32=True).view(num_chunks, num_out)
        ends = torch.searchsorted(sorted_comp, probe.reshape(-1), right=True, out_int32=True).view(num_chunks, num_out)
        self.seg_start = starts
        self.seg_count = ends - starts
        self.num_chunks = num_chunks

    def segment(self, chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(order_local, seg_start, seg_count)`` for ``chunk`` (contiguous int32 rows, no launch)."""
        return self.order_local, self.seg_start[chunk], self.seg_count[chunk]
