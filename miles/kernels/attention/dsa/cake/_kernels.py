"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Host side of the generated DSA training kernels: geometry rules and named launches.

The flat contract every kernel sees: query rows ``[num_rows, heads, d]``, global key rows
``[num_kv, d]`` and per-row key indices ``[num_rows, topk]`` (``-1`` pads).  The kernel variant is a
pure function of the shape (heads per CTA, MLA tail, padded indexer heads), mirroring the generator's
host code one to one so the exported kernels are launched exactly as they were validated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ._jit import MODULES, device_arch, kernel, prebuild

LOG2E = 1.4426950408889634
INDEXER_HEAD_DIM = 128
INDEXER_HEADS = (8, 16, 32, 64)
ATTENTION_SHAPES = ((512, 64), (512, 0))  # (d_v, d_tail)
ATTENTION_D_LATENT = 512  # the K = V latent width every attention stage shares; d_qk = 512 + d_tail
ATTENTION_D_ROPE = 64  # the MLA rope tail width (d_tail 64)
# The tcgen05 forward stages ``dsa_attention_fwd_tc_h{H}_d512t{d_tail}`` (one CTA per query row, 64 query rows per
# token, the generator's ``dsa_h64_fwd`` schedule on the flat contract) exist for these head counts on these
# architectures; every other head count runs the register-MMA forward stages.  ``MILES_DSA_ATTENTION_TC=0`` forces the
# register-MMA forward everywhere.
ATTENTION_TC_HEADS = (8, 16, 32, 64)
ATTENTION_TC_ARCHES = ("sm_100a", "sm_103a")
# The tcgen05 backward rows stages ``dsa_attention_bwd_tc_h{H}_d512t{d_tail}`` (one CTA per query row holding all of the
# row's heads; the key-side dKV is accumulated over the heads in tensor memory and written as ONE FP32 partial per
# (row, slot) for the sorted segmented reducer) exist for rows with exactly 64 heads (the generator's ``dsa_h64_bwd``
# schedule, heads on the MMA M axis) and exactly 8, 16 or 32 heads (the generator's swap-AB ``dsa_swapab_bwd`` schedule, keys
# on the M axis) on ``ATTENTION_TC_ARCHES``; every other head count runs the register-MMA rows kernels (h16 / h32 / the h32x2
# cluster pair).  All stages share one launch ABI.  ``MILES_DSA_ATTENTION_BWD_TC=0`` forces the register-MMA backward
# everywhere.  Numerics: FP32 accumulation in the tensor-core order (deterministic run to run, within the register-MMA
# stages' tolerances).
ATTENTION_TC_BWD_HEADS = (8, 16, 32, 64)
TOPK_STAGE_PREFIX = "dsa_topk_fp32_"
INDEXER_TOPK_STAGE_PREFIX = "dsa_indexer_topk_"
# The tcgen05 lightning-indexer logits stages ``dsa_indexer_logits_tc_h{H}`` (keys on M = 128 via TMA key tiles,
# ``INDEXER_TC_QUERY_COLUMNS // H`` queries x H heads packed on N = 256, FP32 accumulation in tensor memory, the weighted-ReLU
# head sum in the register-MMA stage's order) exist for these indexer head counts on ``ATTENTION_TC_ARCHES``; every other
# case runs the register-MMA logits stage ``dsa_indexer_logits_h{H}``.  Both are followed by the same ``dsa_indexer_clean``
# launch.  ``MILES_DSA_INDEXER_TC=0`` forces the register-MMA stage everywhere.  Numerics: FP32 dot products in the
# tensor-core order (deterministic run to run, within the register-MMA stage's tolerances).
INDEXER_TC_HEADS = (8, 16, 32, 64)
INDEXER_TC_QUERY_COLUMNS = 256  # (query, head) columns per CTA of the tcgen05 stage: queries per CTA = 256 // heads
# The tcgen05 Lightning-indexer backward rows stages ``dsa_indexer_bwd_tc_h{H}`` (one CTA per query row; logits, dQ^T
# and dK^T on tcgen05, FP32 accumulation, one FP32 ``[128]`` partial per (row, slot) for the split-4 reducer -- the same
# contract and numerics class as ``dsa_indexer_bwd_h{H}``) exist for rows with exactly 32 or 64 indexer heads on
# ``ATTENTION_TC_ARCHES``; every other head count runs the register-MMA stages.  ``MILES_DSA_INDEXER_BWD_TC=0`` forces the
# register-MMA indexer backward everywhere.
INDEXER_TC_BWD_HEADS = (32, 64)
# The tcgen05 rows kernel (one CTA per query row, one CTA per SM) amortises a per-row prologue / epilogue over the row's
# 128-key tiles: measured on GB300 it is 1.15-1.25x faster than the register-MMA kernel at 32 heads x topk 2048 and at
# 64 heads x topk 512, but 4-7 % slower at 32 heads x topk 512 (4 tiles), so 32-head rows below this (padded) topk keep the
# register-MMA stage.
INDEXER_TC_BWD_MIN_TOPK = {32: 1024, 64: 0}

# (arch, family) pairs whose stages were prebuilt (or deliberately left lazy) in this process.
_PREBUILT: set[tuple[str, str]] = set()


def _stage_family(name: str) -> str:
    """The prebuild family of a registry stage: ``"topk"`` = the ``dsa_topk_fp32_*`` capacity variants,
    ``"indexer_topk"`` = the two ``dsa_indexer_topk_*`` fused scoring + top-k stages, ``"attention"`` = everything
    else (sparse attention, indexer logits / backward, clean, segmented reduce)."""
    if name.startswith(TOPK_STAGE_PREFIX):
        return "topk"
    if name.startswith(INDEXER_TOPK_STAGE_PREFIX):
        return "indexer_topk"
    return "attention"


def _family_stages(family: str) -> tuple[str, ...]:
    """The registry stages one family prebuilds (see :func:`_stage_family`)."""
    return tuple(stage for stage in MODULES if _stage_family(stage) == family)


def _kernel(name: str, device):
    """The native kernel for stage ``name`` on ``device``'s architecture.

    The first use of a stage family on an architecture prebuilds that family's stages in parallel (about a
    minute per stage; the fused indexer scan takes a few minutes), so the attention / indexer operators never
    compile the top-k variants, the top-k backend never compiles the attention kernels and the fused indexer
    top-k compiles only its own two stages; ``MILES_DSA_TRAIN_PREBUILD=0`` builds each stage lazily.
    """
    arch = device_arch(device)
    family = _stage_family(name)
    if (arch, family) not in _PREBUILT:
        _PREBUILT.add((arch, family))
        if os.environ.get("MILES_DSA_TRAIN_PREBUILD", "1") != "0":
            prebuild(arch, stages=_family_stages(family))
    return kernel(name, arch)


def _attention_name(kind: str, block_h: int, d_v: int, d_tail: int) -> str:
    return f"dsa_attention_{kind}_h{block_h}_d{d_v}t{d_tail}"


def _attention_pair_name(d_v: int, d_tail: int) -> str:
    """The 2-CTA-cluster backward rows kernel of rows with two 32-head blocks (33-64 heads)."""
    return f"dsa_attention_bwd_h32x2_d{d_v}t{d_tail}"


# --------------------------------------------------------------------------- #
# sparse attention
# --------------------------------------------------------------------------- #


def attention_block_h(heads: int) -> int:
    """Heads per CTA: 16 for up to 16 heads, otherwise 32 (larger head counts use several CTAs)."""
    return 16 if heads <= 16 else 32


def attention_head_blocks(heads: int) -> int:
    block = attention_block_h(heads)
    return (heads + block - 1) // block


@dataclass(frozen=True)
class AttentionGeometry:
    heads: int
    d_v: int
    d_tail: int
    block_h: int
    head_blocks: int
    topk: int
    pair: bool  # backward: the two head blocks of a row run as a 2-CTA cluster and write one shared partial
    tc_bwd: bool = (
        False  # backward: one tcgen05 CTA per row holds all of the row's heads (8, 16, 32 or 64) and writes one partial per slot
    )

    @property
    def d_qk(self) -> int:
        return self.d_v + self.d_tail

    @property
    def key_block(self) -> int:
        return 32 if self.block_h == 32 else 64

    @property
    def kv_partials(self) -> int:
        """FP32 key-side partial rows the backward writes per (query row, slot): one per head block, or one per row
        (pair cluster or tcgen05 64-head CTA)."""
        return 1 if (self.pair or self.tc_bwd) else self.head_blocks


def _attention_pair_supported(device) -> bool:
    """The 2-CTA-cluster backward rows kernel needs thread-block clusters (sm_90 and later)."""
    return tuple(torch.cuda.get_device_capability(device)) >= (9, 0)


def attention_geometry(q, kv, indices, d_v: int) -> AttentionGeometry:
    num_rows, heads, d_qk = q.shape
    if kv.dim() != 2 or kv.shape[1] != d_qk:
        raise ValueError(f"kv must be [num_kv, {d_qk}], got {tuple(kv.shape)}")
    if (d_v, d_qk - d_v) not in ATTENTION_SHAPES:
        raise ValueError(f"unsupported (d_v, d_tail) = {(d_v, d_qk - d_v)}; supported: {ATTENTION_SHAPES}")
    if indices.dim() != 2 or indices.shape[0] != num_rows:
        raise ValueError(f"indices must be [{num_rows}, topk], got {tuple(indices.shape)}")
    block_h = attention_block_h(heads)
    topk = int(indices.shape[1])
    if topk % 64 != 0:
        raise ValueError(f"topk must be a multiple of 64 (pad with -1), got {topk}")
    head_blocks = attention_head_blocks(heads)
    tc_bwd = attention_tc_bwd_stage(heads, d_qk - d_v, device_arch(q.device)) is not None
    pair = head_blocks == 2 and not tc_bwd and _attention_pair_supported(q.device)
    return AttentionGeometry(heads, d_v, d_qk - d_v, block_h, head_blocks, topk, pair, tc_bwd)


def attention_tc_stage(heads: int, d_tail: int, arch: str) -> str | None:
    """The tcgen05 forward stage for ``heads`` on ``arch``, or ``None`` when the register-MMA forward applies."""
    if heads not in ATTENTION_TC_HEADS or arch not in ATTENTION_TC_ARCHES:
        return None
    if os.environ.get("MILES_DSA_ATTENTION_TC", "1") == "0":
        return None
    return f"dsa_attention_fwd_tc_h{heads}_d512t{d_tail}"


def attention_tc_bwd_stage(heads: int, d_tail: int, arch: str) -> str | None:
    """The tcgen05 backward rows stage for ``heads`` on ``arch``, or ``None`` when the register-MMA rows kernels apply."""
    if heads not in ATTENTION_TC_BWD_HEADS or arch not in ATTENTION_TC_ARCHES:
        return None
    if os.environ.get("MILES_DSA_ATTENTION_BWD_TC", "1") == "0":
        return None
    return f"dsa_attention_bwd_tc_h{heads}_d512t{d_tail}"


def indexer_tc_stage(heads: int, arch: str) -> str | None:
    """The tcgen05 indexer logits stage for ``heads`` on ``arch``, or ``None`` when the register-MMA logits stage applies."""
    if heads not in INDEXER_TC_HEADS or arch not in ATTENTION_TC_ARCHES:
        return None
    if os.environ.get("MILES_DSA_INDEXER_TC", "1") == "0":
        return None
    return f"dsa_indexer_logits_tc_h{heads}"


def indexer_tc_bwd_stage(heads: int, topk: int, arch: str) -> str | None:
    """The tcgen05 indexer backward rows stage for ``heads`` rows of (padded) ``topk`` slots on ``arch``, or ``None`` when
    the register-MMA stage applies (other head counts / architectures, the kill switch, or ``topk`` below
    ``INDEXER_TC_BWD_MIN_TOPK[heads]``)."""
    if heads not in INDEXER_TC_BWD_HEADS or arch not in ATTENTION_TC_ARCHES:
        return None
    if os.environ.get("MILES_DSA_INDEXER_BWD_TC", "1") == "0":
        return None
    if int(topk) < INDEXER_TC_BWD_MIN_TOPK[heads]:
        return None
    return f"dsa_indexer_bwd_tc_h{heads}"


def _storage_alias(tensor):
    """``(flat contiguous alias of the storage, element offset)`` for a raw pointer argument.

    The generated launchers check raw pointer arguments for contiguity and the tcgen05 kernel derives the alignment
    of its vector loads from ``(row stride | element offset)`` on the assumption that the pointer it receives is the
    (aligned) storage base; a contiguous view that starts inside its storage therefore travels as the whole storage
    viewed flat plus its element offset.
    """
    if tensor.is_contiguous() and tensor.storage_offset() == 0:
        return tensor, 0
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    flat.set_(tensor.untyped_storage())
    return flat, int(tensor.storage_offset())


def _launch_attention_forward_tc(stage: str, q, kv, indices, sink, out, lse, sm_scale: float, geo: AttentionGeometry):
    """Launch the tcgen05 forward: one CTA per query row (``grid = (num_rows, 1, 1)``), the ``dsa_h64_fwd`` ABI.

    ``q_latent`` / ``q_rope`` / ``kv_latent`` / ``out`` are TMA tensor maps the launcher encodes from the bound views
    (``q[:, :, :512]``, the rope tail ``q[:, :, 512:]``, ``kv[:, :512]``; the no-rope program never loads its
    ``q_rope`` descriptor and takes a placeholder view); ``k_rope`` is the packed ``kv`` storage with the tail's
    element offset and the row stride ``d_qk``; ``o_lo`` is never written (placeholder = ``out``); ``topk_length`` is
    unused (``has_topk_length = derive_length = 0``: the kernel reads ``indices`` to the end of the row, ``-1``
    anywhere).
    """
    num_rows, num_kv = int(q.shape[0]), int(kv.shape[0])
    d_latent, d_qk = ATTENTION_D_LATENT, geo.d_qk
    q_latent = q[:, :, :d_latent]
    q_rope = q[:, :, d_latent:] if geo.d_tail else q[:, :, :ATTENTION_D_ROPE]
    kv_latent = kv[:, :d_latent]
    k_rope, k_rope_offset = _storage_alias(kv)
    k_rope_offset += d_latent if geo.d_tail else 0
    indices_ptr, indices_offset = _storage_alias(indices)
    _kernel(stage, q.device).launch(
        grid=(num_rows, 1, 1),
        q_latent=q_latent,
        q_rope=q_rope,
        kv_latent=kv_latent,
        k_rope=k_rope,
        out=out,
        o_lo=out,
        lse=lse,
        indices=indices_ptr,
        topk_length=indices_ptr,
        sink=sink,
        num_queries=num_rows,
        num_kv=num_kv,
        topk=geo.topk,
        idx_stride=int(indices.stride(0)),
        indices_offset=indices_offset,
        k_rope_stride=d_qk,
        k_rope_offset=k_rope_offset,
        has_topk_length=0,
        derive_length=0,
        scale_log2=float(sm_scale) * LOG2E,
        heads=geo.heads,
    )


def dsa_attention_forward(q, kv, indices, sink, sm_scale: float, *, d_v: int = 512, out=None):
    """Flat-contract forward: ``(out[num_rows, heads, d_v] BF16, lse[num_rows, heads] FP32 base-2)``.

    ``out`` may be a caller-owned contiguous ``[num_rows, heads, d_v]`` buffer (for example a flat view of a
    layout-shaped tensor); it is allocated here otherwise.

    Head counts in ``ATTENTION_TC_HEADS`` run the tcgen05 forward stage (one CTA per row; FP32 accumulation in the
    tensor-core order, bit-deterministic run to run, within the register-MMA stages' tolerances but not bit-identical
    to them); every other head count runs the register-MMA stage selected by ``attention_block_h``.  Both write the
    same base-2 ``lse`` the backward consumes.
    """
    geo = attention_geometry(q, kv, indices, d_v)
    num_rows = q.shape[0]
    if out is None:
        out = torch.empty(num_rows, geo.heads, geo.d_v, dtype=q.dtype, device=q.device)
    elif out.shape != (num_rows, geo.heads, geo.d_v) or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{num_rows}, {geo.heads}, {geo.d_v}] {q.dtype} tensor")
    lse = torch.empty(num_rows, geo.heads, dtype=torch.float32, device=q.device)
    if num_rows == 0:
        return out, lse
    tc_stage = attention_tc_stage(geo.heads, geo.d_tail, device_arch(q.device))
    if tc_stage is not None:
        _launch_attention_forward_tc(tc_stage, q, kv, indices, sink, out, lse, sm_scale, geo)
        return out, lse
    _kernel(_attention_name("fwd", geo.block_h, geo.d_v, geo.d_tail), q.device).launch(
        grid=(num_rows, geo.head_blocks, 1),
        q=q,
        kv=kv,
        sink=sink,
        indices=indices,
        out=out,
        lse=lse,
        heads=geo.heads,
        topk=geo.topk,
        scale_log2=float(sm_scale) * LOG2E,
    )
    return out, lse


ATTENTION_D_V = 512  # the value width of every ATTENTION_SHAPES entry; ``dsa_attention_delta`` is traced for it


def dsa_attention_delta_launch(o, do):
    """``delta[row, h] = sum_d o * do`` in FP32 for flat ``[num_rows, heads, 512]`` BF16 ``o`` and ``do`` (fixed order)."""
    num_rows, heads, d_v = o.shape
    if d_v != ATTENTION_D_V or do.shape != o.shape or o.dtype != torch.bfloat16 or do.dtype != torch.bfloat16:
        raise ValueError(
            f"delta expects BF16 o and do of shape [rows, heads, {ATTENTION_D_V}], got {tuple(o.shape)} / {tuple(do.shape)}"
        )
    if not (o.is_contiguous() and do.is_contiguous()):
        raise ValueError("delta expects contiguous o and do")
    delta = torch.empty(num_rows, heads, dtype=torch.float32, device=o.device)
    if num_rows == 0:
        return delta
    _kernel("dsa_attention_delta", o.device).launch(grid=(num_rows, 1, 1), o=o, do=do, delta=delta, heads=heads)
    return delta


def dsa_attention_backward_rows(q, kv, do, indices, lse, delta, dq, partial, sm_scale: float, geo: AttentionGeometry):
    """Launch the backward kernel on a contiguous slice of query rows (fills ``dq`` rows and ``partial``).

    ``partial`` holds ``geo.kv_partials`` FP32 rows per (query row, slot): with ``geo.pair`` the two head-block CTAs of
    a row form one cluster (grid ``2 * num_rows``) and write a single summed partial per slot.
    """
    num_rows = q.shape[0]
    if num_rows == 0:
        return
    if geo.tc_bwd:
        _launch_attention_backward_tc(q, kv, do, indices, lse, delta, dq, partial, sm_scale, geo)
        return
    if geo.pair:
        name, grid = _attention_pair_name(geo.d_v, geo.d_tail), (2 * num_rows, 1, 1)
    else:
        name, grid = _attention_name("bwd", geo.block_h, geo.d_v, geo.d_tail), (num_rows, geo.head_blocks, 1)
    _kernel(name, q.device).launch(
        grid=grid,
        q=q,
        kv=kv,
        do=do,
        indices=indices,
        lse=lse,
        delta=delta,
        dq=dq,
        partial=partial,
        heads=geo.heads,
        topk=geo.topk,
        num_head_blocks=geo.head_blocks,
        sm_scale=float(sm_scale),
        scale_log2=float(sm_scale) * LOG2E,
    )


def _launch_attention_backward_tc(
    q, kv, do, indices, lse, delta, dq, partial, sm_scale: float, geo: AttentionGeometry
):
    """Launch the tcgen05 backward rows kernel: one CTA per query row (``grid = (num_rows, 1, 1)``), the ``dsa_h64_bwd`` ABI.

    ``q_latent`` / ``q_rope`` / ``dout`` / ``dq_latent`` / ``dq_rope`` / ``kv_latent`` are TMA tensor maps encoded from
    the bound views (``q[:, :, :512]``, the rope tails ``q[:, :, 512:]`` / ``kv[:, 512:]``, ``kv[:, :512]``; the no-rope
    program never touches its rope descriptors and takes placeholder views); the partial rows ``partial[(row * topk + slot), 0:d_qk]`` are addressed through ``dkv_f32`` / ``dkr_f32``
    (row stride ``d_qk``, rope column 512); ``lse`` is the forward's base-2 lse.  ``out`` / ``o_lo`` / ``topk_length`` /
    ``dkv_dst_map`` / the key-pass buffers are inert single-pass bindings (never read).
    """
    num_rows, num_kv = int(q.shape[0]), int(kv.shape[0])
    d_latent, d_qk = ATTENTION_D_LATENT, geo.d_qk
    q_latent, dq_latent, kv_latent = q[:, :, :d_latent], dq[:, :, :d_latent], kv[:, :d_latent]
    if geo.d_tail:
        q_rope, dq_rope, k_rope = q[:, :, d_latent:], dq[:, :, d_latent:], kv[:, d_latent:]
    else:
        q_rope, dq_rope, k_rope = q[:, :, :ATTENTION_D_ROPE], dq[:, :, :ATTENTION_D_ROPE], kv
    indices_ptr, indices_offset = _storage_alias(indices)
    _kernel(attention_tc_bwd_stage(geo.heads, geo.d_tail, device_arch(q.device)), q.device).launch(
        grid=(num_rows, 1, 1),
        q_latent=q_latent,
        q_rope=q_rope,
        dout=do,
        dq_latent=dq_latent,
        dq_rope=dq_rope,
        kv_latent=kv_latent,
        k_rope=k_rope,
        lse=lse,
        delta=delta,
        out=do,
        o_lo=do,
        indices=indices_ptr,
        topk_length=indices_ptr,
        dkv_f32=partial,
        dkr_f32=partial,
        dkv_stride=d_qk,
        dkr_stride=d_qk,
        dkr_col0=d_latent,
        dkv_dst_map=indices_ptr,
        dkv_has_map=0,
        num_queries=num_rows,
        num_kv=num_kv,
        topk=geo.topk,
        idx_stride=int(indices.stride(0)),
        indices_offset=indices_offset,
        has_topk_length=0,
        token_base=0,
        token_step=1,
        scale_log2=float(sm_scale) * LOG2E,
        sm_scale=float(sm_scale),
        pass_lo=0,
        pass_hi=num_kv,
        dq_mode=0,
        dq_partial=delta,
        key_scratch=indices_ptr,
        pass_counts=indices_ptr,
    )


# SMEM combine buffer of the split reducer: 4 warps x (dim / 4) float4 columns.
REDUCE_SPLIT_MAX_DIM = 576


def dsa_segmented_reduce_launch(partial, order, seg_start, seg_count, out, *, split: int = 1) -> None:
    """``out[key, :] += sum partial[order[p], :]`` over each key's sorted segment.

    ``split=1``: ``dsa_segmented_reduce`` adds the segment pair by pair in ``order``
    sequence (``keys_per_cta`` keys share a CTA).  ``split=4``: ``dsa_segmented_reduce_split4``
    runs one CTA per key whose four warps each add a contiguous quarter of the segment in
    ``order`` sequence; the quarter sums are combined in warp order.  Both depend only on
    ``order`` / ``seg_count`` (deterministic FP32); their rounding differs.
    """
    num_out = int(out.shape[0])
    dim = int(partial.shape[1])
    if num_out == 0:
        return
    if split == 4:
        if dim > REDUCE_SPLIT_MAX_DIM:
            raise ValueError(f"split reducer supports dim <= {REDUCE_SPLIT_MAX_DIM}, got {dim}")
        _kernel("dsa_segmented_reduce_split4", out.device).launch(
            grid=(num_out, 1, 1),
            partial=partial,
            order=order,
            seg_start=seg_start,
            seg_count=seg_count,
            out=out,
            dim=dim,
            num_out=num_out,
        )
        return
    if split != 1:
        raise ValueError(f"split must be 1 or 4, got {split}")
    chunks = dim // 4
    keys_per_cta = 1
    for candidate in (4, 2):
        if 128 // candidate >= chunks:
            keys_per_cta = candidate
            break
    _kernel("dsa_segmented_reduce", out.device).launch(
        grid=((num_out + keys_per_cta - 1) // keys_per_cta, 1, 1),
        partial=partial,
        order=order,
        seg_start=seg_start,
        seg_count=seg_count,
        out=out,
        dim=dim,
        keys_per_cta=keys_per_cta,
        num_out=num_out,
    )


# --------------------------------------------------------------------------- #
# lightning indexer
# --------------------------------------------------------------------------- #


def indexer_padded_heads(heads: int) -> int:
    if heads <= 0 or heads > 64:
        raise ValueError(f"indexer heads must be in 1..64, got {heads}")
    return 16 if heads <= 16 else (32 if heads <= 32 else 64)


def indexer_key_block(heads: int) -> int:
    return 64 if indexer_padded_heads(heads) == 16 else 32


def dsa_indexer_logits(index_q, index_k, weights, cu_ks, cu_ke, *, batch: int, seq_len: int, seq_len_kv: int):
    """Batched indexer logits ``[batch, seq_len, seq_len_kv]`` (FP32; ``-inf`` outside each query's key range).

    ``index_q`` is ``[batch * seq_len, heads, 128]`` BF16, ``index_k`` ``[batch * seq_len_kv, 128]``
    BF16, ``weights`` ``[batch * seq_len, heads]`` FP32, ``cu_ks``/``cu_ke`` int32 ``[seq_len]`` key
    bounds shared by the batch.

    Head counts in ``INDEXER_TC_HEADS`` run the tcgen05 logits stage on ``ATTENTION_TC_ARCHES`` (FP32 in the tensor-core
    order, bit-deterministic run to run, within the register-MMA stage's tolerances but not bit-identical to it); every
    other case runs the register-MMA stage.  ``dsa_indexer_clean`` follows either.
    """
    heads, dim = int(index_q.shape[1]), int(index_q.shape[2])
    if dim != INDEXER_HEAD_DIM:
        raise ValueError(f"indexer head dim must be {INDEXER_HEAD_DIM}, got {dim}")
    if heads not in INDEXER_HEADS:
        raise ValueError(f"indexer heads must be one of {INDEXER_HEADS}, got {heads}")
    logits = torch.empty(batch, seq_len, seq_len_kv, dtype=torch.float32, device=index_q.device)
    if batch == 0 or seq_len == 0:
        return logits
    bq = 64 // heads
    if seq_len_kv > 0:
        tc_stage = indexer_tc_stage(heads, device_arch(index_q.device))
        if tc_stage is not None:
            # the tcgen05 stage: ``index_q`` / ``index_k`` are TMA tensor maps the launcher encodes from the bound 2-D views
            pq = INDEXER_TC_QUERY_COLUMNS // heads
            _kernel(tc_stage, index_q.device).launch(
                grid=((seq_len + pq - 1) // pq, batch, 1),
                index_q=index_q.reshape(batch * seq_len * heads, dim),
                index_k=index_k,
                weights=weights,
                cu_ks=cu_ks,
                cu_ke=cu_ke,
                logits=logits,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
            )
        else:
            _kernel(f"dsa_indexer_logits_h{heads}", index_q.device).launch(
                grid=((seq_len + bq - 1) // bq, batch, 1),
                index_q=index_q,
                index_k=index_k,
                weights=weights,
                cu_ks=cu_ks,
                cu_ke=cu_ke,
                logits=logits,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
            )
        _kernel("dsa_indexer_clean", index_q.device).launch(
            grid=(seq_len, batch, 1),
            logits=logits,
            cu_ks=cu_ks,
            cu_ke=cu_ke,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
        )
    return logits


def dsa_indexer_backward_rows(index_q, index_k, weights, topk_indices, grad_scores, d_index_q, d_weights, partial):
    """Launch the indexer backward on a contiguous slice of query rows.

    ``partial`` receives one FP32 ``[128]`` key-side partial per ``(row, slot)`` at
    ``row * topk + slot``; rows of padded slots (``-1``) are not written.
    """
    num_rows, heads, dim = index_q.shape
    if num_rows == 0:
        return
    if dim != INDEXER_HEAD_DIM:
        raise ValueError(f"indexer head dim must be {INDEXER_HEAD_DIM}, got {dim}")
    topk = int(topk_indices.shape[1])
    hp = indexer_padded_heads(int(heads))
    if topk % indexer_key_block(int(heads)) != 0:
        raise ValueError(
            f"indexer topk must be a multiple of {indexer_key_block(int(heads))} for {heads} heads, got {topk}"
        )
    stage = indexer_tc_bwd_stage(int(heads), topk, device_arch(index_q.device)) or f"dsa_indexer_bwd_h{hp}"
    _kernel(stage, index_q.device).launch(
        grid=(num_rows, 1, 1),
        index_q=index_q,
        index_k=index_k,
        weights=weights,
        topk_indices=topk_indices,
        grad_scores=grad_scores,
        d_index_q=d_index_q,
        d_weights=d_weights,
        partial=partial,
        heads=int(heads),
        topk=topk,
    )


# --------------------------------------------------------------------------- #
# indexer top-k
# --------------------------------------------------------------------------- #

# One exported variant per shared-memory row capacity (``num_keys <= capacity``), all traced for
# ``K = TOPK_K`` slots: ``dsa_topk_fp32_c{capacity}_k512``.  ``TOPK_CAPACITIES[-1]`` is the kernel's
# maximum row length; 4096 is the DeepSeek-V4.1 compressed row length.
TOPK_CAPACITIES = (4096, 8192, 16384, 32768)
TOPK_K = 512

_TOPK_NO_ROW_END: dict = {}


def _topk_name(capacity: int) -> str:
    return f"{TOPK_STAGE_PREFIX}c{capacity}_k{TOPK_K}"


def topk_capacity(num_keys: int) -> int:
    """The exported top-k variant for a row length: the smallest ``TOPK_CAPACITIES`` entry with ``num_keys <= capacity``.

    A variant stages at most ``capacity`` positions per row and has no device-side ``num_keys`` check,
    so the host must select it here.
    """
    num_keys = int(num_keys)
    if num_keys < 1 or num_keys > TOPK_CAPACITIES[-1]:
        raise ValueError(f"top-k row length must be in 1..{TOPK_CAPACITIES[-1]}, got {num_keys}")
    return next(capacity for capacity in TOPK_CAPACITIES if num_keys <= capacity)


def _topk_no_row_end(device):
    """The ignored ``row_end`` operand of a launch with ``has_row_end == 0`` (one int32 per device)."""
    tensor = _TOPK_NO_ROW_END.get(device)
    if tensor is None:
        tensor = _TOPK_NO_ROW_END[device] = torch.zeros(1, dtype=torch.int32, device=device)
    return tensor


def dsa_topk(logits, topk: int):
    """Deterministic FP32 row top-k: int32 ``[rows, topk]`` key indices of ``logits[rows, num_keys]``.

    Slot ``j`` of a row holds the key with the ``j``-th highest score (ties: the smaller index first);
    ``-inf`` and NaN positions are never selected and every unfilled slot is ``-1`` -- a tail after the
    valid picks, also when a row has fewer than ``topk`` finite scores (the kernel writes it; nothing is
    post-processed here).  Identical inputs give bit-identical outputs.  ``num_keys`` must be at most
    ``TOPK_CAPACITIES[-1]`` and ``topk`` at most ``TOPK_K``: the exported variants are traced for
    ``K = 512`` and a smaller ``topk`` keeps the first ``topk`` slots.  When ``topk > num_keys`` the
    row is padded with ``-inf`` columns up to ``topk`` and the kernel pads the missing slots with ``-1``.
    """
    if logits.dim() != 2 or logits.dtype != torch.float32:
        raise TypeError(f"top-k logits must be a 2-D float32 tensor, got {tuple(logits.shape)} {logits.dtype}")
    rows, num_keys = (int(v) for v in logits.shape)
    topk = int(topk)
    if topk < 1 or topk > TOPK_K:
        raise ValueError(f"the cake top-k backend supports 1 <= topk <= {TOPK_K}, got topk={topk}")
    if num_keys > TOPK_CAPACITIES[-1]:
        raise ValueError(
            f"the cake top-k backend supports at most {TOPK_CAPACITIES[-1]} keys per row, got num_keys={num_keys}"
        )
    if num_keys < topk:
        logits = torch.nn.functional.pad(logits, (0, topk - num_keys), value=float("-inf"))
        num_keys = topk
    if not logits.is_cuda:
        raise ValueError("the cake top-k backend needs CUDA logits")
    name = _topk_name(topk_capacity(num_keys))
    if name not in MODULES:
        raise NotImplementedError(f"{name}: the generated top-k kernels are not in this checkout's registry.json")
    out = torch.empty(rows, TOPK_K, dtype=torch.int32, device=logits.device)
    if rows == 0:
        return out[:, :topk]
    _kernel(name, logits.device).launch(
        grid=(rows, 1, 1),
        logits=logits.contiguous(),
        out=out,
        row_end=_topk_no_row_end(logits.device),
        n_kv=num_keys,
        has_row_end=0,
    )
    return out if topk == TOPK_K else out[:, :topk].contiguous()


# --------------------------------------------------------------------------- #
# fused indexer scoring + top-k
# --------------------------------------------------------------------------- #

# Two stages traced once for 32 indexer heads of dim 128 and at most 512 slots: the persistent scan (``tcgen05``
# + TMEM + TMA) scores the keys of every packed segment in FP32 and keeps each query's exact top-k on chip, the
# finalize rewrites each row into ascending key order with the ``-1`` / ``-inf`` tail.  ``topk``, ``ratio``,
# ``softmax_scale`` and the segment table are launch arguments, so the pair serves every call within the bound.
INDEXER_TOPK_HEADS = 32
INDEXER_TOPK_K_MAX = 512
INDEXER_TOPK_SCAN_STAGE = f"{INDEXER_TOPK_STAGE_PREFIX}scan_h{INDEXER_TOPK_HEADS}_d{INDEXER_HEAD_DIM}"
INDEXER_TOPK_FINALIZE_STAGE = f"{INDEXER_TOPK_STAGE_PREFIX}finalize_k{INDEXER_TOPK_K_MAX}"
# Generator constants of the exported scan.  The launch scalars derived from them size the candidate workspace and
# pace the selection trigger; the outputs are bitwise independent of their values.
_SCAN_BLOCK_Q = 4  # queries per scan unit
_SCAN_TILE_KEYS = 128  # keys per MMA tile
_SCAN_CAND_MULT = 4  # default candidate slots per top-k slot
_SCAN_CAND_FLOOR = 8192  # candidate-capacity floor (slots)
_SCAN_CHECK_PERIOD_MAX = 32
_SCAN_CHECK_PERIOD_CAP_DIVISOR = 4 * _SCAN_TILE_KEYS
_SCAN_SAMPLE_SHIFT_PERMILLE = 250
_SCAN_SAMPLE_TILES = ((128, 8), (640, 16))  # (estimated mean unit tiles <=, sample_tiles_max); longer units: 32
_SCAN_SAMPLE_TILES_LONG = 32

_INDEXER_TOPK_NO_OFFSETS: dict = {}


def indexer_topk_cand_cap(topk: int) -> int:
    """Candidate slots per (warpgroup, query) of the scan: ``max(topk * mult, topk + 128)`` rounded up to a multiple
    of 128, ``mult = max(4, ceil(8192 / topk))`` -- 8192 for every top-k that divides 8192 (64, 128, 256, 512)."""
    topk = int(topk)
    if topk < 1:
        raise ValueError(f"topk must be positive, got {topk}")
    mult = max(_SCAN_CAND_MULT, -(-_SCAN_CAND_FLOOR // topk))
    cap = max(topk * mult, topk + _SCAN_TILE_KEYS)
    return (cap + _SCAN_TILE_KEYS - 1) // _SCAN_TILE_KEYS * _SCAN_TILE_KEYS


def indexer_topk_check_period(topk: int, cand_cap: int) -> int:
    """Key tiles between two selection-trigger checks: ``max(1, min(32, (cand_cap - topk) // 128, cand_cap // 512))``
    (16 at ``topk = 512``, ``cand_cap = 8192``)."""
    topk, cap = int(topk), int(cand_cap)
    return max(1, min(_SCAN_CHECK_PERIOD_MAX, (cap - topk) // _SCAN_TILE_KEYS, cap // _SCAN_CHECK_PERIOD_CAP_DIVISOR))


def indexer_topk_sample_tiles(total_q: int, total_k: int, num_segments: int, ratio: int) -> int:
    """Cap on the key tiles sampled for a unit's first threshold, by the call's estimated mean unit tiles
    ``(Tkv - T / (2 ratio)) / S / 128``: 8 up to 128 tiles, 16 up to 640, 32 beyond."""
    segments = max(1, int(num_segments))
    r = max(1, int(ratio))
    mean_tiles = max(0.0, (int(total_k) - int(total_q) / (2 * r)) / segments) / _SCAN_TILE_KEYS
    for bound, tiles in _SCAN_SAMPLE_TILES:
        if mean_tiles <= bound:
            return tiles
    return _SCAN_SAMPLE_TILES_LONG


def indexer_topk_key_bits(num_kv: int) -> int:
    """Key bits the finalize sorts: ``max(1, (num_kv - 1).bit_length())`` (every valid key id is below ``num_kv``)."""
    return max(1, (int(num_kv) - 1).bit_length())


def indexer_topk_workspace_elems(grid: int, cand_cap: int) -> int:
    """int64 entries of the scan workspace: ``cand_cap`` packed candidates per (CTA, query of the 4-query unit)."""
    return int(grid) * _SCAN_BLOCK_Q * int(cand_cap)


def _indexer_topk_no_offsets(device):
    """The ignored ``q_offsets`` operand of a launch with ``has_offsets == 0`` (one int64 per device, never read)."""
    tensor = _INDEXER_TOPK_NO_OFFSETS.get(device)
    if tensor is None:
        tensor = _INDEXER_TOPK_NO_OFFSETS[device] = torch.zeros(1, dtype=torch.int64, device=device)
    return tensor


def dsa_indexer_topk(
    index_q, index_k, weights, cu_seqlens_q, cu_seqlens_k, *, topk: int, softmax_scale: float, ratio: int = 1
):
    """Fused indexer scoring + exact top-k over packed segments: ``(indices int32 [T, topk], scores fp32 [T, topk])``.

    ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * q[t, h, :] . k[j, :])`` in FP32 (no fast-math; a row whose
    logits are all non-positive scores ``+0.0``).  Key ``j`` of segment ``s`` (``Lq`` queries, ``Lk`` keys; key ids
    are local to the segment) is visible to its query ``u`` iff ``j < min(Lk, floor((offset + u + 1) / ratio))``,
    ``offset = Lk - Lq`` at ``ratio == 1`` and ``0`` otherwise.  Each row keeps the exact top ``min(topk, visible)``
    keys by score descending, then key id ascending, emitted in ascending key-id order; the tail is ``-1`` /
    ``-inf``.  No floating-point atomics: identical inputs give bit-identical outputs, whatever the launch scalars.

    ``index_q`` BF16 contiguous ``[T, 32, 128]``, ``index_k`` BF16 ``[Tkv, 128]`` (unit inner stride, row stride a
    multiple of 8 elements), ``weights`` FP32 contiguous ``[T, 32]``, ``cu_seqlens_q`` / ``cu_seqlens_k`` int32
    contiguous ``[S + 1]`` on the device (``cu_seqlens_q[-1] == T``: a row outside every segment is not written),
    ``1 <= topk <= 512``, ``softmax_scale > 0``, integer ``ratio >= 1``.  Two launches on the current stream: the
    persistent scan (one CTA per SM) and the per-row finalize.  ``T == 0`` returns empty outputs without a launch.
    Every limit raises ``ValueError`` before any launch; there is no fallback to another path.
    """
    heads, dim = INDEXER_TOPK_HEADS, INDEXER_HEAD_DIM
    if index_q.dim() != 3 or tuple(index_q.shape[1:]) != (heads, dim) or index_q.dtype != torch.bfloat16:
        raise ValueError(f"index_q must be bf16 [T, {heads}, {dim}], got {tuple(index_q.shape)} {index_q.dtype}")
    if not index_q.is_contiguous():
        raise ValueError("index_q must be contiguous")
    if index_k.dim() != 2 or index_k.shape[1] != dim or index_k.dtype != torch.bfloat16:
        raise ValueError(f"index_k must be bf16 [Tkv, {dim}], got {tuple(index_k.shape)} {index_k.dtype}")
    if index_k.stride(1) != 1 or index_k.stride(0) < dim or index_k.stride(0) % 8:
        raise ValueError(
            f"index_k needs a unit inner stride and a row stride >= {dim} that is a multiple of 8, "
            f"got {index_k.stride()}"
        )
    if index_q.data_ptr() % 16 or index_k.data_ptr() % 16:
        raise ValueError("index_q and index_k must be 16-byte aligned (TMA sources)")
    num_rows, num_kv = int(index_q.shape[0]), int(index_k.shape[0])
    if tuple(weights.shape) != (num_rows, heads) or weights.dtype != torch.float32 or not weights.is_contiguous():
        raise ValueError(f"weights must be contiguous fp32 [T, {heads}], got {tuple(weights.shape)} {weights.dtype}")
    for name, t in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if t.dim() != 1 or t.dtype != torch.int32 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32 [S + 1], got {tuple(t.shape)} {t.dtype}")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must both be [S + 1]")
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= INDEXER_TOPK_K_MAX:
        raise ValueError(f"the fused indexer top-k supports 1 <= topk <= {INDEXER_TOPK_K_MAX}, got topk={topk!r}")
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
        raise ValueError(f"ratio must be a positive integer, got {ratio!r}")
    if not float(softmax_scale) > 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale!r}")
    device = index_q.device
    if device.type != "cuda":
        raise ValueError("the fused indexer top-k needs CUDA tensors")
    for name, t in (
        ("index_k", index_k),
        ("weights", weights),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if t.device != device:
            raise ValueError(f"{name} must be on {device}, got {t.device}")
    num_segments = int(cu_seqlens_q.numel()) - 1
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)
    scores = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    if num_rows == 0:
        return indices, scores
    if num_kv < 1:
        raise ValueError("index_k must hold at least one key row (TMA source extent)")
    if num_segments < 1:
        raise ValueError("at least one segment is needed for a non-empty query set (cu_seqlens_q has one entry)")
    grid = int(torch.cuda.get_device_properties(device).multi_processor_count)
    cap = indexer_topk_cand_cap(topk)
    workspace = torch.empty(indexer_topk_workspace_elems(grid, cap), dtype=torch.int64, device=device)
    _kernel(INDEXER_TOPK_SCAN_STAGE, device).launch(
        grid=(grid, 1, 1),
        Q=index_q.view(num_rows * heads, dim),
        K=index_k,
        W=weights,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_offsets=_indexer_topk_no_offsets(device),
        Indices=indices,
        Scores=scores,
        Cand=workspace,
        num_segments=num_segments,
        top_k=topk,
        ratio=ratio,
        has_offsets=0,
        cand_cap=cap,
        first_cap=cap,
        sample_tiles_max=indexer_topk_sample_tiles(num_rows, num_kv, num_segments, ratio),
        sample_shift_permille=_SCAN_SAMPLE_SHIFT_PERMILLE,
        check_period=indexer_topk_check_period(topk, cap),
        grid_ctas=grid,
        softmax_scale=float(softmax_scale),
        n_split=1,
    )
    _kernel(INDEXER_TOPK_FINALIZE_STAGE, device).launch(
        grid=(num_rows, 1, 1),
        Indices=indices,
        Scores=scores,
        top_k=topk,
        key_bits=indexer_topk_key_bits(num_kv),
    )
    return indices, scores


__all__ = [
    "ATTENTION_D_LATENT",
    "ATTENTION_D_ROPE",
    "ATTENTION_SHAPES",
    "ATTENTION_TC_ARCHES",
    "ATTENTION_TC_BWD_HEADS",
    "ATTENTION_TC_HEADS",
    "INDEXER_HEADS",
    "INDEXER_HEAD_DIM",
    "INDEXER_TOPK_FINALIZE_STAGE",
    "INDEXER_TOPK_HEADS",
    "INDEXER_TOPK_K_MAX",
    "INDEXER_TC_HEADS",
    "INDEXER_TC_QUERY_COLUMNS",
    "INDEXER_TC_BWD_HEADS",
    "INDEXER_TC_BWD_MIN_TOPK",
    "REDUCE_SPLIT_MAX_DIM",
    "INDEXER_TOPK_SCAN_STAGE",
    "MODULES",
    "TOPK_CAPACITIES",
    "TOPK_K",
    "AttentionGeometry",
    "attention_block_h",
    "attention_geometry",
    "attention_head_blocks",
    "attention_tc_bwd_stage",
    "attention_tc_stage",
    "dsa_attention_backward_rows",
    "dsa_attention_delta_launch",
    "dsa_attention_forward",
    "dsa_indexer_backward_rows",
    "dsa_indexer_logits",
    "dsa_indexer_topk",
    "dsa_segmented_reduce_launch",
    "dsa_topk",
    "indexer_key_block",
    "indexer_tc_stage",
    "indexer_padded_heads",
    "indexer_topk_cand_cap",
    "indexer_topk_check_period",
    "indexer_topk_key_bits",
    "indexer_topk_sample_tiles",
    "indexer_topk_workspace_elems",
    "topk_capacity",
    "indexer_tc_bwd_stage",
]
