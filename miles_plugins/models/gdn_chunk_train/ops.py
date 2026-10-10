"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic ``chunk_gated_delta_rule`` (forward + backward) on the generated kernels (SM100a / SM103a).

Drop-in replacement for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` (FLA v0.5.2) as the
Qwen GDN wrappers call it: ``[B, T, H, K]`` inputs, packed varlen through ``cu_seqlens``, in-kernel
q/k L2 normalisation, optional FP32 initial/final state, and context-parallel state passing through
an ``FLACPContext``.  Both passes are bit-deterministic call to call: no atomics, fixed reduction
order, and a launch geometry that is a pure function of the shape and the device.

The kernels mirror FLA's chunk-64 decomposition one to one (prep = L2 norm + gate cumsum, WY solve,
state scan, output; backward: dv_local, adjoint state scan, dq/dk/dw/dg, WY pullback, finalize) and
recompute the transient WY/state tensors exactly in the backward, as FLA does.  ``K = V = 128``,
``HV % H == 0`` (grouped value heads without materialised ``repeat_interleave``), BF16 q/k/v.

Host-side wiring (see ``design_doc/active/MILES_GDN_CHUNKED_BACKWARD.md`` in the kernel repository, section
"Host wiring"): the chunk table is a pure function of the micro-batch's segment lengths, so it is built once per
distinct layout from the host copy of ``cu_seqlens`` that Megatron's ``packed_seq_params`` already carries
(``cu_seqlens_cpu``; no device-to-host synchronisation, no launch) and shared by every layer and by the
activation-recompute forward; a device-only ``cu_seqlens`` falls back to the on-device table kernel.  The forward
keeps the WY / state tensors (``w``, ``u``, ``h``, ``v_new``) on the autograd tape by default
(``MILES_GDN_CHUNK_TRAIN_SAVE_STATE=0`` recomputes them in the backward instead; both give bit-identical gradients).

Context parallelism reuses FLA v0.5.2's ``fla.ops.cp`` pre-processing exactly
(``chunk_gated_delta_rule_fwd_h_pre_process``, ``compress_h0``, ``expand_h0``,
``chunk_gated_delta_rule_bwd_dhu_pre_process``), so ``build_gdn_cp_context`` semantics are
unchanged.  Those helpers are only imported when a ``cp_context`` is supplied.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from functools import cache, lru_cache
from collections.abc import Sequence
from typing import Any

import torch

from ._jit import MODULES, device_arch, kernel, prebuild

__all__ = [
    "ChunkGatedDeltaRuleFunction",
    "ChunkMeta",
    "chunk_gated_delta_rule",
    "chunk_meta_from_host",
    "gdn_chunk_backward",
    "gdn_chunk_forward",
    "scan_plans",
    "ScanPlan",
]

CHUNK = 64
HEAD_DIM = 128
# Register-scan configurations (value rows per CTA, K split) in preference order; every warp owns
# 16 value rows and K / ksplit keys.  Must match the exported ``fwd_h_r*``/``dhu_r*`` stages.
REG_SCAN_CONFIGS = ((32, 2), (32, 1), (64, 1))
# Tests may pin a scan configuration; production always derives it from shape and device.
FORCED_SCAN_CONFIG: tuple[int, int] | None = None
# Keep w/u/h/v_new on the autograd tape (default) instead of recomputing them in the backward.
SAVE_STATE = os.environ.get("MILES_GDN_CHUNK_TRAIN_SAVE_STATE", "1") == "1"


# --------------------------------------------------------------------------- #
# chunk table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScanPlan:
    """Launch plan of one state scan: the register-scan configuration ``(value rows per CTA, K split)`` and the chunks
    per segment (``0``: one segment per sequence, the serial scan).  A pure function of the layout and the device
    (:func:`choose_scan_plan`); the segment table built from it carries the configuration, so the launch runs exactly
    the stages the plan was priced for on both table routes.
    """

    config: tuple[int, int]
    segment_len: int


@dataclass(frozen=True)
class SegmentTable:
    """Scan segments: every sequence's chunk range cut into near-equal pieces (all int32 CUDA tensors).

    Segment ``s`` covers chunks ``[seg_chunk_start[s], seg_chunk_start[s + 1])`` of sequence ``seg_seq[s]``;
    the segments of sequence ``i`` are ``[seq_seg_start[i], seq_seg_start[i + 1])`` in chunk order.
    ``num_segments`` is the launch extent: a table built on the device is padded to its upper bound with
    empty segments (``seg_seq == -1``) that the scan kernels skip.  ``segment_len == 0`` means one segment
    per sequence (the serial scan).  ``scan_config`` is the register-scan configuration ``(value rows per CTA,
    K split)`` the table was planned for (:class:`ScanPlan`).
    """

    seg_chunk_start: torch.Tensor
    seg_seq: torch.Tensor
    seq_seg_start: torch.Tensor
    num_segments: int
    segment_len: int
    scan_config: tuple[int, int]


@dataclass(frozen=True)
class ChunkMeta:
    """Chunk table for packed varlen sequences (all int32 CUDA tensors).

    ``num_chunks`` is the launch extent; when the table is built on the device from ``cu_seqlens``
    it is an upper bound (``T // 64 + num_seqs``) and the trailing entries are padding chunks with
    ``chunk_len == 0`` that every chunk kernel skips.  ``fwd_segments`` / ``bwd_segments`` are the
    segment tables of the forward state scan and of the backward state-gradient scan
    (:func:`scan_plans`; ``segment_len == 0`` keeps the serial scan).
    """

    chunk_start: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    num_chunks: int
    num_seqs: int
    seq_lens: tuple[int, ...] | None
    fwd_segments: SegmentTable
    bwd_segments: SegmentTable


def segment_bounds(num_chunks: int, segment_len: int) -> list[int]:
    """Segment boundaries of one sequence (chunk offsets), balanced to within one chunk.

    The device table builder (the ``meta`` stage) evaluates the same integer formula.
    """
    if segment_len <= 0 or num_chunks <= segment_len:
        return [0, num_chunks]
    count = -(-num_chunks // segment_len)
    return [(s * num_chunks) // count for s in range(count + 1)]


def _segment_lists(seq_chunk_start: list[int], segment_len: int) -> tuple[list[int], list[int], list[int]]:
    seg_chunk_start = [0]
    seg_seq: list[int] = []
    seq_seg_start = [0]
    for seq in range(len(seq_chunk_start) - 1):
        c0 = seq_chunk_start[seq]
        bounds = segment_bounds(seq_chunk_start[seq + 1] - c0, segment_len)
        seg_chunk_start.extend(c0 + b for b in bounds[1:])
        seg_seq.extend([seq] * (len(bounds) - 1))
        seq_seg_start.append(len(seg_seq))
    return seg_chunk_start, seg_seq, seq_seg_start


def build_chunk_meta(seq_lens, device, *, scan_plans: tuple[ScanPlan, ScanPlan]) -> ChunkMeta:
    """Exact chunk and segment tables from host sequence lengths (one pinned copy, no synchronisation).

    ``scan_plans`` are the layout's forward and backward scan plans (:func:`scan_plans`).
    """
    starts: list[int] = []
    lens: list[int] = []
    seq_chunk_start = [0]
    token = 0
    for seq_len in seq_lens:
        seq_len = int(seq_len)
        for chunk_start in range(0, seq_len, CHUNK):
            starts.append(token + chunk_start)
            lens.append(min(CHUNK, seq_len - chunk_start))
        seq_chunk_start.append(len(starts))
        token += seq_len
    parts: list[list[int]] = [starts, lens, seq_chunk_start]
    for plan in scan_plans:
        parts.extend(_segment_lists(seq_chunk_start, int(plan.segment_len)))

    # One host tensor, one asynchronous copy (pinned staging): a pageable ``torch.tensor(..., device="cuda")``
    # would synchronise the stream before each copy.
    table = torch.tensor([x for part in parts for x in part], dtype=torch.int32)
    device = torch.device(device)
    table = table.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else table.to(device)
    views = []
    offset = 0
    for part in parts:
        views.append(table[offset : offset + len(part)])
        offset += len(part)
    segments = [
        SegmentTable(
            seg_chunk_start=views[3 + 3 * i],
            seg_seq=views[4 + 3 * i],
            seq_seg_start=views[5 + 3 * i],
            num_segments=len(parts[4 + 3 * i]),
            segment_len=int(scan_plans[i].segment_len),
            scan_config=tuple(scan_plans[i].config),
        )
        for i in range(2)
    ]
    return ChunkMeta(
        chunk_start=views[0],
        chunk_len=views[1],
        seq_chunk_start=views[2],
        num_chunks=len(starts),
        num_seqs=len(seq_lens),
        seq_lens=tuple(int(x) for x in seq_lens),
        fwd_segments=segments[0],
        bwd_segments=segments[1],
    )


_HOST_META_BUILDS = 0  # chunk tables built from host boundaries (tests assert one per micro-batch layout)


def _seq_lens_from_host(cu_seqlens_host) -> tuple[int, ...]:
    """Segment lengths from a host copy of ``cu_seqlens``: a tuple / list of ints or a CPU integer tensor."""
    if isinstance(cu_seqlens_host, torch.Tensor):
        if cu_seqlens_host.is_cuda:
            raise ValueError("cu_seqlens_cpu must live on the host")
        values = cu_seqlens_host.flatten().tolist()
    else:
        values = list(cu_seqlens_host)
    values = [int(x) for x in values]
    if len(values) < 2:
        raise ValueError("cu_seqlens must have at least two entries")
    if values[0] != 0 or any(b < a for a, b in zip(values[:-1], values[1:], strict=True)):
        raise ValueError(f"cu_seqlens must be non-decreasing and start at 0, got {values[:8]}...")
    return tuple(b - a for a, b in zip(values[:-1], values[1:], strict=True))


@lru_cache(maxsize=256)
def _host_chunk_meta(seq_lens: tuple[int, ...], scan_plans: tuple[ScanPlan, ScanPlan], device_index: int) -> ChunkMeta:
    global _HOST_META_BUILDS
    _HOST_META_BUILDS += 1
    return build_chunk_meta(seq_lens, torch.device("cuda", device_index), scan_plans=scan_plans)


def _host_meta(seq_lens, num_v_heads: int, device_index: int, arch: str, total_tokens: int | None = None) -> ChunkMeta:
    # The scan plans (configuration + segment length) are part of the layout key; ``total_tokens`` is the padded
    # token extent the device table path sees (``T``) so both paths derive identical tables.
    seq_lens = tuple(int(x) for x in seq_lens)
    tokens = int(total_tokens) if total_tokens is not None else sum(seq_lens)
    plans = scan_plans(tokens, len(seq_lens), int(num_v_heads), int(device_index), arch, seq_lens)
    return _host_chunk_meta(seq_lens, plans, int(device_index))


def chunk_meta_from_host(
    cu_seqlens_host, device, num_v_heads: int, total_tokens: int | None = None, arch: str | None = None
) -> ChunkMeta:
    """Exact chunk (and, given ``num_v_heads``, scan segment) tables for a micro-batch layout, built once per
    distinct layout and device.

    The tables are a pure function of the scan plans, the head count and the device, so every layer of a
    micro-batch (and the activation-recompute forward) shares one build: a few tiny int32 tensors, no
    device-to-host synchronisation, no launch, exact launch extents.
    """
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if arch is None:
        arch = device_arch(device)
    return _host_meta(_seq_lens_from_host(cu_seqlens_host), int(num_v_heads), int(index), arch, total_tokens)


def segments_upper_bound(num_chunks_max: int, num_seqs: int, segment_len: int) -> int:
    """Launch extent of a device-built segment table: ``sum(ceil(n_i / L)) <= num_chunks_max // L + num_seqs``."""
    return num_seqs + (num_chunks_max // int(segment_len) if int(segment_len) > 0 else 0)


def build_chunk_meta_from_cu_seqlens(
    cu_seqlens: torch.Tensor, total_tokens: int, arch: str, *, scan_plans: tuple[ScanPlan, ScanPlan]
) -> ChunkMeta:
    """Chunk and segment tables from a device ``cu_seqlens`` tensor without synchronising the host.

    The launch extents are upper bounds (``total_tokens // 64 + num_seqs`` chunks, :func:`segments_upper_bound`
    segments); the padding entries are skipped by every kernel.
    """
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor with at least two entries")
    num_seqs = int(cu_seqlens.numel()) - 1
    num_chunks_max = total_tokens // CHUNK + num_seqs
    seg_max = [segments_upper_bound(num_chunks_max, num_seqs, plan.segment_len) for plan in scan_plans]
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    cu32 = cu32.contiguous()
    device = cu_seqlens.device
    sizes = [num_chunks_max, num_chunks_max, num_seqs + 1]
    for n in seg_max:
        sizes.extend((n + 1, n, num_seqs + 1))
    table = torch.empty(sum(sizes), dtype=torch.int32, device=device)
    views = []
    offset = 0
    for n in sizes:
        views.append(table[offset : offset + n])
        offset += n
    kernel("meta", arch).launch(
        grid=(1, 1, 1),
        cu_seqlens=cu32,
        chunk_start=views[0],
        chunk_len=views[1],
        seq_chunk_start=views[2],
        fwd_seg_chunk_start=views[3],
        fwd_seg_seq=views[4],
        fwd_seq_seg_start=views[5],
        bwd_seg_chunk_start=views[6],
        bwd_seg_seq=views[7],
        bwd_seq_seg_start=views[8],
        num_seqs=num_seqs,
        num_chunks_max=num_chunks_max,
        total_tokens=int(total_tokens),
        fwd_segment_len=int(scan_plans[0].segment_len),
        bwd_segment_len=int(scan_plans[1].segment_len),
        fwd_num_segments_max=seg_max[0],
        bwd_num_segments_max=seg_max[1],
    )
    segments = [
        SegmentTable(
            seg_chunk_start=views[3 + 3 * i],
            seg_seq=views[4 + 3 * i],
            seq_seg_start=views[5 + 3 * i],
            num_segments=seg_max[i],
            segment_len=int(scan_plans[i].segment_len),
            scan_config=tuple(scan_plans[i].config),
        )
        for i in range(2)
    ]
    return ChunkMeta(
        chunk_start=views[0],
        chunk_len=views[1],
        seq_chunk_start=views[2],
        num_chunks=num_chunks_max,
        num_seqs=num_seqs,
        seq_lens=None,
        fwd_segments=segments[0],
        bwd_segments=segments[1],
    )


# --------------------------------------------------------------------------- #
# scan configuration (a pure function of shape and device)
# --------------------------------------------------------------------------- #
@cache
def _device_props(device_index: int) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return int(props.multi_processor_count), int(props.shared_memory_per_block_optin), int(props.major)


def _scan_record(kind: str, vb: int, ks: int, arch: str) -> dict:
    return MODULES[f"{kind}_r{vb}k{ks}"][arch]


def choose_register_scan_config(
    kind: str, num_seqs: int, num_v_heads: int, device_index: int, arch: str
) -> tuple[int, int]:
    """Register-scan (value rows per CTA, K split) whose grid is fully co-resident on the device.

    The preferred configuration (32 rows, K split in two -> four warps) has the shortest per-chunk
    chain and fills all four SM sub-partitions; wider blocks are used only when the preferred grid
    would need more than one wave.  ``MILES_GDN_CHUNK_TRAIN_SCAN_CFG=vb,ksplit`` (or :data:`FORCED_SCAN_CONFIG`)
    pins a configuration for experiments and tests.  The choice depends only on shape and device, never on data;
    it is the baseline the segment planner (:func:`choose_scan_plan`) prices every alternative against.
    """
    if FORCED_SCAN_CONFIG is not None:
        return int(FORCED_SCAN_CONFIG[0]), int(FORCED_SCAN_CONFIG[1])
    forced = os.environ.get("MILES_GDN_CHUNK_TRAIN_SCAN_CFG")
    if forced:
        vb, ks = (int(x) for x in forced.split(","))
        return vb, ks
    return _choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, arch)


@cache
def _choose_register_scan_config(kind: str, num_seqs: int, num_v_heads: int, device_index: int, arch: str):
    sm_count, smem_optin, _major = _device_props(device_index)
    for vb, ks in REG_SCAN_CONFIGS:
        ctas = num_seqs * num_v_heads * (HEAD_DIM // vb)
        threads = 32 * (vb // 16) * ks
        smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
        resident = min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4)
        if resident >= 1 and ctas <= sm_count * resident:
            return vb, ks
    return 64, 1


def _scan_stage(kind: str, config: tuple[int, int], device: torch.device, arch: str) -> tuple[str, int]:
    """Stage name and value block of the ``kind`` register scan in ``config`` (the planned configuration)."""
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    vb, ks = config
    _sm_count, smem_optin, major = _device_props(device_index)
    record = _scan_record(kind, vb, ks, arch)
    if major < 9 or int(record["scan"]["select_smem_bytes"]) > smem_optin:
        raise NotImplementedError(
            f"the register-resident {kind} scan does not fit device {device_index} (only Blackwell is supported)"
        )
    return f"{kind}_r{vb}k{ks}", vb


# --------------------------------------------------------------------------- #
# segment-parallel state scans (a pure function of the layout bound and the device)
# --------------------------------------------------------------------------- #
PRESCAN_CONFIGS = ((32, 2), (64, 1))  # the register-scan configurations with pre-scan stages (loom: PRESCAN_CONFIGS)
MIN_SEGMENT_CHUNKS = 4  # 256 tokens: shorter segments pay more in pre-scan + stitch than they save
# Cost model of the split, in chunk steps of the (32,2) register scan at one CTA per SM, re-fit on the round-R3 GB300
# forced-L sweep (1x8192 and 1x16384 tokens at HV 12, L = 13/22/26/43/64/86, CUPTI per kernel; fwd_h 1.23 us and dhu
# 1.58 us per serial chunk step): a second co-resident CTA raises the per-step time to ~1.60x (192/240/288 CTAs on 160
# SMs measured 1.54/1.60/1.63x; the step is latency-bound but two CTAs share the SM's issue slots), a tail wave behind
# a full wave runs at the same factor (480/576 CTAs measured 3.40/3.21x), and a third round (960 CTAs, 6.77x) is
# outside the model (MAX_SCAN_ROUNDS).  With the gate-adaptive warm-up (round R3, C1) the pre-scan runs only the
# trailing (forward) / leading (backward) chunks whose summed gate drops below the warm-up threshold: ~3.8 chunk steps
# per consumed segment = PRESCAN_WARMUP_CHUNKS x PRESCAN_STEP_FACTOR (1.16: one more operand tile) on the value blocks
# of the S-1 consumed segments (the identity blocks exit).  The stitch launch costs 3.4 us + 2.8 us per boundary (both
# directions, both lengths) = STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS x (max k - 1) fwd_h steps; a boundary without a
# map is a copy of ``B``.  The exact fallback (weak gating: full-segment pre-scan, identity blocks, 3xBF16 128^3 product
# + tile rebuild) is not priced.
# Mirror of ``loom/examples/weave/gdn_chunk_training.py`` (same constants, same functions): both sides must agree so the
# host table and the device table split identically.
SCAN_CO_RESIDENT_FACTOR = 1.60
PRESCAN_STEP_FACTOR = 1.16
PRESCAN_WARMUP_CHUNKS = 3.3  # modelled warm-up length in chunk steps (strong gating: n_w = 1 plus the prologue and launch floor)
STITCH_VSPLIT = 4  # stitch CTAs per (sequence, head): V-column blocks of the [K][V] start state (grid z; mirrors the kernel constant)
STITCH_FIXED_CHUNKS = 2.75  # stitch launch floor in chunk steps
STITCH_COPY_CHUNKS = 2.3  # one copy-only stitch step (serial over the boundaries) in chunk steps
SPLIT_GAIN_THRESHOLD = 0.95  # split only when the modelled makespan beats the serial scan by at least 5 %
MAX_SCAN_ROUNDS = 2  # a split may launch at most two rounds of the resident set (the sweep's calibrated range)
# Per-chunk step time of each register-scan configuration relative to the preferred (32, 2) block in the same residency
# class (round R3, C6): the (64, 1) block halves the grid (one CTA per 64 value rows) at a longer per-chunk chain (16
# instead of 8 key blocks per warp, no K-split reduction); (32, 1) keeps the (32, 2) grid and doubles the chain.  Scales
# the scan and pre-scan terms of a plan, not the stitch.  Initial values from the round-R2 per-configuration step times;
# re-fit per architecture from the pinned-configuration sweep (tactic: calibrated_segment_planner_...).
SCAN_STEP_FACTOR = {(32, 2): 1.0, (32, 1): 1.18, (64, 1): 1.27}
FORCED_SEGMENT_LEN: int | None = None  # tests pin the segment length (0 = serial); production derives it
# Warm-up threshold of the pre-scans in log2: the trailing (forward) / leading (backward) chunks of a segment whose
# summed gate is below it bound the segment's map ``|M| <= 2^thresh`` and the dropped earlier contributions to ``B``
# to the same relative size; -10 nats is FlashQLA's warm-up rule (cp_fwd.py at 0.1.2+7c7dfe1).
# ``MILES_GDN_CHUNK_TRAIN_WARMUP_LOG2`` overrides it for experiments (``-inf``: the exact path on every segment).
WARMUP_THRESH_LOG2 = -10.0 / math.log(2.0)


def warmup_thresh_log2() -> float:
    """The pre-scan warm-up threshold (log2), :data:`WARMUP_THRESH_LOG2` unless ``MILES_GDN_CHUNK_TRAIN_WARMUP_LOG2`` is set."""
    forced = os.environ.get("MILES_GDN_CHUNK_TRAIN_WARMUP_LOG2")
    return float(forced) if forced else WARMUP_THRESH_LOG2


def _scan_residency(kind: str, vb: int, ks: int, device_index: int, arch: str) -> int:
    _sm_count, smem_optin, _major = _device_props(device_index)
    threads = 32 * (vb // 16) * ks
    smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
    return max(1, min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4))


def _wave_factor(ctas: int, sm_count: int, residency: int) -> float:
    """Makespan factor of ``ctas`` concurrently alive latency-bound scan CTAs relative to one CTA per SM.

    Co-resident CTAs (at most two per SM, when the scan's shared memory allows it) finish together at
    :data:`SCAN_CO_RESIDENT_FACTOR` x the single-CTA step time; CTAs beyond the resident set run as further waves.  A
    tail wave is priced at 1.0 only when it is the sole wave and fits one CTA per SM: behind a full wave it starts
    staggered as resident CTAs finish and shares SMs with them, so it runs at the full wave's factor (480 / 576 CTAs on
    160 SMs measured 3.40 / 3.21x against 2.55 / 3.10x under the round-R2 rule).
    """
    co_resident = residency >= 2
    per_wave = sm_count * (2 if co_resident else 1)
    full_factor = SCAN_CO_RESIDENT_FACTOR if co_resident else 1.0
    full, tail = divmod(ctas, per_wave)
    cost = full * full_factor
    if tail:
        cost += full_factor if (full or tail > sm_count) else 1.0
    return cost


def _scan_makespan(chunks: Sequence[int], splits: Sequence[int], per_segment: int, sm_count: int, residency: int) -> float:
    """Modelled makespan, in chunk steps of one CTA per SM, of a scan grid over ``chunks`` cut into ``splits`` segments.

    Sequence ``i`` runs ``splits[i] x per_segment`` CTAs for ``ceil(chunks[i] / splits[i])`` steps (its segments are
    equal-sized within one chunk).  The first ``sm_count x residency`` CTAs are resident from the start, the shortest
    first (the grid interleaves the sequences' segments, so every sequence is represented in the first wave and the
    short ones free their slots first); every further CTA starts when a resident CTA finishes and is priced at the
    longest duration.  Whenever more than one CTA per SM is launched the whole makespan runs at
    :data:`SCAN_CO_RESIDENT_FACTOR`: the dispatch order, not the durations, decides which SMs hold two CTAs, so a
    short sequence leaving early does not un-double the SMs that hold two long CTAs (the real 8481 + 223 micro-batch:
    192 CTAs with a 4-chunk pad measured 1.53x over the full 45 steps; its six-segment grid of 336 CTAs ran at
    1.6 x (4 + 23) steps, the 16 CTAs beyond the resident set starting when the pad's CTAs finished).  One sequence
    reduces to ``steps x _wave_factor(all CTAs)`` within :data:`MAX_SCAN_ROUNDS`.
    """
    runs = sorted((-(-n // k), k * per_segment) for n, k in zip(chunks, splits))
    slots = sm_count * (2 if residency >= 2 else 1)
    longest = runs[-1][0]
    releases = []
    capacity, waiting = slots, 0
    for steps, ctas in runs:
        resident = min(ctas, capacity)
        capacity -= resident
        if resident:
            releases.append((steps, resident))
        waiting += ctas - resident
    makespan = longest
    while waiting > 0 and releases:
        next_releases = []
        for at, count in releases:
            if waiting <= 0:
                break
            started = min(count, waiting)
            waiting -= started
            makespan = max(makespan, at + longest)
            next_releases.append((at + longest, started))
        releases = next_releases
    factor = SCAN_CO_RESIDENT_FACTOR if (residency >= 2 and sum(c for _s, c in runs) > sm_count) else 1.0
    return factor * makespan


def bound_layout_chunks(chunks_bound: int, num_seqs: int) -> tuple[int, ...]:
    """Per-sequence chunk counts of the layout a padded chunk bound allows with the longest serial chain.

    One sequence holds every chunk but one per other sequence; this is all the device table builder (no host copy of
    ``cu_seqlens``) can price.
    """
    return (chunks_bound - (num_seqs - 1),) + (1,) * (num_seqs - 1)


def _serial_cost(chunks: Sequence[int], per_segment: int, sm_count: int, residency: int, step_factor: float) -> float:
    """Modelled makespan of the serial scan (one segment per sequence) in chunk steps of the (32, 2) scan."""
    return step_factor * _scan_makespan(chunks, (1,) * len(chunks), per_segment, sm_count, residency)


def _best_split(chunks: Sequence[int], per_segment: int, sm_count: int, residency: int, step_factor: float) -> tuple[int, float]:
    """``(segment_len, modelled cost)`` of the cheapest admissible split of ``chunks`` (``(0, inf)`` when none is).

    A segment length ``L`` cuts sequence ``i`` into ``k_i = ceil(n_i / L)`` equal-sized segments (``k_i = 1``: unsplit)
    and costs ``f x makespan(k) + f x 3.3 x 1.16 x waves(sum (k_i - 1) x per_segment) + 2.75 + 2.3 x (max k_i - 1)``
    chunk steps for the configuration's step factor ``f`` -- the pass-2 scan over every sequence's segments
    (:func:`_scan_makespan`), the warm-up pre-scan of the consumed segments' value blocks, the stitch launch floor plus
    one serial copy-only step per boundary.  Candidates whose grid exceeds :data:`MAX_SCAN_ROUNDS` rounds of the
    resident set lie outside the calibration and are skipped.
    """
    longest = max(chunks)
    per_wave = sm_count * (2 if residency >= 2 else 1)
    best_len, best_cost = 0, math.inf
    for segments in range(2, longest // MIN_SEGMENT_CHUNKS + 1):
        seg_len = -(-longest // segments)
        if seg_len < MIN_SEGMENT_CHUNKS or seg_len >= longest:
            continue
        splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
        if sum(splits) * per_segment > MAX_SCAN_ROUNDS * per_wave:
            continue
        scan = step_factor * _scan_makespan(chunks, splits, per_segment, sm_count, residency)
        pre = PRESCAN_WARMUP_CHUNKS * PRESCAN_STEP_FACTOR * step_factor * _wave_factor(sum(k - 1 for k in splits) * per_segment, sm_count, residency)
        cost = scan + pre + STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
        if cost < best_cost:
            best_len, best_cost = seg_len, cost
    return best_len, best_cost


def plan_segments(
    chunks_bound: int,
    num_seqs: int,
    per_segment: int,
    sm_count: int,
    residency: int,
    seq_chunks: tuple[int, ...] | None = None,
    step_factor: float = 1.0,
) -> tuple[int, float]:
    """``(segment_len, modelled makespan in chunk steps)`` of a layout in one configuration; ``0`` keeps the serial scan.

    Pure function of the layout and the device.  Without ``seq_chunks`` (the device route, which knows only the padded
    chunk bound ``T // 64 + num_seqs``) it prices :func:`bound_layout_chunks`; with ``seq_chunks`` (the host route:
    the exact per-sequence chunk counts) it prices the tables that will actually run.  The cheapest split
    (:func:`_best_split`) is taken when it beats ``0.95 x`` the serial scan's own makespan (:func:`_serial_cost`), both
    at the configuration's ``step_factor``.  Balanced packed layouts whose sequences are all shorter than ``L`` keep one
    segment per sequence (the pre-scan and stitch then launch on the device route and exit immediately).
    """
    chunks = tuple(int(c) for c in seq_chunks) if seq_chunks is not None else bound_layout_chunks(chunks_bound, num_seqs)
    serial = _serial_cost(chunks, per_segment, sm_count, residency, step_factor)
    best_len, best_cost = _best_split(chunks, per_segment, sm_count, residency, step_factor)
    if best_len == 0 or best_cost >= SPLIT_GAIN_THRESHOLD * serial:
        return 0, serial
    return best_len, best_cost


def _scan_fits(kind: str, config: tuple[int, int], device_index: int, arch: str) -> str | None:
    """``None`` when the ``kind`` register scan in ``config`` is rendered for ``arch`` and fits the device, else why not."""
    vb, ks = config
    stage = f"{kind}_r{vb}k{ks}"
    _sm_count, smem_optin, major = _device_props(device_index)
    if stage not in MODULES or arch not in MODULES[stage]:
        return f"no {stage} stage is rendered for {arch}"
    if major < 9:
        return f"compute capability {major}.x has no register scan"
    smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
    if smem > smem_optin:
        return f"the {kind} register scan {config} needs {smem} B of shared memory, above the device opt-in {smem_optin} B"
    return None


def _prescan_fits(kind: str, config: tuple[int, int], device_index: int, arch: str) -> str | None:
    """``None`` when ``config`` can be split for ``kind`` on this device (pre-scan stage rendered, scan fits), else why not."""
    if config not in PRESCAN_CONFIGS:
        return f"its register-scan configuration {config} has no pre-scan stage (only {PRESCAN_CONFIGS} have one)"
    pre = f"{kind}_pre_r{config[0]}k{config[1]}"
    if pre not in MODULES or arch not in MODULES[pre]:
        return f"no {pre} stage is rendered for {arch}"
    return _scan_fits(kind, config, device_index, arch)


def choose_scan_plan(
    kind: str,
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None = None,
) -> ScanPlan:
    """Register-scan configuration and chunks per segment of the ``kind`` state scan of a layout.

    The baseline is :func:`choose_register_scan_config` run serially (one segment per sequence).  Every register
    configuration rendered for the device is priced serially at its :data:`SCAN_STEP_FACTOR`, and the configurations
    with pre-scan stages (:data:`PRESCAN_CONFIGS`) also at their cheapest split (:func:`_best_split`); the exact
    per-sequence chunk counts are priced when ``seq_lens`` is known (host route), the padded chunk bound
    ``T // 64 + num_seqs`` otherwise (device route).  The cheapest plan is taken when it beats ``0.95 x`` the baseline
    (one rule for a configuration switch and for a split); otherwise the baseline runs.
    ``MILES_GDN_CHUNK_TRAIN_SCAN_CFG`` / :data:`FORCED_SCAN_CONFIG` pin the configuration (the segment length is then
    planned for it); ``MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS`` / :data:`FORCED_SEGMENT_LEN` pin the segment length of the
    baseline (or pinned) configuration, validated at plan time: a pin on a configuration without a pre-scan stage is
    dropped with a :class:`RuntimeWarning` naming the layout, and the serial scan runs.
    """
    env = os.environ.get("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS")
    forced_len = int(FORCED_SEGMENT_LEN) if FORCED_SEGMENT_LEN is not None else (int(env) if env else None)
    baseline = choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, arch)
    if forced_len is not None:
        unsplittable = _prescan_fits(kind, baseline, device_index, arch)
        if forced_len > 0 and unsplittable is not None:
            warnings.warn(
                f"forced segment length {forced_len} ignored for the {kind} scan of this layout ({num_seqs} sequences, "
                f"{num_v_heads} value heads, {total_tokens} tokens): {unsplittable}; the serial scan runs",
                RuntimeWarning,
                stacklevel=2,
            )
            return ScanPlan(baseline, 0)
        return ScanPlan(baseline, forced_len)
    if _scan_fits(kind, baseline, device_index, arch) is not None:
        return ScanPlan(baseline, 0)  # the launch reports the unsupported device (_scan_stage)
    pinned = FORCED_SCAN_CONFIG is not None or bool(os.environ.get("MILES_GDN_CHUNK_TRAIN_SCAN_CFG"))
    chunks_bound = total_tokens // CHUNK + num_seqs
    chunks = tuple(-(-int(x) // CHUNK) for x in seq_lens) if seq_lens is not None else bound_layout_chunks(chunks_bound, num_seqs)
    sm_count, _smem_optin, _major = _device_props(device_index)
    candidates = [baseline] if pinned else [config for config in REG_SCAN_CONFIGS if _scan_fits(kind, config, device_index, arch) is None]
    baseline_cost = _serial_cost(chunks, num_v_heads * (HEAD_DIM // baseline[0]), sm_count, _scan_residency(kind, baseline[0], baseline[1], device_index, arch), SCAN_STEP_FACTOR[baseline])
    best = (baseline, 0, baseline_cost)
    for config in candidates:
        per_segment = num_v_heads * (HEAD_DIM // config[0])
        residency = _scan_residency(kind, config[0], config[1], device_index, arch)
        options = [(0, _serial_cost(chunks, per_segment, sm_count, residency, SCAN_STEP_FACTOR[config]))]
        if _prescan_fits(kind, config, device_index, arch) is None:
            options.append(_best_split(chunks, per_segment, sm_count, residency, SCAN_STEP_FACTOR[config]))
        for seg_len, cost in options:
            if cost < best[2]:
                best = (config, seg_len, cost)
    config, seg_len, cost = best
    if (config, seg_len) != (baseline, 0) and cost >= SPLIT_GAIN_THRESHOLD * baseline_cost:
        return ScanPlan(baseline, 0)
    return ScanPlan(config, seg_len)


def _plan_pins() -> tuple:
    """The test / experiment pins a plan depends on besides its arguments (part of the :func:`scan_plans` memo key)."""
    return (
        None if FORCED_SEGMENT_LEN is None else int(FORCED_SEGMENT_LEN),
        os.environ.get("MILES_GDN_CHUNK_TRAIN_SEG_CHUNKS"),
        None if FORCED_SCAN_CONFIG is None else (int(FORCED_SCAN_CONFIG[0]), int(FORCED_SCAN_CONFIG[1])),
        os.environ.get("MILES_GDN_CHUNK_TRAIN_SCAN_CFG"),
    )


@lru_cache(maxsize=4096)
def _scan_plans_memo(
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None,
    device_props: tuple[int, int, int],
    pins: tuple,
) -> tuple[ScanPlan, ScanPlan]:
    # ``device_props`` and ``pins`` are key material only: the plan reads the same values through the helpers
    return (
        choose_scan_plan("fwd_h", total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens),
        choose_scan_plan("dhu", total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens),
    )


def scan_plans(
    total_tokens: int, num_seqs: int, num_v_heads: int, device_index: int, arch: str, seq_lens: tuple[int, ...] | None = None
) -> tuple[ScanPlan, ScanPlan]:
    """``(forward scan, backward scan)`` plans of a layout (see :func:`choose_scan_plan`).

    Memoised per layout, device (index and properties) and pin state, like loom's ``scan_plans``: the host route
    (``chunk_meta_from_host``) plans on every call and the joint search costs ~60 us of host time per call; a
    repeated layout is a dictionary hit, a pin change always re-plans.
    """
    seq_key = None if seq_lens is None else tuple(int(x) for x in seq_lens)
    return _scan_plans_memo(int(total_tokens), int(num_seqs), int(num_v_heads), int(device_index), str(arch), seq_key, _device_props(int(device_index)), _plan_pins())


def segment_lens(
    total_tokens: int, num_seqs: int, num_v_heads: int, device_index: int, arch: str, seq_lens: tuple[int, ...] | None = None
) -> tuple[int, int]:
    """The two plans' segment lengths (:func:`scan_plans`); the per-case record of the benchmark and DET harnesses."""
    return tuple(plan.segment_len for plan in scan_plans(total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens))


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _launch_wy(arch, kn, v, g_cs, beta32, A, meta: ChunkMeta, num_heads: int, num_v_heads: int, *, recompute: bool):
    w = torch.empty(kn.shape[0], num_v_heads, HEAD_DIM, dtype=kn.dtype, device=kn.device)
    u = torch.empty_like(v)
    kernel("wy", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        kn=kn,
        v=v,
        g_cs=g_cs,
        beta=beta32,
        A_out=A,
        w_out=w,
        u_out=u,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        recompute=1 if recompute else 0,
    )
    return w, u


def _launch_fwd_h(
    arch, kn, w, u, g_cs, initial_state, meta: ChunkMeta, num_heads: int, num_v_heads: int, *, store_final_state: bool
):
    device = kn.device
    h = torch.empty(meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=kn.dtype, device=device)
    v_new = torch.empty_like(u)
    # The kernel writes every element of ``final_state`` when requested; no fill kernel is needed.
    final_state = torch.empty(
        meta.num_seqs if store_final_state else 0, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
    )
    h0 = initial_state.contiguous().float() if initial_state is not None else final_state
    use_init = 1 if initial_state is not None else 0
    segs = meta.fwd_segments
    stage, vb = _scan_stage("fwd_h", segs.scan_config, device, arch)  # the configuration the table was planned for
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # Segment-parallel scan: every segment's affine map in parallel (value and identity column blocks of
        # the augmented state), the serial stitch of the start states, then the ordinary scan per segment
        # from its FP32 start state.
        pre = stage.replace("fwd_h_r", "fwd_h_pre_r")
        if pre not in MODULES or arch not in MODULES[pre]:
            raise NotImplementedError(f"the segmented fwd_h scan needs the pre-scan stage {pre} for {arch}")
        pre_threads = int(MODULES[pre][arch]["block"][0])
        if segs.segment_len > pre_threads:
            raise ValueError(f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks per segment the {pre} warm-up prologue scans")
        seg_state = torch.empty(segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM, dtype=torch.float32, device=device)
        # per-segment flags written by the pre-scan (1: exact fallback, map block present) and read by the stitch
        seg_flag = torch.empty(segs.num_segments, num_v_heads, dtype=torch.int32, device=device)
        kernel(pre, arch).launch(
            grid=(segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb),
            kn=kn,
            w=w,
            u=u,
            g_cs=g_cs,
            h0=h0,
            h_out=h,
            v_new=v_new,
            final_state=final_state,
            seg_state=seg_state,
            chunk_start=meta.chunk_start,
            chunk_len=meta.chunk_len,
            seg_chunk_start=segs.seg_chunk_start,
            seg_seq=segs.seg_seq,
            seq_chunk_start=meta.seq_chunk_start,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            use_initial_state=use_init,
            store_final_state=0,
            seg_flag=seg_flag,
            warmup_thresh_log2=warmup_thresh_log2(),
        )
        h_start = torch.empty(segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
        kernel("stitch", arch).launch(
            grid=(meta.num_seqs, num_v_heads, STITCH_VSPLIT),
            seg_state=seg_state,
            init_state=h0,
            state_out=h_start,
            seq_seg_start=segs.seq_seg_start,
            seg_flag=seg_flag,
            num_v_heads=num_v_heads,
            use_init=use_init,
            reverse=0,
        )
        h0, use_init = h_start, 1
    kernel(stage, arch).launch(
        grid=(segs.num_segments, num_v_heads, HEAD_DIM // vb),
        kn=kn,
        w=w,
        u=u,
        g_cs=g_cs,
        h0=h0,
        h_out=h,
        v_new=v_new,
        final_state=final_state,
        seg_state=final_state,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        seg_chunk_start=segs.seg_chunk_start,
        seg_seq=segs.seg_seq,
        seq_chunk_start=meta.seq_chunk_start,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        use_initial_state=use_init,
        store_final_state=1 if store_final_state else 0,
    )
    return h, v_new, final_state


def gdn_chunk_forward_stage1(
    q, k, v, g, beta, *, meta: ChunkMeta, scale: float | None = None, normalize_qk: bool = True
) -> dict[str, Any]:
    """Prep (l2norm, gate cumsum) and WY (A, w, u); no state is touched yet."""
    total_tokens, num_heads, key_dim = q.shape
    num_v_heads, value_dim = v.shape[1], v.shape[2]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("gdn_chunk_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    arch = device_arch(device)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g32 = g.contiguous().float()
    beta = beta.contiguous()
    # BF16 beta is converted inside the prep kernel (no separate cast kernel); FP32 passes through.
    beta_bf16 = beta.dtype == torch.bfloat16
    beta32 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device) if beta_bf16 else beta.float()

    qn = torch.empty_like(q)
    kn = torch.empty_like(k)
    rstd_q = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=device)
    rstd_k = torch.empty_like(rstd_q)
    g_cs = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("prep_bf16beta" if beta_bf16 else "prep", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        q=q,
        k=k,
        g=g32,
        beta=beta,
        beta32_out=beta32,
        qn=qn,
        kn=kn,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        g_cs=g_cs,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        normalize_qk=1 if normalize_qk else 0,
    )
    A = torch.empty(total_tokens, num_v_heads, CHUNK, dtype=q.dtype, device=device)  # fully written by the WY kernel
    w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, recompute=False)
    return {
        "arch": arch,
        "normalize_qk": normalize_qk,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "qn": qn,
        "kn": kn,
        "rstd_q": rstd_q,
        "rstd_k": rstd_k,
        "g_cs": g_cs,
        "A": A,
        "w": w,
        "u": u,
        "v": v,
        "beta32": beta32,
        "meta": meta,
        "scale": float(scale),
    }


def gdn_chunk_forward_stage2(
    stage1: dict[str, Any], *, initial_state=None, output_final_state: bool = False
) -> dict[str, Any]:
    """State scan and output; extends the stage-1 dict in place."""
    arch = stage1["arch"]
    meta: ChunkMeta = stage1["meta"]
    kn, qn, w, u, g_cs, v = (stage1[name] for name in ("kn", "qn", "w", "u", "g_cs", "v"))
    num_heads, num_v_heads = stage1["num_heads"], stage1["num_v_heads"]
    h, v_new, final_state = _launch_fwd_h(
        arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=output_final_state
    )
    o = torch.empty_like(v)
    kernel("fwd_o", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        v_new=v_new,
        h=h,
        g_cs=g_cs,
        o_out=o,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=float(stage1["scale"]),
    )
    stage1.update(
        {
            "output": o,
            "final_state": final_state if output_final_state else None,
            "initial_state": initial_state,
            "h": h,
            "v_new": v_new,
        }
    )
    return stage1


def gdn_chunk_forward(
    q,
    k,
    v,
    g,
    beta,
    *,
    meta: ChunkMeta,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    normalize_qk: bool = True,
) -> dict[str, Any]:
    """Run the deterministic chunked forward; returns outputs and the tape ``gdn_chunk_backward`` needs."""
    stage1 = gdn_chunk_forward_stage1(q, k, v, g, beta, meta=meta, scale=scale, normalize_qk=normalize_qk)
    return gdn_chunk_forward_stage2(stage1, initial_state=initial_state, output_final_state=output_final_state)


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
def gdn_chunk_backward(
    forward: dict[str, Any], do, dht=None, *, initial_state=None, dht_from_dv_local=None, dbeta_dtype=None
) -> dict[str, Any]:
    """Run the deterministic chunked backward from the (possibly pruned) forward tape.

    ``forward`` must carry ``arch``, ``qn``, ``kn``, ``rstd_q``, ``rstd_k``, ``v``, ``g_cs``, ``beta32``,
    ``A``, ``meta``, ``scale`` and ``normalize_qk``.  ``w``/``u``/``h``/``v_new`` are recomputed exactly
    (same kernels, same order) when the tape does not carry them.  ``initial_state`` is the FP32 state
    the forward started from (``None`` for a zero state); its gradient is returned only when it is
    given.  ``dht_from_dv_local(w, dv_local)`` optionally supplies the final-state gradient after
    ``dv_local`` exists (context-parallel state passing).
    """
    arch = forward["arch"]
    meta: ChunkMeta = forward["meta"]
    qn, kn = forward["qn"], forward["kn"]
    A = forward["A"]
    g_cs = forward["g_cs"]
    scale = float(forward["scale"])
    v = forward["v"]
    beta32 = forward["beta32"]
    total_tokens, num_heads, _ = qn.shape
    num_v_heads = v.shape[1]
    device = qn.device
    do = do.contiguous()
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    if forward.get("w") is None or forward.get("u") is None:
        w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, recompute=True)
    else:
        w, u = forward["w"], forward["u"]
    if forward.get("h") is None or forward.get("v_new") is None:
        h, v_new, _ = _launch_fwd_h(
            arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=False
        )
    else:
        h, v_new = forward["h"], forward["v_new"]

    dv_local = torch.empty_like(v)
    kernel("dv_local", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        do=do,
        g_cs=g_cs,
        dv_local=dv_local,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=scale,
    )

    if dht_from_dv_local is not None:
        dht = dht_from_dv_local(w, dv_local)
    dh = torch.empty_like(h)
    dv2 = torch.empty_like(v)
    # Written in full by the scan when requested; no fill kernel.
    dh0 = torch.empty(
        meta.num_seqs if has_initial_state else 0, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
    )
    dht32 = dht.contiguous().float() if dht is not None else dh0
    use_dht = 1 if dht is not None else 0
    segs = meta.bwd_segments
    scan_stage, vb = _scan_stage("dhu", segs.scan_config, device, arch)  # the configuration the table was planned for
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # Segment-parallel reverse scan: per-segment affine maps, reverse stitch of the segment end states,
        # then the ordinary reverse scan per segment from its FP32 end state.
        pre = scan_stage.replace("dhu_r", "dhu_pre_r")
        if pre not in MODULES or arch not in MODULES[pre]:
            raise NotImplementedError(f"the segmented dhu scan needs the pre-scan stage {pre} for {arch}")
        pre_threads = int(MODULES[pre][arch]["block"][0])
        if segs.segment_len > pre_threads:
            raise ValueError(f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks per segment the {pre} warm-up prologue scans")
        seg_state = torch.empty(segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM, dtype=torch.float32, device=device)
        seg_flag = torch.empty(segs.num_segments, num_v_heads, dtype=torch.int32, device=device)
        kernel(pre, arch).launch(
            grid=(segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb),
            qn=qn,
            kn=kn,
            w=w,
            do=do,
            dv_local=dv_local,
            g_cs=g_cs,
            dht=dht32,
            dh_out=dh,
            dv2=dv2,
            dh0=dh0,
            seg_state=seg_state,
            chunk_start=meta.chunk_start,
            chunk_len=meta.chunk_len,
            seg_chunk_start=segs.seg_chunk_start,
            seg_seq=segs.seg_seq,
            seq_chunk_start=meta.seq_chunk_start,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            use_final_state_grad=use_dht,
            store_initial_state_grad=0,
            scale=scale,
            seg_flag=seg_flag,
            warmup_thresh_log2=warmup_thresh_log2(),
        )
        dh_end = torch.empty(segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
        kernel("stitch", arch).launch(
            grid=(meta.num_seqs, num_v_heads, STITCH_VSPLIT),
            seg_state=seg_state,
            init_state=dht32,
            state_out=dh_end,
            seq_seg_start=segs.seq_seg_start,
            seg_flag=seg_flag,
            num_v_heads=num_v_heads,
            use_init=use_dht,
            reverse=1,
        )
        dht32, use_dht = dh_end, 1
    kernel(scan_stage, arch).launch(
        grid=(segs.num_segments, num_v_heads, HEAD_DIM // vb),
        qn=qn,
        kn=kn,
        w=w,
        do=do,
        dv_local=dv_local,
        g_cs=g_cs,
        dht=dht32,
        dh_out=dh,
        dv2=dv2,
        dh0=dh0,
        seg_state=dh0,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        seg_chunk_start=segs.seg_chunk_start,
        seg_seq=segs.seg_seq,
        seq_chunk_start=meta.seq_chunk_start,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        use_final_state_grad=use_dht,
        store_initial_state_grad=1 if has_initial_state else 0,
        scale=scale,
    )

    dq_hv = torch.empty(total_tokens, num_v_heads, HEAD_DIM, dtype=qn.dtype, device=device)
    dk_hv = torch.empty_like(dq_hv)
    dw = torch.empty_like(w)
    dg1 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("dqkwg", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        v_new=v_new,
        do=do,
        dv2=dv2,
        h=h,
        dh=dh,
        g_cs=g_cs,
        dq_out=dq_hv,
        dk_out=dk_hv,
        dw_out=dw,
        dg_out=dg1,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=scale,
    )

    dk2 = torch.empty_like(dq_hv)
    dv = torch.empty_like(v)
    # ``dbeta`` is produced directly in the caller's dtype (BF16 or FP32); no cast kernel follows.
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    dbeta = torch.empty(
        total_tokens, num_v_heads, dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device
    )
    dg2 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("wy_bwd_bf16" if dbeta_bf16 else "wy_bwd", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        kn=kn,
        v=v,
        A=A,
        dw=dw,
        du=dv2,
        g_cs=g_cs,
        beta=beta32,
        dk2_out=dk2,
        dv_out=dv,
        dbeta_out=dbeta,
        dg2_out=dg2,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
    )

    dq = torch.empty_like(qn)
    dk = torch.empty_like(kn)
    dg = torch.empty_like(dg2)
    kernel("finalize", arch).launch(
        grid=(meta.num_chunks, num_heads, 1),
        qn=qn,
        kn=kn,
        rstd_q=forward["rstd_q"],
        rstd_k=forward["rstd_k"],
        dq_hv=dq_hv,
        dk_hv=dk_hv,
        dk2=dk2,
        dg1=dg1,
        dg2=dg2,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        normalize_qk=1 if forward.get("normalize_qk", True) else 0,
    )
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": dg,
        "dbeta": dbeta,
        "dinitial_state": dh0 if has_initial_state else None,
    }


# --------------------------------------------------------------------------- #
# autograd wrapper with FLA's signature
# --------------------------------------------------------------------------- #
def _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch: int, seq_len: int, device, arch: str, num_v_heads: int) -> ChunkMeta:
    """Chunk and segment tables without a device-to-host synchronisation.

    Equal-length batches and a caller-supplied host copy of ``cu_seqlens`` take the exact tables from the
    per-layout cache (one build per micro-batch layout, shared by every layer); a device-only ``cu_seqlens``
    builds them on the GPU (padded launch extents, one launch per call).
    """
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    total_tokens = int(batch) * int(seq_len)
    if cu_seqlens is None:
        return _host_meta((int(seq_len),) * int(batch), num_v_heads, int(index), arch, total_tokens)
    if batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    if cu_seqlens_cpu is not None:
        return chunk_meta_from_host(cu_seqlens_cpu, device, num_v_heads, total_tokens, arch)
    num_seqs = int(cu_seqlens.numel()) - 1
    return build_chunk_meta_from_cu_seqlens(
        cu_seqlens, total_tokens, arch, scan_plans=scan_plans(total_tokens, num_seqs, int(num_v_heads), int(index), arch)
    )


def _cp_modules():
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
        compress_h0,
        expand_h0,
    )

    return (
        chunk_gated_delta_rule_fwd_h_pre_process,
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        compress_h0,
        expand_h0,
    )


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Deterministic chunked gated delta rule with FLA's autograd contract."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        cp_context: Any,
    ):
        batch, seq_len, num_heads, key_dim = q.shape
        num_v_heads, value_dim = v.shape[2], v.shape[3]
        arch = device_arch(q.device)
        if os.environ.get("MILES_GDN_CHUNK_TRAIN_PREBUILD", "1") == "1":
            prebuild(arch)  # cached: builds all stages once so no stage compiles inside a later step

        def flat(t, heads, dim):
            return t.reshape(batch * seq_len, heads, dim)

        q_flat, k_flat = flat(q, num_heads, key_dim), flat(k, num_heads, key_dim)
        v_flat = flat(v, num_v_heads, value_dim)
        g_flat = g.reshape(batch * seq_len, num_v_heads)
        beta_flat = beta.reshape(batch * seq_len, num_v_heads)
        meta = _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch, seq_len, q.device, arch, num_v_heads)

        use_cp = cp_context is not None and getattr(cp_context, "group", None) is not None
        if use_cp and initial_state is not None:
            raise ValueError("chunk_gated_delta_rule: initial_state must be None under context parallelism")

        # prep + WY first; CP derives this rank's incoming state from w/u
        fwd: dict[str, Any]
        if use_cp:
            fwd_h_pre, _bwd_pre, compress_h0, _expand_h0 = _cp_modules()
            fwd = gdn_chunk_forward_stage1(
                q_flat, k_flat, v_flat, g_flat, beta_flat, meta=meta, scale=scale, normalize_qk=use_qk_l2norm_in_kernel
            )
            initial_state = fwd_h_pre(
                k=fwd["kn"].unsqueeze(0),
                w=fwd["w"].unsqueeze(0),
                u=fwd["u"].unsqueeze(0),
                g=fwd["g_cs"].unsqueeze(0),
                cu_seqlens=cu_seqlens,
                initial_state=None,
                context=cp_context,
                state_v_first=False,
                chunk_size=CHUNK,
            )
            fwd = gdn_chunk_forward_stage2(fwd, initial_state=initial_state, output_final_state=output_final_state)
            saved_state = compress_h0(initial_state, context=cp_context)
        else:
            fwd = gdn_chunk_forward(
                q_flat,
                k_flat,
                v_flat,
                g_flat,
                beta_flat,
                meta=meta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
            saved_state = initial_state

        # The WY / state tensors stay on the tape by default: the backward would otherwise rerun the WY and state-scan
        # kernels (bit-identical either way, ~20 % of the backward's device time at 8k tokens).
        state_tensors = (fwd["w"], fwd["u"], fwd["h"], fwd["v_new"]) if SAVE_STATE else ()
        ctx.save_for_backward(
            fwd["qn"],
            fwd["kn"],
            fwd["rstd_q"],
            fwd["rstd_k"],
            fwd["v"],
            fwd["g_cs"],
            fwd["beta32"],
            fwd["A"],
            saved_state,
            cu_seqlens,
            *state_tensors,
        )
        ctx.saved_state_tensors = bool(state_tensors)
        ctx.meta = meta
        ctx.arch = arch
        ctx.scale = float(scale)
        ctx.shapes = (batch, seq_len, num_heads, num_v_heads, key_dim, value_dim)
        ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
        ctx.cp_context = cp_context if use_cp else None
        ctx.has_initial_state_input = initial_state is not None and not use_cp
        ctx.dtypes = (g.dtype, beta.dtype)
        output = fwd["output"].reshape(batch, seq_len, num_v_heads, value_dim)
        final_state = fwd["final_state"] if output_final_state else None
        return output.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, cu_seqlens, *state_tensors = ctx.saved_tensors
        batch, seq_len, num_heads, num_v_heads, key_dim, value_dim = ctx.shapes
        g_dtype, beta_dtype = ctx.dtypes
        do_flat = do.reshape(batch * seq_len, num_v_heads, value_dim).contiguous()
        tape = {
            "arch": ctx.arch,
            "qn": qn,
            "kn": kn,
            "rstd_q": rstd_q,
            "rstd_k": rstd_k,
            "v": v_flat,
            "g_cs": g_cs,
            "beta32": beta32,
            "A": A,
            "meta": ctx.meta,
            "scale": ctx.scale,
            "normalize_qk": ctx.normalize_qk,
        }
        if ctx.saved_state_tensors:
            tape["w"], tape["u"], tape["h"], tape["v_new"] = state_tensors
        cp_context = ctx.cp_context
        initial_state = saved_state
        dht_hook = None
        if cp_context is not None:
            _fwd_pre, bwd_dhu_pre, _compress_h0, expand_h0 = _cp_modules()
            initial_state = expand_h0(saved_state, context=cp_context)
            if dht is not None:
                raise ValueError("chunk_gated_delta_rule: final-state gradients are not part of the CP contract")

            def dht_hook(w, dv_local):
                dht_local, _ = bwd_dhu_pre(
                    q=qn.unsqueeze(0),
                    k=kn.unsqueeze(0),
                    w=w.unsqueeze(0),
                    do=do_flat.unsqueeze(0),
                    dv=dv_local.unsqueeze(0),
                    g=g_cs.unsqueeze(0),
                    scale=ctx.scale,
                    cu_seqlens=cu_seqlens,
                    dht=None,
                    initial_state=initial_state,
                    context=cp_context,
                    state_v_first=False,
                    chunk_size=CHUNK,
                )
                return dht_local

        grads = gdn_chunk_backward(
            tape, do_flat, dht, initial_state=initial_state, dht_from_dv_local=dht_hook, dbeta_dtype=beta_dtype
        )
        dq = grads["dq"].reshape(batch, seq_len, num_heads, key_dim)
        dk = grads["dk"].reshape(batch, seq_len, num_heads, key_dim)
        dv = grads["dv"].reshape(batch, seq_len, num_v_heads, value_dim)
        dg = grads["dg"].reshape(batch, seq_len, num_v_heads).to(g_dtype)  # no-op for the FP32 gate
        dbeta = grads["dbeta"].reshape(batch, seq_len, num_v_heads).to(beta_dtype)  # no-op: produced in beta's dtype
        dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None, None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | Sequence[int] | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cp_context: Any = None,
    chunk_size: int = 64,
    **unsupported: Any,
):
    """FLA v0.5.2 ``chunk_gated_delta_rule`` signature on the deterministic kernels.

    Supported: ``[B, T, H, K]`` q/k and ``[B, T, HV, V]`` v with ``K = V = 128`` and ``HV % H == 0``,
    BF16 q/k/v, packed varlen via ``cu_seqlens`` (``B = 1``), FP32 ``initial_state`` /
    ``output_final_state`` (V-last ``[N, HV, K, V]``), ``use_qk_l2norm_in_kernel``, and ``cp_context``
    state passing.  Gate fusion (``use_gate_in_kernel``), beta sigmoid fusion, ``state_v_first``,
    ``head_first``, and non-64 chunk sizes are rejected explicitly.

    ``cu_seqlens_cpu`` (the same boundaries on the host: a tuple of ints such as Megatron's
    ``packed_seq_params.cu_seqlens_host``, or a CPU tensor) selects the exact per-layout chunk table that
    every layer of the micro-batch shares; without it the table is built on the device per call.
    """
    for name, value in unsupported.items():
        if value not in (None, False):
            raise NotImplementedError(
                f"chunk_gated_delta_rule: {name}={value!r} is not supported by the deterministic kernels"
            )
    if chunk_size != CHUNK:
        raise NotImplementedError(f"chunk_gated_delta_rule: chunk_size must be {CHUNK}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise NotImplementedError("chunk_gated_delta_rule: q/k/v must be bfloat16")
    if q.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise NotImplementedError("chunk_gated_delta_rule: head dims must be 128")
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if cu_seqlens is not None and cu_seqlens_cpu is None and cp_context is not None:
        cu_seqlens_cpu = getattr(cp_context, "cu_seqlens_cpu", None)
    return ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
    )
