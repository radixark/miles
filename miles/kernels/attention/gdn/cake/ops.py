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

Host-side wiring (the generator's design note, section "Host wiring", has the rationale): the chunk table is a
pure function of the micro-batch's segment lengths, so it is built once per
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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import torch

from ._jit import MODULES, device_arch, kernel, launcher, prebuild

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
# Host-side caches of round R6-7.  ``GDN_HOST_CACHE=0`` disables all of them at import time (the A/B arm): the pool of
# pure-scratch device workspaces (:class:`Workspaces`), the per-call front cache of the chunk-table lookup
# (:func:`_chunk_meta`) and the per-device architecture tag (:func:`_arch_of`).  The positional launchers and the
# shape pass-through are not caches and stay on in both arms.
HOST_CACHE = os.environ.get("GDN_HOST_CACHE", "1") == "1"


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


# --------------------------------------------------------------------------- #
# finalize kernel selection: a pure function of shape and device, mirroring the generator's finalize_stage
# --------------------------------------------------------------------------- #
FINALIZE_MLP_GROUP = (
    3  # value-head group the one-wave finalize is unrolled for (HV = 3 H at every Qwen3.8-27B TP degree)
)
FINALIZE_MLP_MIN_BLOCKS = (
    4  # its launch-bound residency: __launch_bounds__(128, 4) -> 128-register cap -> four CTAs per SM
)
FORCED_FINALIZE: str | None = (
    None  # tests / experiments: "finalize" or "finalize_mlp" (also MILES_GDN_CHUNK_TRAIN_FINALIZE)
)
FINALIZE_STAGES = ("finalize", "finalize_mlp")


def finalize_stage(num_chunks: int, num_heads: int, num_v_heads: int, sm_count: int) -> str:
    """Stage name of the finalize kernel for a layout on a device: ``finalize_mlp`` only where it cannot lose.

    The one-wave kernel needs the value-head group it is unrolled for (``FINALIZE_MLP_GROUP``) and is selected only
    while the (chunks, key heads) grid is co-resident in one wave at the residency its launch bound guarantees
    (``FINALIZE_MLP_MIN_BLOCKS`` CTAs per SM): up to that grid both kernels run every CTA concurrently with the same
    warps per SM, so the one-wave kernel keeps the thread-level parallelism of the shared kernel and adds the
    memory-level parallelism of a whole row pair in flight.  Beyond it the shared 40-register kernel's higher residency
    hides latency across more warps while the one-wave kernel would run several waves, so the shared kernel stays
    (the TP1 / TP2 / 16k-token rows).  ``MILES_GDN_CHUNK_TRAIN_FINALIZE`` (or :data:`FORCED_FINALIZE`) pins a stage for
    experiments and tests; a pinned ``finalize_mlp`` outside its group contract raises instead of launching the kernel.
    """

    if num_heads <= 0 or num_v_heads % num_heads != 0:
        raise ValueError(f"num_v_heads={num_v_heads} must be a positive multiple of num_heads={num_heads}")
    forced = FORCED_FINALIZE if FORCED_FINALIZE is not None else os.environ.get("MILES_GDN_CHUNK_TRAIN_FINALIZE")
    group_ok = num_v_heads == FINALIZE_MLP_GROUP * num_heads
    if forced:
        if forced not in FINALIZE_STAGES:
            raise ValueError(f"MILES_GDN_CHUNK_TRAIN_FINALIZE={forced!r}; expected one of {FINALIZE_STAGES}")
        if forced == "finalize_mlp" and not group_ok:
            raise ValueError(
                f"finalize_mlp is unrolled for a value-head group of {FINALIZE_MLP_GROUP}; "
                f"num_heads={num_heads}, num_v_heads={num_v_heads} is group {num_v_heads // num_heads}"
            )
        return forced
    if group_ok and int(num_chunks) * int(num_heads) <= FINALIZE_MLP_MIN_BLOCKS * int(sm_count):
        return "finalize_mlp"
    return "finalize"


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
PRESCAN_CONFIGS = ((32, 2), (64, 1))  # the register-scan configurations with pre-scan stages (generator: same name)
MIN_SEGMENT_CHUNKS = 4  # 256 tokens: shorter segments pay more in pre-scan + stitch than they save
# Cost model of the split, in chunk steps of the (32,2) register scan at one CTA per SM, re-fit on the round-R3 GB300
# forced-L sweep (1x8192 and 1x16384 tokens at HV 12, L = 13/22/26/43/64/86, CUPTI per kernel; fwd_h 1.23 us and dhu
# 1.58 us per serial chunk step): a second co-resident CTA raises the per-step time to ~1.60x (192/240/288 CTAs on 160
# SMs measured 1.54/1.60/1.63x; the step is latency-bound but two CTAs share the SM's issue slots), a tail wave behind
# a full wave runs at the same factor (480/576 CTAs measured 3.40/3.21x), and a third round (960 CTAs, 6.77x) is
# outside the model (MAX_SCAN_ROUNDS).  At one CTA per SM (every dhu configuration: 140-157 KB of shared memory) there
# is no co-residency: the waves run back to back and ``_scan_makespan`` prices them exactly by slot release, so the
# wave count is quantized -- ceil(CTAs / SMs) full-length waves whatever the fill of the last one (192 serial CTAs on
# 148 / 152 SMs: two waves of 64 steps, the second 26-30 % full; 576 CTAs in 22-chunk segments: four waves of 22
# steps).  Round R6 prices such splits up to MAX_SCAN_WAVES_SERIAL waves (the round-R6 forced-L arms at 3 and
# 4 waves); the co-resident bound stays at MAX_SCAN_ROUNDS rounds.  With the gate-adaptive warm-up (round R3, C1) the pre-scan runs only the
# trailing (forward) / leading (backward) chunks whose summed gate drops below the warm-up threshold: ~3.8 chunk steps
# per consumed segment = PRESCAN_WARMUP_CHUNKS x PRESCAN_STEP_FACTOR (1.16: one more operand tile) on the value blocks
# of the S-1 consumed segments (the identity blocks exit).  The stitch launch costs 3.4 us + 2.8 us per boundary (both
# directions, both lengths) = STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS x (max k - 1) fwd_h steps; a boundary without a
# map is a copy of ``B``.  The exact fallback (weak gating: full-segment pre-scan, identity blocks, 3xBF16 128^3 product
# + tile rebuild) is not priced.
# Mirror of the generator's ``gdn_chunk_training.py`` (same constants, same functions): both sides must agree so the
# host table and the device table split identically.
SCAN_CO_RESIDENT_FACTOR = 1.60
PRESCAN_STEP_FACTOR = 1.16
PRESCAN_WARMUP_CHUNKS = (
    3.3  # modelled warm-up length in chunk steps (strong gating: n_w = 1 plus the prologue and launch floor)
)
STITCH_VSPLIT = 4  # stitch CTAs per (sequence, head): V-column blocks of the [K][V] start state (grid z; mirrors the kernel constant)
STITCH_FIXED_CHUNKS = 2.75  # stitch launch floor in chunk steps
STITCH_COPY_CHUNKS = 2.3  # one copy-only stitch step (serial over the boundaries) in chunk steps
SPLIT_GAIN_THRESHOLD = 0.95  # split only when the modelled makespan beats the serial scan by at least 5 %
MAX_SCAN_ROUNDS = (
    2  # a co-resident split may launch at most two rounds of the resident set (the R3 sweep's calibrated range)
)
MAX_SCAN_WAVES_SERIAL = 4  # a one-CTA-per-SM split may launch at most four sequential waves (round R6 forced-L arms)
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


def _scan_makespan(
    chunks: Sequence[int], splits: Sequence[int], per_segment: int, sm_count: int, residency: int
) -> float:
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
    runs = sorted((-(-n // k), k * per_segment) for n, k in zip(chunks, splits, strict=False))
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


def split_cta_limit(sm_count: int, residency: int) -> int:
    """Largest grid (CTAs) a split may launch on ``sm_count`` SMs: :data:`MAX_SCAN_ROUNDS` rounds of the two-CTA
    resident set when the scan is co-resident, :data:`MAX_SCAN_WAVES_SERIAL` sequential waves at one CTA per SM.  Both
    bounds are four CTAs per SM; they are kept apart because they were calibrated separately (R3 rounds, R6 waves)."""
    return MAX_SCAN_ROUNDS * 2 * sm_count if residency >= 2 else MAX_SCAN_WAVES_SERIAL * sm_count


def _best_split(
    chunks: Sequence[int], per_segment: int, sm_count: int, residency: int, step_factor: float
) -> tuple[int, float]:
    """``(segment_len, modelled cost)`` of the cheapest admissible split of ``chunks`` (``(0, inf)`` when none is).

    A segment length ``L`` cuts sequence ``i`` into ``k_i = ceil(n_i / L)`` equal-sized segments (``k_i = 1``: unsplit)
    and costs ``f x makespan(k) + f x 3.3 x 1.16 x waves(sum (k_i - 1) x per_segment) + 2.75 + 2.3 x (max k_i - 1)``
    chunk steps for the configuration's step factor ``f`` -- the pass-2 scan over every sequence's segments
    (:func:`_scan_makespan`), the warm-up pre-scan of the consumed segments' value blocks, the stitch launch floor plus
    one serial copy-only step per boundary.  Candidates whose grid exceeds :func:`split_cta_limit` (two rounds of the
    resident set when co-resident, four sequential waves at one CTA per SM) lie outside the calibration and are skipped.
    """
    longest = max(chunks)
    limit = split_cta_limit(sm_count, residency)
    best_len, best_cost = 0, math.inf
    for segments in range(2, longest // MIN_SEGMENT_CHUNKS + 1):
        seg_len = -(-longest // segments)
        if seg_len < MIN_SEGMENT_CHUNKS or seg_len >= longest:
            continue
        splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
        if sum(splits) * per_segment > limit:
            continue
        scan = step_factor * _scan_makespan(chunks, splits, per_segment, sm_count, residency)
        pre = (
            PRESCAN_WARMUP_CHUNKS
            * PRESCAN_STEP_FACTOR
            * step_factor
            * _wave_factor(sum(k - 1 for k in splits) * per_segment, sm_count, residency)
        )
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
    segment per sequence (the pre-scan -- and, for the backward scan, the stitch -- then launch on the device route and
    exit immediately).
    """
    chunks = (
        tuple(int(c) for c in seq_chunks) if seq_chunks is not None else bound_layout_chunks(chunks_bound, num_seqs)
    )
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
    chunks = (
        tuple(-(-int(x) // CHUNK) for x in seq_lens)
        if seq_lens is not None
        else bound_layout_chunks(chunks_bound, num_seqs)
    )
    sm_count, _smem_optin, _major = _device_props(device_index)
    candidates = (
        [baseline]
        if pinned
        else [config for config in REG_SCAN_CONFIGS if _scan_fits(kind, config, device_index, arch) is None]
    )
    baseline_cost = _serial_cost(
        chunks,
        num_v_heads * (HEAD_DIM // baseline[0]),
        sm_count,
        _scan_residency(kind, baseline[0], baseline[1], device_index, arch),
        SCAN_STEP_FACTOR[baseline],
    )
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
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None = None,
) -> tuple[ScanPlan, ScanPlan]:
    """``(forward scan, backward scan)`` plans of a layout (see :func:`choose_scan_plan`).

    Memoised per layout, device (index, properties) and pin state, like the generator's ``scan_plans``: the host route
    (``chunk_meta_from_host``) plans on every call and the joint search costs ~60 us of host time per call; a
    repeated layout is a dictionary hit, a pin change always re-plans.
    """
    seq_key = None if seq_lens is None else tuple(int(x) for x in seq_lens)
    return _scan_plans_memo(
        int(total_tokens),
        int(num_seqs),
        int(num_v_heads),
        int(device_index),
        str(arch),
        seq_key,
        _device_props(int(device_index)),
        _plan_pins(),
    )


def segment_lens(
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """The two plans' segment lengths (:func:`scan_plans`); the per-case record of the benchmark and DET harnesses."""
    return tuple(
        plan.segment_len for plan in scan_plans(total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens)
    )


# --------------------------------------------------------------------------- #
# host-side caches (round R6-7): scratch workspaces, chunk-table front cache, device architecture
# --------------------------------------------------------------------------- #
WORKSPACE_SHAPES_PER_ROLE = (
    4  # (shape, dtype) buffers kept per role, device and stream; the oldest is released to the allocator
)


class Workspaces:
    """Pool of pure-scratch device buffers: tensors written and read only by the kernels of ONE forward or ONE backward
    call, never saved for backward, never returned to the caller, never kept in ``ctx`` or in a returned dict.

    Tables are keyed by (device index, CUDA stream handle) and, inside, by role name and exact (shape, dtype).  Reuse is
    stream-ordered: every plugin kernel enqueues on the torch current stream of its tensors' device
    (``csrc/cake_host_shim.h``, ``CurrentStream``), so a later call's writes into a pooled buffer are enqueued on that
    stream after the earlier call's reads of it and execute after them; a call on another stream gets its own table and
    never aliases a buffer the first stream may still be reading.  Forward roles carry the prefix ``f_``, backward roles
    ``b_``: a backward-time recompute of the forward stages (``SAVE_STATE=0``, activation recompute) shares no buffer
    with the backward's own scratch.  Forward (main thread) and backward (autograd worker) never run concurrently in the
    training loop (``backward()`` blocks its caller); two Python threads driving forward passes on the same device and
    stream at once are not supported.  Memory: at most :data:`WORKSPACE_SHAPES_PER_ROLE` buffers per role per table; an
    evicted buffer returns to the caching allocator once its last user is gone.
    """

    __slots__ = ("tables", "hits", "misses")

    def __init__(self):
        self.tables: dict[tuple[int, int], dict[str, dict]] = {}
        self.hits = 0
        self.misses = 0

    def table(self, device_index: int) -> dict:
        """The role table of ``device_index`` on its current stream (one ``current_stream`` query per call)."""
        key = (device_index, torch.cuda.current_stream(device_index).cuda_stream)
        try:
            return self.tables[key]
        except KeyError:
            table = self.tables[key] = {}
            return table

    def take(self, table: dict, role: str, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """The pooled buffer of ``role`` with exactly ``shape`` / ``dtype`` (allocated on a miss)."""
        key = (tuple(shape), dtype)
        try:
            buffers = table[role]
        except KeyError:
            buffers = table[role] = {}
        buffer = buffers.get(key)
        if buffer is None:
            self.misses += 1
            buffer = buffers[key] = torch.empty(shape, dtype=dtype, device=device)
            if len(buffers) > WORKSPACE_SHAPES_PER_ROLE:
                del buffers[next(iter(buffers))]
        else:
            self.hits += 1
        return buffer

    def clear(self):
        self.tables.clear()


class FreshWorkspaces(Workspaces):
    """The ``GDN_HOST_CACHE=0`` arm: every scratch buffer is a fresh ``torch.empty`` (the pre-R6-7 behaviour)."""

    __slots__ = ()

    def table(self, device_index: int) -> dict:
        return {}

    def take(self, table, role, shape, dtype, device):
        self.misses += 1
        return torch.empty(shape, dtype=dtype, device=device)


WORKSPACES: Workspaces = Workspaces() if HOST_CACHE else FreshWorkspaces()

_DEVICE_ARCH: dict[int, str] = {}


def _arch_of(device_index: int) -> str:
    """``device_arch`` of a CUDA device index, memoised per index (``torch.cuda.get_device_capability`` resolves the
    device in Python on every call otherwise)."""
    arch = _DEVICE_ARCH.get(device_index)
    if arch is None:
        arch = device_arch(torch.device("cuda", device_index))
        if HOST_CACHE:
            _DEVICE_ARCH[device_index] = arch
    return arch


_META_FRONT: dict = (
    {}
)  # (layout key, num_v_heads, device index, total_tokens, plan pins) -> ChunkMeta (see _chunk_meta)
_META_FRONT_LIMIT = 256
_META_FRONT_HITS = 0


def host_cache_clear():
    """Drop every R6-7 host-side cache of this module (tests / experiments; the per-layout table memo
    ``_host_chunk_meta`` and the plan memo ``_scan_plans_memo`` are separate ``lru_cache`` objects)."""
    global _META_FRONT_HITS
    WORKSPACES.clear()
    _META_FRONT.clear()
    _DEVICE_ARCH.clear()
    _META_FRONT_HITS = 0


@cache
def _scan_stages(kind: str, config: tuple[int, int], device_index: int, arch: str) -> tuple[str, int, str | None, int]:
    """``(scan stage, value block, pre-scan stage or None, chunks the pre-scan prologue covers)`` of the ``kind``
    register scan in ``config`` on one device -- :func:`_scan_stage` plus the pre-scan lookups, resolved once per key.
    """
    stage, vb = _scan_stage(kind, config, torch.device("cuda", device_index), arch)
    pre = stage.replace(f"{kind}_r", f"{kind}_pre_r")
    if pre not in MODULES or arch not in MODULES[pre]:
        return stage, vb, None, 0
    return stage, vb, pre, int(MODULES[pre][arch]["block"][0])


@cache
def _split_scan_stage(stage: str, arch: str) -> str | None:
    """The split-plan instantiation ``<stage>_split`` of a forward scan stage (the forward stitch folded into its prologue;
    exported for every forward configuration), or ``None`` when this architecture's registry carries none."""
    split = f"{stage}_split"
    return split if split in MODULES and arch in MODULES[split] else None


# Positional launch orders = the exported plan order of every stage (registry ``arg_plan`` without the trailing grid
# entries; identical on sm_100a and sm_103a).  ``_jit.NativeKernel.launcher`` verifies the name set against the plan at
# first use and permutes if an export ever reorders a plan; the values at every call site follow these tuples one to one.
_PREP_ARGS = (
    "q",
    "k",
    "g",
    "beta",
    "beta32_out",
    "qn",
    "kn",
    "rstd_q",
    "rstd_k",
    "g_cs",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "normalize_qk",
)
_WY_ARGS = (
    "kn",
    "v",
    "g_cs",
    "beta",
    "A_out",
    "w_out",
    "u_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "recompute",
)
_FWD_H_ARGS = (
    "kn",
    "w",
    "u",
    "g_cs",
    "h0",
    "h_out",
    "v_new",
    "final_state",
    "seg_state",
    "chunk_start",
    "chunk_len",
    "seg_chunk_start",
    "seg_seq",
    "seq_chunk_start",
    "num_heads",
    "num_v_heads",
    "use_initial_state",
    "store_final_state",
)
# the split-plan instantiation of the forward scan reads the per-segment flags too (its prologue folds the forward
# stitch); the pre-scan adds the warm-up threshold on top
_FWD_H_SPLIT_ARGS = _FWD_H_ARGS + ("seg_flag",)
_FWD_H_PRE_ARGS = _FWD_H_SPLIT_ARGS + ("warmup_thresh_log2",)
_STITCH_ARGS = (
    "seg_state",
    "init_state",
    "state_out",
    "seq_seg_start",
    "seg_flag",
    "num_v_heads",
    "use_init",
    "reverse",
)
_FWD_O_ARGS = (
    "qn",
    "kn",
    "v_new",
    "h",
    "g_cs",
    "o_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "scale",
)
_DV_LOCAL_ARGS = (
    "qn",
    "kn",
    "do",
    "g_cs",
    "dv_local",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "scale",
)
_DHU_ARGS = (
    "qn",
    "kn",
    "w",
    "do",
    "dv_local",
    "g_cs",
    "dht",
    "dh_out",
    "dv2",
    "dh0",
    "seg_state",
    "chunk_start",
    "chunk_len",
    "seg_chunk_start",
    "seg_seq",
    "seq_chunk_start",
    "num_heads",
    "num_v_heads",
    "use_final_state_grad",
    "store_initial_state_grad",
    "scale",
)
_DHU_PRE_ARGS = _DHU_ARGS + ("seg_flag", "warmup_thresh_log2")
_DQKWG_ARGS = (
    "qn",
    "kn",
    "v_new",
    "do",
    "dv2",
    "h",
    "dh",
    "g_cs",
    "dq_out",
    "dk_out",
    "dw_out",
    "dg_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "scale",
)
_WY_BWD_ARGS = (
    "kn",
    "v",
    "A",
    "dw",
    "du",
    "g_cs",
    "beta",
    "dk2_out",
    "dv_out",
    "dbeta_out",
    "dg2_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
)
_FINALIZE_ARGS = (
    "qn",
    "kn",
    "rstd_q",
    "rstd_k",
    "dq_hv",
    "dk_hv",
    "dk2",
    "dg1",
    "dg2",
    "dq_out",
    "dk_out",
    "dg_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "normalize_qk",
)
# stage -> call-site order, for the wiring test that pins every launcher to its exported plan
LAUNCH_ARGS = {
    "prep": _PREP_ARGS,
    "prep_bf16beta": _PREP_ARGS,
    "wy": _WY_ARGS,
    "fwd_o": _FWD_O_ARGS,
    "stitch": _STITCH_ARGS,
    "dv_local": _DV_LOCAL_ARGS,
    "dqkwg": _DQKWG_ARGS,
    "wy_bwd": _WY_BWD_ARGS,
    "wy_bwd_bf16": _WY_BWD_ARGS,
    "finalize": _FINALIZE_ARGS,
    "finalize_mlp": _FINALIZE_ARGS,  # same plan as finalize; the host picks the stage per layout / device (finalize_stage)
    **{f"fwd_h_r{vb}k{ks}": _FWD_H_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"fwd_h_r{vb}k{ks}_split": _FWD_H_SPLIT_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"fwd_h_pre_r{vb}k{ks}": _FWD_H_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
    **{f"dhu_r{vb}k{ks}": _DHU_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"dhu_pre_r{vb}k{ks}": _DHU_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
}


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _launch_wy(
    arch,
    kn,
    v,
    g_cs,
    beta32,
    A,
    meta: ChunkMeta,
    num_heads: int,
    num_v_heads: int,
    total_tokens: int,
    *,
    recompute: bool,
):
    # ``w`` / ``u`` go on the autograd tape (SAVE_STATE) or into the returned forward dict: fresh allocations, never pooled.
    w = torch.empty((total_tokens, num_v_heads, HEAD_DIM), dtype=kn.dtype, device=kn.device)
    u = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    launcher("wy", arch, _WY_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        kn,
        v,
        g_cs,
        beta32,
        A,
        w,
        u,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        1 if recompute else 0,
    )
    return w, u


def _launch_fwd_h(
    arch,
    kn,
    w,
    u,
    g_cs,
    initial_state,
    meta: ChunkMeta,
    num_heads: int,
    num_v_heads: int,
    *,
    store_final_state: bool,
    table: dict,
):
    device = kn.device
    # ``h`` / ``v_new`` are saved for backward (or returned): fresh.  ``final_state`` is returned when requested; otherwise
    # it is the 0-element placeholder the kernels never touch (``store_final_state=0`` / ``use_initial_state=0``), pooled.
    h = torch.empty((meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=kn.dtype, device=device)
    v_new = torch.empty(u.shape, dtype=u.dtype, device=device)
    if store_final_state:  # the kernel writes every element of ``final_state``; no fill kernel is needed
        final_state = torch.empty((meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=device)
    else:
        final_state = WORKSPACES.take(
            table, "f_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32, device
        )
    if initial_state is not None:
        h0 = initial_state.contiguous()
        if h0.dtype != torch.float32:
            h0 = h0.float()
        use_init = 1
    else:
        h0, use_init = final_state, 0
    segs = meta.fwd_segments
    stage, vb, pre, pre_threads = _scan_stages(
        "fwd_h", segs.scan_config, device.index, arch
    )  # the planned configuration
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # Segment-parallel scan: every segment's affine map in parallel (value and identity column blocks of the
        # augmented state), then the split instantiation of the scan per segment, whose prologue derives the segment's
        # FP32 start state from the maps and flags (the forward stitch folded into pass 2: no stitch launch, no
        # start-state scratch).  ``seg_state`` / ``seg_flag`` (pre-scan -> scan) live only inside this call: pooled.
        # The serial plan launches the plain scan below.
        split_stage = _split_scan_stage(stage, arch)
        if pre is None or split_stage is None:
            raise NotImplementedError(
                f"the segmented fwd_h scan needs the stages {stage.replace('fwd_h_r', 'fwd_h_pre_r')} and {stage}_split for {arch}"
            )
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks per segment the {pre} warm-up prologue scans"
            )
        seg_state = WORKSPACES.take(
            table, "f_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM), torch.float32, device
        )
        # per-segment flags written by the pre-scan (1: exact fallback, map block present) and read by the scan's prologue
        seg_flag = WORKSPACES.take(table, "f_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device)
        launcher(pre, arch, _FWD_H_PRE_ARGS)(
            segs.num_segments,
            num_v_heads,
            2 * HEAD_DIM // vb,
            kn,
            w,
            u,
            g_cs,
            h0,
            h,
            v_new,
            final_state,
            seg_state,
            meta.chunk_start,
            meta.chunk_len,
            segs.seg_chunk_start,
            segs.seg_seq,
            meta.seq_chunk_start,
            num_heads,
            num_v_heads,
            use_init,
            0,
            seg_flag,
            warmup_thresh_log2(),
        )
        launcher(split_stage, arch, _FWD_H_SPLIT_ARGS)(
            segs.num_segments,
            num_v_heads,
            HEAD_DIM // vb,
            kn,
            w,
            u,
            g_cs,
            h0,
            h,
            v_new,
            final_state,
            seg_state,
            meta.chunk_start,
            meta.chunk_len,
            segs.seg_chunk_start,
            segs.seg_seq,
            meta.seq_chunk_start,
            num_heads,
            num_v_heads,
            use_init,
            1 if store_final_state else 0,
            seg_flag,
        )
        return h, v_new, final_state
    launcher(stage, arch, _FWD_H_ARGS)(
        segs.num_segments,
        num_v_heads,
        HEAD_DIM // vb,
        kn,
        w,
        u,
        g_cs,
        h0,
        h,
        v_new,
        final_state,
        final_state,
        meta.chunk_start,
        meta.chunk_len,
        segs.seg_chunk_start,
        segs.seg_seq,
        meta.seq_chunk_start,
        num_heads,
        num_v_heads,
        use_init,
        1 if store_final_state else 0,
    )
    return h, v_new, final_state


def gdn_chunk_forward_stage1(
    q, k, v, g, beta, *, meta: ChunkMeta, scale: float | None = None, normalize_qk: bool = True
) -> dict[str, Any]:
    """Prep (l2norm, gate cumsum) and WY (A, w, u); no state is touched yet.

    ``q`` / ``k`` / ``v`` may be ``[T, H, D]`` or the autograd wrapper's ``[B, T, H, D]``: the kernels address tokens
    through the chunk table and never see a tensor shape, so the batch dimension passes through unflattened (no
    ``reshape`` per call) and every per-token output is allocated in the input's own shape.
    """
    num_heads, key_dim = q.shape[-2], q.shape[-1]
    num_v_heads, value_dim = v.shape[-2], v.shape[-1]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("gdn_chunk_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    total_tokens = q.numel() // (num_heads * key_dim)
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = _arch_of(index)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g32 = g.contiguous()
    if g32.dtype != torch.float32:
        g32 = g32.float()
    beta = beta.contiguous()
    # BF16 beta is converted inside the prep kernel (no separate cast kernel); FP32 passes through.
    beta_bf16 = beta.dtype == torch.bfloat16
    if beta_bf16:
        beta32 = torch.empty((total_tokens, num_v_heads), dtype=torch.float32, device=device)
    else:
        beta32 = beta if beta.dtype == torch.float32 else beta.float()

    # every tensor below is saved for backward: fresh allocations (the pool holds scratch only)
    qn = torch.empty(q.shape, dtype=q.dtype, device=device)
    kn = torch.empty(k.shape, dtype=k.dtype, device=device)
    rstd_q = torch.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    rstd_k = torch.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    g_cs = torch.empty((total_tokens, num_v_heads), dtype=torch.float32, device=device)
    launcher("prep_bf16beta" if beta_bf16 else "prep", arch, _PREP_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        q,
        k,
        g32,
        beta,
        beta32,
        qn,
        kn,
        rstd_q,
        rstd_k,
        g_cs,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        1 if normalize_qk else 0,
    )
    A = torch.empty((total_tokens, num_v_heads, CHUNK), dtype=q.dtype, device=device)  # fully written by the WY kernel
    w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, total_tokens, recompute=False)
    return {
        "arch": arch,
        "normalize_qk": normalize_qk,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "total_tokens": total_tokens,
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
    table = WORKSPACES.table(kn.device.index)
    h, v_new, final_state = _launch_fwd_h(
        arch,
        kn,
        w,
        u,
        g_cs,
        initial_state,
        meta,
        num_heads,
        num_v_heads,
        store_final_state=output_final_state,
        table=table,
    )
    o = torch.empty(v.shape, dtype=v.dtype, device=v.device)  # returned: fresh, in v's own shape
    launcher("fwd_o", arch, _FWD_O_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        qn,
        kn,
        v_new,
        h,
        g_cs,
        o,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        float(stage1["scale"]),
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
    ``dv_local`` exists (context-parallel state passing).  ``g_shape`` / ``beta_shape`` in the tape give the
    shapes of the returned ``dg`` / ``dbeta`` (default ``[T, HV]``).

    Scratch of this call (``dv_local``, ``dh``, ``dv2``, the segmented scan's ``seg_state`` / ``seg_flag`` / ``dh_end``,
    ``dq_hv`` / ``dk_hv`` / ``dw`` / ``dg1``, ``dk2`` / ``dg2`` and the 0-element ``dh0`` placeholder) comes from the
    workspace pool; every returned gradient is a fresh allocation.
    """
    arch = forward["arch"]
    meta: ChunkMeta = forward["meta"]
    qn, kn = forward["qn"], forward["kn"]
    A = forward["A"]
    g_cs = forward["g_cs"]
    scale = float(forward["scale"])
    v = forward["v"]
    beta32 = forward["beta32"]
    num_heads = qn.shape[-2]
    num_v_heads = v.shape[-2]
    total_tokens = qn.numel() // (num_heads * qn.shape[-1])
    device = qn.device
    table = WORKSPACES.table(device.index)
    do = do.contiguous()
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    if forward.get("w") is None or forward.get("u") is None:
        w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, total_tokens, recompute=True)
    else:
        w, u = forward["w"], forward["u"]
    if forward.get("h") is None or forward.get("v_new") is None:
        h, v_new, _ = _launch_fwd_h(
            arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=False, table=table
        )
    else:
        h, v_new = forward["h"], forward["v_new"]

    token_hv = (total_tokens, num_v_heads, HEAD_DIM)
    dv_local = WORKSPACES.take(table, "b_dv_local", token_hv, v.dtype, device)
    launcher("dv_local", arch, _DV_LOCAL_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        qn,
        kn,
        do,
        g_cs,
        dv_local,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        scale,
    )

    if dht_from_dv_local is not None:
        dht = dht_from_dv_local(w, dv_local)
    dh = WORKSPACES.take(table, "b_dh", h.shape, h.dtype, device)
    dv2 = WORKSPACES.take(table, "b_dv2", token_hv, v.dtype, device)
    if has_initial_state:  # written in full by the scan when requested (no fill kernel); returned: fresh
        dh0 = torch.empty((meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=device)
    else:
        dh0 = WORKSPACES.take(table, "b_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32, device)
    if dht is not None:
        dht32 = dht.contiguous()
        if dht32.dtype != torch.float32:
            dht32 = dht32.float()
        use_dht = 1
    else:
        dht32, use_dht = dh0, 0
    segs = meta.bwd_segments
    scan_stage, vb, pre, pre_threads = _scan_stages(
        "dhu", segs.scan_config, device.index, arch
    )  # the planned configuration
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # Segment-parallel reverse scan: per-segment affine maps, reverse stitch of the segment end states,
        # then the ordinary reverse scan per segment from its FP32 end state.
        if pre is None:
            raise NotImplementedError(
                f"the segmented dhu scan needs the pre-scan stage {scan_stage.replace('dhu_r', 'dhu_pre_r')} for {arch}"
            )
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks per segment the {pre} warm-up prologue scans"
            )
        seg_state = WORKSPACES.take(
            table, "b_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM), torch.float32, device
        )
        seg_flag = WORKSPACES.take(table, "b_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device)
        launcher(pre, arch, _DHU_PRE_ARGS)(
            segs.num_segments,
            num_v_heads,
            2 * HEAD_DIM // vb,
            qn,
            kn,
            w,
            do,
            dv_local,
            g_cs,
            dht32,
            dh,
            dv2,
            dh0,
            seg_state,
            meta.chunk_start,
            meta.chunk_len,
            segs.seg_chunk_start,
            segs.seg_seq,
            meta.seq_chunk_start,
            num_heads,
            num_v_heads,
            use_dht,
            0,
            scale,
            seg_flag,
            warmup_thresh_log2(),
        )
        dh_end = WORKSPACES.take(
            table, "b_dh_end", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32, device
        )
        launcher("stitch", arch, _STITCH_ARGS)(
            meta.num_seqs,
            num_v_heads,
            STITCH_VSPLIT,
            seg_state,
            dht32,
            dh_end,
            segs.seq_seg_start,
            seg_flag,
            num_v_heads,
            use_dht,
            1,
        )
        dht32, use_dht = dh_end, 1
    launcher(scan_stage, arch, _DHU_ARGS)(
        segs.num_segments,
        num_v_heads,
        HEAD_DIM // vb,
        qn,
        kn,
        w,
        do,
        dv_local,
        g_cs,
        dht32,
        dh,
        dv2,
        dh0,
        dh0,
        meta.chunk_start,
        meta.chunk_len,
        segs.seg_chunk_start,
        segs.seg_seq,
        meta.seq_chunk_start,
        num_heads,
        num_v_heads,
        use_dht,
        1 if has_initial_state else 0,
        scale,
    )

    dq_hv = WORKSPACES.take(table, "b_dq_hv", token_hv, qn.dtype, device)
    dk_hv = WORKSPACES.take(table, "b_dk_hv", token_hv, qn.dtype, device)
    dw = WORKSPACES.take(table, "b_dw", w.shape, w.dtype, device)
    dg1 = WORKSPACES.take(table, "b_dg1", (total_tokens, num_v_heads), torch.float32, device)
    launcher("dqkwg", arch, _DQKWG_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        qn,
        kn,
        v_new,
        do,
        dv2,
        h,
        dh,
        g_cs,
        dq_hv,
        dk_hv,
        dw,
        dg1,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        scale,
    )

    dk2 = WORKSPACES.take(table, "b_dk2", token_hv, qn.dtype, device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=device)  # returned
    # ``dbeta`` is produced directly in the caller's dtype (BF16 or FP32); no cast kernel follows.  Returned: fresh.
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    dbeta = torch.empty(
        forward.get("beta_shape") or (total_tokens, num_v_heads),
        dtype=torch.bfloat16 if dbeta_bf16 else torch.float32,
        device=device,
    )
    dg2 = WORKSPACES.take(table, "b_dg2", (total_tokens, num_v_heads), torch.float32, device)
    launcher("wy_bwd_bf16" if dbeta_bf16 else "wy_bwd", arch, _WY_BWD_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        kn,
        v,
        A,
        dw,
        dv2,
        g_cs,
        beta32,
        dk2,
        dv,
        dbeta,
        dg2,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
    )

    dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)  # returned, in q's own shape
    dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
    dg = torch.empty(forward.get("g_shape") or (total_tokens, num_v_heads), dtype=torch.float32, device=device)
    sm_count, _smem_optin, _major = _device_props(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    launcher(finalize_stage(meta.num_chunks, num_heads, num_v_heads, sm_count), arch, _FINALIZE_ARGS)(
        meta.num_chunks,
        num_heads,
        1,
        qn,
        kn,
        forward["rstd_q"],
        forward["rstd_k"],
        dq_hv,
        dk_hv,
        dk2,
        dg1,
        dg2,
        dq,
        dk,
        dg,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        1 if forward.get("normalize_qk", True) else 0,
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
def _chunk_meta(
    cu_seqlens, cu_seqlens_cpu, batch: int, seq_len: int, device, arch: str, num_v_heads: int
) -> ChunkMeta:
    """Chunk and segment tables without a device-to-host synchronisation.

    Equal-length batches and a caller-supplied host copy of ``cu_seqlens`` take the exact tables from the
    per-layout cache (one build per micro-batch layout, shared by every layer); a device-only ``cu_seqlens``
    builds them on the GPU (padded launch extents, one launch per call).  With :data:`HOST_CACHE` the per-call lookup
    itself is a single dictionary hit keyed by the layout (the ``cu_seqlens_cpu`` tuple, or ``("equal", B, T)``), the
    head count, the device, the padded token extent and the plan pins (:func:`_plan_pins`), so a repeated layout skips
    the boundary validation and the two ``lru_cache`` lookups of :func:`chunk_meta_from_host`.
    """
    global _META_FRONT_HITS
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    total_tokens = int(batch) * int(seq_len)
    if cu_seqlens is None:
        layout = ("equal", int(batch), int(seq_len))
    elif batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    elif cu_seqlens_cpu is None:
        num_seqs = int(cu_seqlens.numel()) - 1
        return build_chunk_meta_from_cu_seqlens(
            cu_seqlens,
            total_tokens,
            arch,
            scan_plans=scan_plans(total_tokens, num_seqs, int(num_v_heads), int(index), arch),
        )
    else:
        layout = (
            cu_seqlens_cpu if type(cu_seqlens_cpu) is tuple else None
        )  # a CPU tensor takes the validated path below
    key = None
    if HOST_CACHE and layout is not None:
        key = (layout, int(num_v_heads), int(index), total_tokens, _plan_pins())
        meta = _META_FRONT.get(key)
        if meta is not None:
            _META_FRONT_HITS += 1
            return meta
    if cu_seqlens is None:
        meta = _host_meta((int(seq_len),) * int(batch), num_v_heads, int(index), arch, total_tokens)
    else:
        meta = chunk_meta_from_host(cu_seqlens_cpu, device, num_v_heads, total_tokens, arch)
    if key is not None:
        if len(_META_FRONT) >= _META_FRONT_LIMIT:
            del _META_FRONT[next(iter(_META_FRONT))]
        _META_FRONT[key] = meta
    return meta


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
        index = q.device.index if q.device.index is not None else torch.cuda.current_device()
        arch = _arch_of(index)
        if os.environ.get("MILES_GDN_CHUNK_TRAIN_PREBUILD", "1") == "1":
            prebuild(arch)  # cached: builds all stages once so no stage compiles inside a later step
        tokens = batch * seq_len
        # ``[B, T, H, D]`` passes through unflattened: the kernels address tokens through the chunk table and never see a
        # shape, and every per-token output / gradient is allocated in its input's own shape (no reshape per call).
        meta = _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch, seq_len, q.device, arch, num_v_heads)

        use_cp = cp_context is not None and getattr(cp_context, "group", None) is not None
        if use_cp and initial_state is not None:
            raise ValueError("chunk_gated_delta_rule: initial_state must be None under context parallelism")

        # prep + WY first; CP derives this rank's incoming state from w/u
        fwd: dict[str, Any]
        if use_cp:
            fwd_h_pre, _bwd_pre, compress_h0, _expand_h0 = _cp_modules()
            fwd = gdn_chunk_forward_stage1(
                q, k, v, g, beta, meta=meta, scale=scale, normalize_qk=use_qk_l2norm_in_kernel
            )
            initial_state = fwd_h_pre(
                k=fwd["kn"].reshape(1, tokens, num_heads, key_dim),
                w=fwd["w"].reshape(1, tokens, num_v_heads, key_dim),
                u=fwd["u"].reshape(1, tokens, num_v_heads, value_dim),
                g=fwd["g_cs"].reshape(1, tokens, num_v_heads),
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
                q,
                k,
                v,
                g,
                beta,
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
        ctx.grad_shapes = (g.shape, beta.shape)
        ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
        ctx.cp_context = cp_context if use_cp else None
        ctx.has_initial_state_input = initial_state is not None and not use_cp
        ctx.dtypes = (g.dtype, beta.dtype)
        output = fwd["output"]  # allocated in v's shape [B, T, HV, V]
        if output.dtype != q.dtype:
            output = output.to(q.dtype)
        final_state = fwd["final_state"] if output_final_state else None
        return output, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, cu_seqlens, *state_tensors = ctx.saved_tensors
        batch, seq_len, num_heads, num_v_heads, key_dim, value_dim = ctx.shapes
        g_shape, beta_shape = ctx.grad_shapes
        g_dtype, beta_dtype = ctx.dtypes
        do = do.contiguous()  # [B, T, HV, V]; a no-op for the usual contiguous incoming gradient
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
            "g_shape": g_shape,
            "beta_shape": beta_shape,
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
            tokens = batch * seq_len

            def dht_hook(w, dv_local):
                dht_local, _ = bwd_dhu_pre(
                    q=qn.reshape(1, tokens, num_heads, key_dim),
                    k=kn.reshape(1, tokens, num_heads, key_dim),
                    w=w.reshape(1, tokens, num_v_heads, key_dim),
                    do=do.reshape(1, tokens, num_v_heads, value_dim),
                    dv=dv_local.reshape(1, tokens, num_v_heads, value_dim),
                    g=g_cs.reshape(1, tokens, num_v_heads),
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
            tape, do, dht, initial_state=initial_state, dht_from_dv_local=dht_hook, dbeta_dtype=beta_dtype
        )
        dq, dk, dv = grads["dq"], grads["dk"], grads["dv"]  # allocated in q / k / v's own shapes
        dg = grads["dg"]  # allocated in g's shape, FP32
        if dg.dtype != g_dtype:
            dg = dg.to(g_dtype)
        dbeta = grads["dbeta"]  # allocated in beta's shape and dtype
        if dbeta.dtype != beta_dtype:
            dbeta = dbeta.to(beta_dtype)
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
